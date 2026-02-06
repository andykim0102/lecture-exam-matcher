# app.py
# ==============================================================================
#  Med-Study OS: 의대생을 위한 스마트 학습 어시스턴트
#  기능: 족보 PDF 분석, 실시간 강의 매칭, 음성 녹음 요약, AI 질의응답
#  업데이트: 미리 분석(Batch Processing) 기능 추가, AI JSON 파싱 강화
# ==============================================================================

import time
import re
import json
import random
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import google.generativeai as genai
from google.api_core import retry, exceptions

# ------------------------------------------------------------------------------
# 1. 페이지 설정 및 디자인 (CSS)
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Med-Study OS",
    layout="wide",
    page_icon="🩺",
    initial_sidebar_state="expanded"
)

# 프리미엄 디자인 CSS
st.markdown("""
<style>
    /* Global Fonts & Colors */
    .stApp { background-color: #f8f9fa; font-family: 'Pretendard', sans-serif; }
    h1, h2, h3, h4, h5, h6, .stMarkdown { color: #2c3e50 !important; }
    
    /* Card Container */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff;
        border: 1px solid #eef2f6;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(200, 210, 230, 0.2);
        margin-bottom: 20px;
        transition: transform 0.2s;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #dee2e6;
        transform: translateY(-2px);
    }

    /* Badges */
    .badge {
        display: inline-flex; align-items: center; justify-content: center;
        padding: 4px 10px; border-radius: 99px; font-size: 0.75rem; 
        font-weight: 700; margin-right: 5px; margin-bottom: 8px;
    }
    .badge-blue { background-color: #e3f2fd; color: #1565c0; border: 1px solid #bbdefb; }
    .badge-red { background-color: #ffebee; color: #c62828; border: 1px solid #ffcdd2; }
    .badge-gray { background-color: #f5f5f5; color: #616161; border: 1px solid #eeeeee; }
    
    /* Typography */
    .q-header { font-size: 1.1rem; font-weight: 800; color: #1a1a1a; margin: 8px 0; }
    .q-body { 
        font-size: 0.95rem; color: #495057; line-height: 1.7; 
        background-color: #fafafa; padding: 15px; border-radius: 10px; 
        border: 1px solid #f1f3f5; white-space: pre-wrap; 
    }
    .dashed-line { border-top: 2px dashed #e0e0e0; margin: 15px 0; width: 100%; height: 0; }

    /* Login Animation */
    .login-logo { font-size: 5rem; animation: bounce 2s infinite; display: inline-block; margin-bottom: 20px; }
    @keyframes bounce { 0%, 100% {transform: translateY(0);} 50% {transform: translateY(-15px);} }

    /* Custom Button */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #007aff 0%, #0062cc 100%);
        border: none; box-shadow: 0 4px 12px rgba(0,122,255,0.2);
        transition: 0.2s;
    }
    div.stButton > button[kind="primary"]:hover { transform: scale(1.02); }
    
    /* Layout */
    .block-container { padding-top: 2rem; max-width: 1200px; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 2. 세션 상태 초기화 (State Management)
# ------------------------------------------------------------------------------
# 기본 상태 변수들
state_defaults = {
    "logged_in": False, "db": [], "api_key": None, "api_key_ok": False,
    "text_models": [], "embedding_models": [], "best_text_model": None, "best_embedding_model": None,
    "lecture_doc": None, "lecture_filename": None, "current_page": 0,
    "edit_target_subject": None, "subject_detail_view": None, "t2_selected_subject": None,
    "transcribed_text": "", "chat_history": [],
    "last_page_sig": None, "last_ai_sig": None, "last_ai_data": None, "last_related": [],
    "tr_res": None,
    # [NEW] 분석 캐시 (페이지별 분석 결과 저장)
    "analysis_cache": {} 
}

for k, v in state_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ------------------------------------------------------------------------------
# 3. 핵심 로직 & AI 함수 (Core Logic)
# ------------------------------------------------------------------------------

def ensure_configured():
    if st.session_state.get("api_key"):
        genai.configure(api_key=st.session_state["api_key"])

@st.cache_data(show_spinner=False)
def list_available_models(api_key: str):
    try:
        genai.configure(api_key=api_key)
        all_models = list(genai.list_models())
        text_mods = [m.name for m in all_models if "generateContent" in getattr(m, "supported_generation_methods", [])]
        embed_mods = [m.name for m in all_models if "embedContent" in getattr(m, "supported_generation_methods", [])]
        return text_mods, embed_mods
    except: return [], []

def get_best_model(models, keywords):
    if not models: return None
    for k in keywords:
        found = [m for m in models if k in m]
        if found: return found[0]
    return models[0]

# [Text Beautifier]
def clean_jokbo_text(text):
    if not text: return ""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'(?m)^(\d+)\.', r'**\1.**', text)
    text = re.sub(r'(?<!\n)(①|②|③|④|⑤|❶|❷|❸|❹|❺|\(1\)|\(2\)|\(3\)|\(4\)|\(5\))', r'\n\1', text)
    text = re.sub(r'(?m)^\d+\s*$', '', text) # 페이지 번호 제거
    return text.strip()

# [Robust Embedding]
def get_embedding_robust(text: str, status_placeholder=None):
    text = (text or "").strip()
    if len(text) < 50: return None, "text_too_short"
    ensure_configured()
    
    if not st.session_state.embedding_models:
        _, embs = list_available_models(st.session_state.api_key)
        st.session_state.embedding_models = embs
    
    candidates = st.session_state.embedding_models
    if not candidates: return None, "No models"
    
    sorted_candidates = sorted(candidates, key=lambda x: 0 if 'text-embedding-004' in x else 1)
    
    for model_name in sorted_candidates[:2]:
        for attempt in range(3):
            try:
                time.sleep(1.0)
                if "004" in model_name:
                    res = genai.embed_content(model=model_name, content=text, task_type="retrieval_document")
                else:
                    res = genai.embed_content(model=model_name, content=text)
                if res and "embedding" in res: return res["embedding"], None
            except Exception as e:
                if "429" in str(e): time.sleep(2 * (attempt + 1))
                elif "404" in str(e): break
                else: time.sleep(1)
    return None, "API Error"

def filter_db_by_subject(subject: str, db: list[dict]):
    if not db: return []
    if subject in ["전체", "ALL", ""]: return db
    return [x for x in db if x.get("subject") == subject]

def find_relevant_jokbo(query_text: str, db: list[dict], top_k: int = 10):
    if not db: return []
    query_emb, _ = get_embedding_robust(query_text)
    if not query_emb: return []
    
    valid_items = [item for item in db if item.get("embedding")]
    if not valid_items: return []
    db_embs = [item["embedding"] for item in valid_items]
    
    sims = cosine_similarity([query_emb], db_embs)[0]
    top_idxs = np.argsort(sims)[::-1][:top_k]
    return [{"score": float(sims[i]), "content": valid_items[i]} for i in top_idxs]

# [Improved AI JSON Generator]
def generate_json_response_robust(prompt: str):
    """
    AI 응답에서 JSON을 확실하게 추출하는 함수.
    마크다운 코드 블록(```json ... ```)을 제거하고 파싱합니다.
    """
    ensure_configured()
    target_model = st.session_state.best_text_model or "gemini-1.5-flash"
    
    try:
        # 1. JSON 모드로 요청
        config = genai.GenerationConfig(temperature=0.3, response_mime_type="application/json")
        model = genai.GenerativeModel(target_model, generation_config=config)
        res = model.generate_content(prompt)
        text = res.text
    except:
        # 2. 실패시 일반 텍스트 모드로 재요청
        try:
            model = genai.GenerativeModel(target_model)
            res = model.generate_content(prompt)
            text = res.text
        except Exception as e:
            return {"explanation": f"AI 오류: {str(e)}", "direction": "분석 불가", "twin_question": "생성 불가"}

    # 3. 텍스트 정제 (마크다운 제거)
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()

    # 4. JSON 파싱 시도
    try:
        # 중괄호 사이의 내용만 추출 시도 (가장 바깥쪽 중괄호)
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            text = match.group(0)
        return json.loads(text)
    except json.JSONDecodeError:
        # 파싱 실패시 원본 텍스트라도 보여줌
        return {
            "explanation": text, 
            "direction": "JSON 파싱 실패 (내용은 '정답/해설'에서 확인하세요)", 
            "twin_question": "형식 오류"
        }

def generate_text_response(prompt: str):
    ensure_configured()
    target_model = st.session_state.best_text_model or "gemini-1.5-flash"
    try:
        model = genai.GenerativeModel(target_model)
        res = model.generate_content(prompt)
        return res.text
    except Exception as e: return f"Error: {e}"

def transcribe_image_to_text(image, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([
            "Extract all text from this image exactly as is. Organize by question number.",
            image
        ])
        return response.text
    except: return None

def transcribe_audio_gemini(audio_bytes, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        res = model.generate_content(["Transcribe accurately.", {"mime_type": "audio/wav", "data": audio_bytes}])
        return res.text
    except: return None

def parse_metadata_from_filename(filename):
    year_match = re.search(r'(20\d{2})', filename)
    year = year_match.group(1) if year_match else ""
    exam_type = "기출"
    if "중간" in filename: exam_type = "중간"
    elif "기말" in filename: exam_type = "기말"
    elif "모의" in filename: exam_type = "모의"
    elif "국시" in filename: exam_type = "국시"
    return f"{year} {exam_type}".strip() or "기출"

# --- Prompts ---
def build_page_analysis_prompt_json(lecture_text, related_jokbo, subject):
    jokbo_ctx = "\n".join([f"- {r['content']['text'][:300]}" for r in related_jokbo[:3]])
    return f"""
    Role: Medical Tutor. 
    Task: Analyze lecture content vs jokbo(exam questions).
    Subject: {subject}
    
    [Jokbo Questions]
    {jokbo_ctx}
    
    [Lecture Page]
    {lecture_text[:1500]}
    
    Output JSON (Korean):
    {{
        "direction": "시험 출제 포인트 1~2문장 (핵심 암기 사항)",
        "twin_question": "족보와 유사한 객관식 변형 문제 1개 (보기 포함)",
        "explanation": "위 변형 문제의 정답 및 상세 해설"
    }}
    """

def build_chat_prompt(hist, ctx, rel, q):
    return f"질문: {q}\n강의: {ctx[:1000]}\n족보: {rel}\n답변해주세요."

def build_transcript_prompt(chunks, packs, subj):
    return f"강의 내용을 족보와 연계하여 요약하세요. 과목: {subj}"

def chunk_transcript(text): return [text[i:i+900] for i in range(0, len(text), 900)]
def extract_text_from_pdf(uploaded_file):
    try:
        return fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
    except: return None

def get_subject_files(subject):
    files = {}
    for item in st.session_state.db:
        if item.get("subject") == subject:
            src = item.get("source", "Unknown")
            files[src] = files.get(src, 0) + 1
    return files

def get_subject_stats():
    stats = {}
    for item in st.session_state.db:
        subj = item.get("subject", "기타")
        if subj not in stats: stats[subj] = {"count": 0}
        stats[subj]["count"] += 1
    return stats

def has_jokbo_evidence(related: list[dict]) -> bool:
    return bool(related) and related[0]["score"] >= 0.70


# ==============================================================================
# 4. 메인 UI (Main App UI)
# ==============================================================================

if not st.session_state.logged_in:
    login()
    st.stop()

# [Sidebar]
with st.sidebar:
    st.markdown("### 👤 내 프로필")
    with st.container(border=True):
        c1, c2 = st.columns([1, 3])
        c1.markdown("## 👨‍⚕️")
        c2.markdown("**Student Admin**\n\n<span style='color:gray; font-size:0.8em'>본과 2학년</span>", unsafe_allow_html=True)
        if st.button("로그아웃", use_container_width=True): logout()

    st.markdown("### 📚 내 학습 과목")
    my_subjects = sorted({x.get("subject", "기타") for x in st.session_state.db})
    if my_subjects:
        for s in my_subjects:
            st.markdown(f"<div style='background:white; padding:10px; border-radius:10px; border:1px solid #eee; margin-bottom:5px; font-weight:600;'>📘 {s}</div>", unsafe_allow_html=True)
    else: st.caption("등록된 과목이 없습니다.")
    st.divider()

    st.markdown("### ⚙️ 설정")
    with st.container(border=True):
        api_key_input = st.text_input("Gemini API Key", type="password", key="api_key_input")
        if api_key_input: st.session_state.api_key = api_key_input.strip()
            
        if st.button("🔄 모델 연결 (필수)", use_container_width=True):
            if not st.session_state.api_key: st.error("API Key 필요")
            else:
                with st.spinner("연결 중..."):
                    t_mods, e_mods = list_available_models(st.session_state.api_key)
                    if t_mods and e_mods:
                        st.session_state.api_key_ok = True
                        st.session_state.text_models = t_mods
                        st.session_state.embedding_models = e_mods
                        st.session_state.best_text_model = get_best_model(t_mods, ["flash", "pro"])
                        st.session_state.best_embedding_model = get_best_model(e_mods, ["text-embedding-004", "004"])
                        st.success(f"연결 성공! ({st.session_state.best_text_model})")
                    else: st.error("모델을 찾을 수 없습니다.")
    
    st.markdown("### 📊 DB 현황")
    with st.container(border=True):
        st.metric("총 학습 페이지", len(st.session_state.db))
        if st.button("DB 초기화"): st.session_state.db = []; st.session_state.analysis_cache = {}; st.rerun()

st.title("Med-Study OS")
tab1, tab2, tab3 = st.tabs(["📂 족보 관리", "📖 강의 분석", "🎙️ 강의 녹음/분석"])

# ------------------------------------------------------------------------------
# TAB 1: 족보 관리 (학습)
# ------------------------------------------------------------------------------
with tab1:
    if st.session_state.subject_detail_view:
        target_subj = st.session_state.subject_detail_view
        c_back, c_title = st.columns([1, 5])
        if c_back.button("← 목록"): st.session_state.subject_detail_view = None; st.rerun()
        c_title.markdown(f"### 📂 {target_subj} 파일 목록")
        st.divider()
        file_map = get_subject_files(target_subj)
        for fname, count in file_map.items():
            meta = parse_metadata_from_filename(fname)
            with st.container(border=True):
                c1, c2 = st.columns([5, 1])
                c1.markdown(f"**📄 {fname}**")
                c1.markdown(f"<span class='badge badge-blue'>{meta}</span>", unsafe_allow_html=True)
                c2.caption(f"{count} pages")
    else:
        col_upload, col_list = st.columns([1, 2])
        with col_upload:
            with st.container(border=True):
                st.markdown("#### ➕ 족보 추가")
                up_subj = st.selectbox("과목", ["해부학", "생리학", "약리학", "직접입력"], key="up_subj")
                final_subj = st.text_input("과목명", placeholder="예: 병리학") if up_subj == "직접입력" else up_subj
                
                files = st.file_uploader("PDF 선택", accept_multiple_files=True, type="pdf")
                
                if st.button("학습 시작", type="primary", use_container_width=True):
                    if not st.session_state.api_key_ok: st.error("왼쪽 설정에서 모델을 연결해주세요.")
                    elif not files: st.warning("파일을 선택해주세요.")
                    else:
                        bar = st.progress(0)
                        log_area = st.empty()
                        logs = []
                        def log(m): logs.append(m); log_area.markdown("\n".join([f"- {l}" for l in logs[-5:]]))
                        
                        new_db = []
                        for i, f in enumerate(files):
                            try:
                                log(f"📂 {f.name} 분석 중...")
                                doc = fitz.open(stream=f.getvalue(), filetype="pdf")
                                for p_idx, page in enumerate(doc):
                                    text = page.get_text().strip()
                                    if len(text) < 50: # OCR 시도
                                        try:
                                            pix = page.get_pixmap()
                                            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                                            ocr_text = transcribe_image_to_text(img, st.session_state.api_key)
                                            if ocr_text: text = ocr_text; log(f"✨ P.{p_idx+1} OCR 완료")
                                        except: pass
                                    
                                    text = clean_jokbo_text(text)
                                    emb, err = get_embedding_robust(text)
                                    if emb:
                                        new_db.append({"page": p_idx+1, "text": text, "source": f.name, "embedding": emb, "subject": final_subj})
                                    elif err != "text_too_short": log(f"❌ P.{p_idx+1} 실패")
                                log(f"✅ {f.name} 완료")
                            except Exception as e: log(f"Err: {e}")
                            bar.progress((i+1)/len(files))
                        
                        if new_db:
                            st.session_state.db.extend(new_db)
                            st.success("학습 완료!"); time.sleep(1); st.rerun()
                        else: st.warning("데이터 없음")

        with col_list:
            st.markdown("#### 📚 학습 데이터")
            stats = get_subject_stats()
            subjects = sorted(stats.keys())
            for i in range(0, len(subjects), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(subjects):
                        s = subjects[i+j]
                        with cols[j]:
                            with st.container(border=True):
                                c1, c2 = st.columns([4, 1])
                                if c1.button(f"## {s}", key=f"v_{s}"): st.session_state.subject_detail_view = s; st.rerun()
                                st.markdown(f"**{stats[s]['count']}** pages")

# ------------------------------------------------------------------------------
# TAB 2: 강의 분석 (미리 분석 + 카드 UI)
# ------------------------------------------------------------------------------
with tab2:
    if st.session_state.t2_selected_subject is None:
        st.info("과목을 선택하세요.")
        stats = get_subject_stats()
        cols = st.columns(3)
        for i, s in enumerate(stats):
            if cols[i%3].button(f"## {s}", key=f"t2_{s}", use_container_width=True):
                st.session_state.t2_selected_subject = s
                st.rerun()
    else:
        target_subj = st.session_state.t2_selected_subject
        c_back, c_head = st.columns([1, 5])
        if c_back.button("← 뒤로"): st.session_state.t2_selected_subject = None; st.rerun()
        c_head.markdown(f"#### 📖 {target_subj} 분석")
        
        with st.expander("📂 강의 PDF 열기", expanded=(st.session_state.lecture_doc is None)):
            l_file = st.file_uploader("PDF", type="pdf", key="t2_f", label_visibility="collapsed")
            if l_file and l_file.name != st.session_state.lecture_filename:
                st.session_state.lecture_doc = fitz.open(stream=l_file.getvalue(), filetype="pdf")
                st.session_state.lecture_filename = l_file.name
                st.session_state.current_page = 0
                st.session_state.analysis_cache = {} # 파일 바뀌면 캐시 초기화
                
            # [NEW] 전체 페이지 미리 분석 버튼 (Batch Processing)
            if st.session_state.lecture_doc:
                if st.button("🚀 전체 페이지 미리 분석하기 (속도 향상)", use_container_width=True):
                    if not st.session_state.api_key_ok: st.error("API Key 필요")
                    else:
                        doc = st.session_state.lecture_doc
                        total = len(doc)
                        bar = st.progress(0)
                        sub_db = filter_db_by_subject(target_subj, st.session_state.db)
                        
                        for idx in range(total):
                            try:
                                page = doc.load_page(idx)
                                txt = page.get_text().strip()
                                # 텍스트 없으면 OCR
                                if len(txt) < 50:
                                    pix = page.get_pixmap()
                                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                                    ocr = transcribe_image_to_text(img, st.session_state.api_key)
                                    if ocr: txt = ocr
                                
                                # 분석 수행 및 캐싱
                                if txt:
                                    rel = find_relevant_jokbo(txt, sub_db, top_k=10)
                                    ai_res = None
                                    if has_jokbo_evidence(rel):
                                        prmt = build_page_analysis_prompt_json(txt, rel, target_subj)
                                        ai_res = generate_json_response_robust(prmt)
                                    
                                    st.session_state.analysis_cache[idx] = {
                                        "text": txt,
                                        "related": rel,
                                        "ai_data": ai_res
                                    }
                            except: pass
                            bar.progress((idx + 1) / total)
                        st.success("분석 완료! 이제 페이지를 넘겨보세요.")

        if st.session_state.lecture_doc:
            doc = st.session_state.lecture_doc
            c_view, c_ai = st.columns([1.5, 1.2])
            
            # [Left] 뷰어
            with c_view:
                with st.container(border=True):
                    c1, c2, c3 = st.columns([1, 2, 1])
                    if c1.button("◀"): st.session_state.current_page = max(0, st.session_state.current_page-1); st.rerun()
                    c2.markdown(f"<div style='text-align:center;'><b>Page {st.session_state.current_page+1}</b></div>", unsafe_allow_html=True)
                    if c3.button("▶"): st.session_state.current_page = min(len(doc)-1, st.session_state.current_page+1); st.rerun()
                    
                    page = doc.load_page(st.session_state.current_page)
                    pix = page.get_pixmap(dpi=150)
                    st.image(Image.frombytes("RGB", [pix.width, pix.height], pix.samples), use_container_width=True)

            # [Right] 분석 결과 (캐시 우선 사용)
            with c_ai:
                ai_tab1, ai_tab2 = st.tabs(["📝 족보 매칭", "💬 질의응답"])
                with ai_tab1:
                    cur_idx = st.session_state.current_page
                    cache_data = st.session_state.analysis_cache.get(cur_idx)
                    
                    p_text = ""
                    rel = []
                    res_ai = {}
                    
                    # 1. 캐시가 있으면 즉시 로드
                    if cache_data:
                        p_text = cache_data["text"]
                        rel = cache_data["related"]
                        res_ai = cache_data["ai_data"] or {}
                    
                    # 2. 캐시가 없으면 실시간 분석 (On-demand)
                    else:
                        page = doc.load_page(cur_idx)
                        p_text = page.get_text().strip()
                        if len(p_text) < 50: # Viewer OCR
                            try:
                                pix = page.get_pixmap()
                                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                                ocr = transcribe_image_to_text(img, st.session_state.api_key)
                                if ocr: p_text = ocr
                            except: pass
                        
                        if p_text:
                            sub_db = filter_db_by_subject(target_subj, st.session_state.db)
                            rel = find_relevant_jokbo(p_text, sub_db, top_k=10)
                            if has_jokbo_evidence(rel) and st.session_state.api_key_ok:
                                with st.spinner("실시간 분석 중..."):
                                    prmt = build_page_analysis_prompt_json(p_text, rel, target_subj)
                                    res_ai = generate_json_response_robust(prmt)
                    
                    # 3. 결과 표시
                    if not p_text:
                        st.info("텍스트를 인식할 수 없습니다.")
                    elif not has_jokbo_evidence(rel):
                        st.info("관련 기출 문제가 없습니다.")
                    else:
                        high_rel_count = len([r for r in rel if r['score'] > 0.82])
                        
                        # 카드 렌더링
                        for i, r in enumerate(rel[:2]):
                            score = r['score']
                            src = r['content'].get('source', 'Unknown')
                            txt = r['content'].get('text', '')
                            txt_clean = clean_jokbo_text(txt)[:400]
                            meta = parse_metadata_from_filename(src)
                            
                            freq_html = ""
                            if i == 0 and high_rel_count >= 2:
                                freq_html = f"<span class='badge badge-red'>🔥 {high_rel_count}회 출제</span>"
                            elif score > 0.88:
                                freq_html = "<span class='badge badge-red'>★ 매우 유사</span>"
                            
                            with st.container(border=True):
                                st.markdown(f"<div><span class='badge badge-blue'>기출</span>{freq_html}<span class='badge badge-gray'>{meta}</span></div>", unsafe_allow_html=True)
                                st.markdown(f"<div class='q-header'>Q. (자동 추출 문항)</div>", unsafe_allow_html=True)
                                st.markdown(f"<div class='q-body'>{txt_clean}...</div>", unsafe_allow_html=True)
                                st.markdown("<div class='dashed-line'></div>", unsafe_allow_html=True)
                                
                                c1, c2, c3 = st.columns(3)
                                with c1:
                                    with st.expander("📝 정답/해설"):
                                        if i == 0: st.write(res_ai.get("explanation", "분석 중..."))
                                        else: st.caption("가장 유사한 문제에서 확인하세요.")
                                with c2:
                                    with st.expander("🎯 출제포인트"):
                                        if i == 0: st.write(res_ai.get("direction", "분석 중..."))
                                        else: st.caption("내용 없음")
                                with c3:
                                    with st.expander("🔄 쌍둥이문제"):
                                        if i == 0: st.info(res_ai.get("twin_question", "분석 중..."))
                                        else: st.caption("내용 없음")
                                
                                with st.expander("🔍 전체 지문 보기"):
                                    st.text(clean_jokbo_text(txt))

                with ai_tab2:
                    for msg in st.session_state.chat_history:
                        with st.chat_message(msg["role"]): st.markdown(msg["content"])
                    if q := st.chat_input("질문..."):
                        if st.session_state.api_key_ok:
                            st.session_state.chat_history.append({"role":"user", "content":q})
                            with st.chat_message("user"): st.markdown(q)
                            with st.chat_message("assistant"):
                                with st.spinner("답변 중..."):
                                    prmt = build_chat_prompt(st.session_state.chat_history, p_text, rel, q)
                                    ans = generate_text_response(prmt)
                                    st.markdown(ans)
                                    st.session_state.chat_history.append({"role":"assistant", "content":ans})

# ------------------------------------------------------------------------------
# TAB 3: 녹음 분석
# ------------------------------------------------------------------------------
with tab3:
    with st.container(border=True):
        st.markdown("#### 🎙️ 강의 녹음/분석")
        c_in, c_out = st.columns(2)
        with c_in:
            sub_t3 = st.selectbox("과목", ["전체"] + sorted({x.get("subject", "") for x in st.session_state.db}), key="t3_s")
            t3_mode = st.radio("입력 방식", ["🎤 녹음", "📂 텍스트"], horizontal=True, label_visibility="collapsed")
            target_text = ""
            
            if t3_mode == "🎤 녹음":
                av = st.audio_input("녹음")
                if av and st.button("분석", key="bm"):
                    if st.session_state.api_key_ok:
                        ts = transcribe_audio_gemini(av.getvalue(), st.session_state.api_key)
                        if ts: st.session_state.transcribed_text = ts; target_text = ts
            else:
                ft = st.file_uploader("파일", type="txt"); at = st.text_area("입력")
                if st.button("분석", key="bt"): target_text = (ft.getvalue().decode() if ft else at).strip()
            
            if target_text:
                if st.session_state.api_key_ok:
                    with st.spinner("분석 중..."):
                        sdb = filter_db_by_subject(sub_t3, st.session_state.db)
                        chks = chunk_transcript(target_text)[:10]
                        rels = [find_relevant_jokbo(c, sdb, top_k=3) for c in chks]
                        res = generate_text_response(build_transcript_prompt(chks, rels, sub_t3))
                        st.session_state.tr_res = res
                    st.success("완료")

        with c_out:
            if "tr_res" in st.session_state and st.session_state.tr_res:
                with st.container(border=True):
                    st.markdown("##### 📝 요약 노트"); st.info(st.session_state.tr_res)
                if st.session_state.transcribed_text:
                    with st.expander("🗣️ 전체 스크립트"): st.text(st.session_state.transcribed_text)
            else: st.info("결과 대기 중...")
