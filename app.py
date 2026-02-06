# app.py (UI: Premium Photo-Like Card / Logic: JSON Mode AI + Smart Text Formatter)
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

# ==========================================
# 0. Page config & Custom CSS
# ==========================================
st.set_page_config(page_title="Med-Study OS", layout="wide", page_icon="🩺")

# [CSS] Premium Design System
st.markdown("""
<style>
    /* 1. Global & Fonts */
    .stApp { background-color: #f8f9fa; } 
    h1, h2, h3, h4, h5, h6, span, div, label, .stMarkdown { 
        color: #2c3e50 !important; 
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, sans-serif;
    }
    .gray-text { color: #8e8e93 !important; }
    
    /* 2. Premium Card Container */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff;
        border: 1px solid #eef2f6;
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(200, 210, 230, 0.25);
        transition: all 0.2s ease-in-out;
        margin-bottom: 20px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(200, 210, 230, 0.4);
        border-color: #dee2e6;
    }

    /* 3. Login Animation */
    .login-logo { 
        font-size: 5rem; 
        margin-bottom: 20px; 
        display: inline-block;
        animation: bounce 2s infinite; 
    }
    @keyframes bounce { 
        0%, 20%, 50%, 80%, 100% {transform: translateY(0);} 
        40% {transform: translateY(-20px);} 
        60% {transform: translateY(-10px);} 
    }

    /* 4. Tab Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: transparent; padding: 4px; }
    .stTabs [data-baseweb="tab"] { 
        height: 45px; border-radius: 20px; padding: 0 24px; 
        background-color: #ffffff; border: 1px solid #e0e0e0; 
        font-weight: 600; color: #8e8e93 !important; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .stTabs [aria-selected="true"] { 
        background-color: #007aff !important; color: #ffffff !important; 
        box-shadow: 0 4px 12px rgba(0,122,255,0.3); border: none; 
    }

    /* 5. Badge Styles */
    .badge {
        display: inline-flex; align-items: center; justify-content: center;
        padding: 5px 12px; border-radius: 99px; font-size: 0.75rem; 
        font-weight: 700; margin-right: 6px; margin-bottom: 8px;
        letter-spacing: -0.3px; transition: 0.2s;
    }
    .badge:hover { transform: scale(1.05); }
    .badge-blue { background-color: #e3f2fd; color: #1565c0; border: 1px solid #bbdefb; }
    .badge-red { background-color: #ffebee; color: #c62828; border: 1px solid #ffcdd2; }
    .badge-gray { background-color: #f5f5f5; color: #616161; border: 1px solid #eeeeee; }
    
    /* 6. Question Content */
    .q-header {
        font-size: 1.1rem; font-weight: 800; color: #1a1a1a !important;
        margin-top: 8px; margin-bottom: 12px; line-height: 1.4;
    }
    .q-body {
        font-size: 0.95rem; color: #495057 !important; line-height: 1.8;
        background-color: #fafafa; padding: 18px; border-radius: 12px;
        margin-bottom: 16px; border: 1px solid #f1f3f5;
        white-space: pre-wrap; /* 줄바꿈 유지 */
        font-family: 'Pretendard', sans-serif;
    }

    /* 7. Separator */
    .dashed-line {
        border-top: 2px dashed #e0e0e0; margin: 20px 0; width: 100%; height: 0;
    }

    /* 8. Expander & Chat */
    .streamlit-expanderHeader {
        font-size: 0.9rem; font-weight: 600; color: #555;
        background-color: #fff; border: 1px solid #e9ecef;
        border-radius: 10px; padding: 10px 16px;
    }
    .stChatMessage { background-color: #ffffff; border: 1px solid #f0f0f0; border-radius: 16px; }
    
    /* 9. Buttons */
    div.stButton > button { border-radius: 12px; font-weight: 600; border: none; height: 3rem; transition: 0.2s; }
    div.stButton > button[kind="primary"] { background: linear-gradient(135deg, #007aff 0%, #0062cc 100%); box-shadow: 0 4px 12px rgba(0,122,255,0.3); }
    div.stButton > button[kind="primary"]:hover { box-shadow: 0 6px 16px rgba(0,122,255,0.4); transform: scale(1.01); }
    div[data-testid="stFileUploader"] { padding: 20px; border: 2px dashed #d1d1d6; border-radius: 16px; background-color: #ffffff; }

    .block-container { padding-top: 2rem; max-width: 1200px; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 1. Session state initialization
# ==========================================
defaults = {
    "logged_in": False, "db": [], "api_key": None, "api_key_ok": False,
    "text_models": [], "embedding_models": [], "best_text_model": None, "best_embedding_model": None,
    "lecture_doc": None, "lecture_filename": None, "current_page": 0,
    "edit_target_subject": None, "subject_detail_view": None, "t2_selected_subject": None,
    "transcribed_text": "", "chat_history": [],
    "last_page_sig": None, "last_ai_sig": None, "last_ai_data": None, "last_related": [],
    "tr_res": None 
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==========================================
# 2. Login Logic
# ==========================================
def login():
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("<div style='height: 15vh;'></div>", unsafe_allow_html=True)
        st.markdown("""<div style="text-align: center;"><div class="login-logo">🩺</div><h1 style="font-weight: 800; color: #1c1c1e;">Med-Study OS</h1><p class="login-desc" style="color: #8e8e93; font-size: 1.1rem;">의대생을 위한 스마트 족보 분석기</p></div>""", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("#### 로그인")
            password = st.text_input("비밀번호", type="password", placeholder="1234")
            if st.button("앱 시작하기", type="primary", use_container_width=True):
                if password == "1234":
                    st.session_state.logged_in = True
                    st.rerun()
                else: st.error("비밀번호: 1234")
            st.markdown("<div style='text-align:center; margin-top:15px; font-size:0.8rem; color:#c7c7cc;'>Demo Access: 1234</div>", unsafe_allow_html=True)

def logout():
    st.session_state.logged_in = False
    st.rerun()


# ==========================================
# 3. Helpers & Data Logic
# ==========================================
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

# [Text Beautifier] - 족보 가독성 향상
def clean_jokbo_text(text):
    if not text: return ""
    # 1. 과도한 공백 제거 (3개 이상의 줄바꿈 -> 2개로)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 2. 문항 번호 강조 (줄 시작 부분의 "1." -> "**1.**")
    text = re.sub(r'(?m)^(\d+)\.', r'**\1.**', text)
    # 3. 보기 가독성 (①, (1), 1) 등이 붙어있으면 줄바꿈)
    text = re.sub(r'(?<!\n)(①|②|③|④|⑤|\(1\)|\(2\)|\(3\)|\(4\)|\(5\))', r'\n\1', text)
    # 4. 불필요한 PDF 메타데이터 제거 (페이지 번호 등 단순 숫자)
    text = re.sub(r'(?m)^\d+\s*$', '', text)
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
    if not candidates: return None, "No embedding models available."
    sorted_candidates = sorted(candidates, key=lambda x: 0 if 'text-embedding-004' in x else 1)
    
    max_retries = 3
    for model_name in sorted_candidates[:2]:
        for attempt in range(max_retries):
            try:
                time.sleep(1.2) 
                if "004" in model_name:
                    res = genai.embed_content(model=model_name, content=text, task_type="retrieval_document")
                else:
                    res = genai.embed_content(model=model_name, content=text)
                if res and "embedding" in res:
                    return res["embedding"], None
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "Resource" in err_msg:
                    time.sleep(2 * (attempt + 1))
                elif "404" in err_msg or "Not Found" in err_msg:
                    break
                else:
                    time.sleep(1)
    return None, f"Fail: API Error"

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

# [AI Generation - JSON Mode Support]
def generate_json_response(prompt: str):
    """JSON 출력을 강제하여 정확한 파싱을 보장합니다."""
    ensure_configured()
    target_model = st.session_state.best_text_model or "gemini-1.5-flash"
    
    try:
        # JSON 모드 설정 (Gemini 1.5 Flash 기능 활용)
        config = genai.GenerationConfig(
            temperature=0.3,
            response_mime_type="application/json"
        )
        model = genai.GenerativeModel(target_model, generation_config=config)
        res = model.generate_content(prompt)
        return json.loads(res.text) # JSON 파싱
    except Exception as e:
        # 실패 시 일반 텍스트로 시도 후 fallback 파싱
        try:
            model = genai.GenerativeModel(target_model)
            res = model.generate_content(prompt)
            # 텍스트에서 JSON 추출 시도
            match = re.search(r'\{.*\}', res.text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return {"explanation": res.text, "direction": "분석 실패", "twin_question": "생성 실패"}
        except:
            return {"explanation": "AI 연결 오류", "direction": "오류", "twin_question": "오류"}

def generate_text_response(prompt: str):
    ensure_configured()
    target_model = st.session_state.best_text_model or "gemini-1.5-flash"
    try:
        model = genai.GenerativeModel(target_model)
        res = model.generate_content(prompt)
        return res.text
    except Exception as e:
        return f"AI 응답 오류: {str(e)}"

# [Enhanced OCR]
def transcribe_image_to_text(image, api_key):
    try:
        genai.configure(api_key=api_key)
        target_model = "gemini-1.5-flash"
        model = genai.GenerativeModel(target_model)
        # OCR 프롬프트 강화: 구조 유지 요청
        response = model.generate_content([
            "Extract all text from this image exactly as is. Preserve the line breaks for each option. Format it like a standard exam paper.",
            image
        ])
        return response.text
    except: return None

def transcribe_audio_gemini(audio_bytes, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([
            "Transcribe audio accurately.",
            {"mime_type": "audio/wav", "data": audio_bytes}
        ])
        return response.text
    except: return None

# [Metadata Parser]
def parse_metadata_from_filename(filename):
    year = ""
    exam_type = ""
    year_match = re.search(r'(20\d{2})', filename)
    if year_match: year = year_match.group(1)
    if "중간" in filename: exam_type = "중간"
    elif "기말" in filename: exam_type = "기말"
    elif "모의" in filename: exam_type = "모의"
    elif "국시" in filename: exam_type = "국시"
    full_meta = f"{year} {exam_type}".strip()
    return full_meta if full_meta else "기출"

# --- Prompts (JSON Schema) ---
def build_page_analysis_prompt_json(lecture_text, related_jokbo, subject):
    jokbo_ctx = "\n".join([f"- {r['content']['text'][:300]}" for r in related_jokbo[:3]])
    return f"""
    You are a medical tutor. Analyze the lecture content and related exam questions (Jokbo).
    Subject: {subject}
    
    [Related Exam Questions]
    {jokbo_ctx}
    
    [Lecture Content]
    {lecture_text[:1500]}
    
    Respond in JSON format with these keys:
    {{
        "direction": "1-2 sentences on what to focus on for exams based on this page.",
        "twin_question": "Create 1 multiple-choice question similar to the exam questions. Include options.",
        "explanation": "Provide the answer and a detailed explanation for the twin question."
    }}
    Important: Write values in Korean.
    """

def build_overview_prompt(txt, subj): return f"과목: {subj}\n내용: {txt[:1500]}\n이 강의의 핵심 목표와 족보 기반 공부 전략 3가지를 요약해줘."
def build_chat_prompt(hist, ctx, rel, q):
    jokbo_ctx = "\n".join([f"- {r['content']['text'][:300]}" for r in rel[:3]])
    return f"질문: {q}\n강의내용: {ctx[:1000]}\n족보: {jokbo_ctx}\n답변해주세요."
def build_transcript_prompt(chunks, packs, subj): return f"강의 녹음 내용을 족보와 연결하여 요약해주세요. 과목: {subj}"

def chunk_transcript(text): return [text[i:i+900] for i in range(0, len(text), 900)]
def extract_text_from_pdf(uploaded_file):
    try:
        data = uploaded_file.getvalue()
        return fitz.open(stream=data, filetype="pdf")
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


# ==========================================
# 4. Main App UI
# ==========================================

if not st.session_state.logged_in:
    login()
    st.stop()

# --- Sidebar ---
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
            st.markdown(f"<div style='background:white; padding:12px; border-radius:12px; border:1px solid #eee; margin-bottom:8px; font-weight:600; color:#333;'>📘 {s}</div>", unsafe_allow_html=True)
    else: st.caption("등록된 과목이 없습니다.")
    st.divider()

    st.markdown("### ⚙️ 설정")
    with st.container(border=True):
        api_key_input = st.text_input("Gemini API Key", type="password", key="api_key_input")
        if api_key_input: st.session_state.api_key = api_key_input.strip()
            
        if st.button("🔄 모델 목록 불러오기 (연결 테스트)", use_container_width=True):
            if not st.session_state.api_key: st.error("API Key 필요")
            else:
                with st.spinner("모델 검색 중..."):
                    t_mods, e_mods = list_available_models(st.session_state.api_key)
                    if t_mods and e_mods:
                        st.session_state.api_key_ok = True
                        st.session_state.text_models = t_mods
                        st.session_state.embedding_models = e_mods
                        st.session_state.best_text_model = get_best_model(t_mods, ["flash", "pro"])
                        st.session_state.best_embedding_model = get_best_model(e_mods, ["text-embedding-004", "004"])
                        st.success(f"✅ 연결 성공! ({st.session_state.best_text_model})")
                    else: st.error("🚫 모델을 찾을 수 없습니다.")
    
    st.markdown("### 📊 DB 현황")
    with st.container(border=True):
        st.metric("총 학습 페이지", len(st.session_state.db))
        if st.button("DB 초기화"): st.session_state.db = []; st.rerun()

# --- Main Content ---
st.title("Med-Study OS")
tab1, tab2, tab3 = st.tabs(["📂 족보 관리", "📖 강의 분석", "🎙️ 강의 녹음/분석"])

# --- TAB 1: 족보 관리 ---
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
                if up_subj == "직접입력":
                    final_subj = st.text_input("과목명 입력", placeholder="예: 병리학") or "기타"
                else: final_subj = up_subj
                
                files = st.file_uploader("PDF 선택", accept_multiple_files=True, type="pdf")
                
                if st.button("학습 시작", type="primary", use_container_width=True):
                    if not st.session_state.api_key_ok: st.error("API Key 설정 필요")
                    elif not files: st.warning("파일 선택 필요")
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
                                    # OCR Fallback
                                    if len(text) < 50:
                                        try:
                                            pix = page.get_pixmap()
                                            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                                            ocr_text = transcribe_image_to_text(img, st.session_state.api_key)
                                            if ocr_text: text = ocr_text; log(f"✨ P.{p_idx+1} OCR 성공")
                                        except: pass
                                    
                                    # [New] Clean Text
                                    text = clean_jokbo_text(text)

                                    emb, err = get_embedding_robust(text)
                                    if emb:
                                        new_db.append({"page": p_idx+1, "text": text, "source": f.name, "embedding": emb, "subject": final_subj})
                                    elif err != "text_too_short": log(f"❌ P.{p_idx+1} 실패")
                                log(f"✅ {f.name} 완료")
                            except Exception as e: log(f"Error: {e}")
                            bar.progress((i+1)/len(files))
                        
                        if new_db:
                            st.session_state.db.extend(new_db)
                            st.success("학습 완료!")
                            time.sleep(1); st.rerun()
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
                                if c2.button("✏️", key=f"e_{s}"): pass
                                st.markdown(f"**{stats[s]['count']}** pages")

# --- TAB 2: 강의 분석 (Premium Card UI) ---
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
                st.session_state.last_page_sig = None

        if st.session_state.lecture_doc:
            doc = st.session_state.lecture_doc
            c_view, c_ai = st.columns([1.5, 1.2])
            
            with c_view:
                with st.container(border=True):
                    c1, c2, c3 = st.columns([1, 2, 1])
                    if c1.button("◀"): st.session_state.current_page = max(0, st.session_state.current_page-1); st.rerun()
                    c2.markdown(f"<div style='text-align:center;'><b>Page {st.session_state.current_page+1}</b></div>", unsafe_allow_html=True)
                    if c3.button("▶"): st.session_state.current_page = min(len(doc)-1, st.session_state.current_page+1); st.rerun()
                    
                    page = doc.load_page(st.session_state.current_page)
                    pix = page.get_pixmap(dpi=150)
                    st.image(Image.frombytes("RGB", [pix.width, pix.height], pix.samples), use_container_width=True)
                    p_text = page.get_text().strip()
                    if len(p_text) < 50:
                        try:
                            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            ocr_res = transcribe_image_to_text(img, st.session_state.api_key)
                            if ocr_res: p_text = ocr_res
                        except: pass

            with c_ai:
                ai_tab1, ai_tab2 = st.tabs(["📝 족보 매칭", "💬 질의응답"])
                with ai_tab1:
                    if not p_text: st.info("텍스트 없음")
                    else:
                        psig = hash(p_text)
                        if psig != st.session_state.last_page_sig:
                            st.session_state.last_page_sig = psig
                            sub_db = filter_db_by_subject(target_subj, st.session_state.db)
                            st.session_state.last_related = find_relevant_jokbo(p_text, sub_db, top_k=10)
                            st.session_state.last_ai_sig = None
                        
                        rel = st.session_state.last_related
                        if has_jokbo_evidence(rel):
                            # AI Analysis with JSON Mode
                            if st.session_state.api_key_ok:
                                aisig = (psig, target_subj)
                                if aisig != st.session_state.last_ai_sig:
                                    with st.spinner("AI 분석 중..."):
                                        prmt = build_page_analysis_prompt_json(p_text, rel, target_subj)
                                        # Use JSON Generator
                                        json_res = generate_json_response(prmt)
                                        st.session_state.last_ai_data = json_res
                                        st.session_state.last_ai_sig = aisig
                            
                            res_ai = st.session_state.last_ai_data or {}
                            high_rel_count = len([r for r in rel if r['score'] > 0.82])
                            
                            for i, r in enumerate(rel[:2]):
                                score = r['score']
                                src = r['content'].get('source', 'Unknown')
                                txt = r['content'].get('text', '')
                                
                                # [New] Clean Text for Display
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
                                    # Use Cleaned Text
                                    st.markdown(f"<div class='q-body'>{txt_clean}...</div>", unsafe_allow_html=True)
                                    st.markdown("<div class='dashed-line'></div>", unsafe_allow_html=True)
                                    
                                    c1, c2, c3 = st.columns(3)
                                    with c1:
                                        with st.expander("📝 정답/해설"):
                                            # Using JSON parsed data
                                            st.write(res_ai.get("explanation", "생성 중...") if i==0 else "AI 해설 미제공")
                                    with c2:
                                        with st.expander("🎯 출제포인트"):
                                            st.write(res_ai.get("direction", "생성 중...") if i==0 else "내용 없음")
                                    with c3:
                                        with st.expander("🔄 쌍둥이문제"):
                                            st.info(res_ai.get("twin_question", "생성 중...") if i==0 else "내용 없음")
                                    
                                    with st.expander("🔍 전체 지문 보기"):
                                        st.text(clean_jokbo_text(txt))
                        else: st.info("관련 기출 없음")

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

# --- TAB 3: 녹음 ---
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
