# app.py (UI: Photo-Like Card Style / Logic: Full Features Restored)
import time
import re
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

# Custom CSS for UI Enhancement
st.markdown("""
<style>
    /* 1. Global Settings */
    .stApp { background-color: #f8f9fa; } 
    h1, h2, h3, h4, h5, h6, p, span, div, label, .stMarkdown { color: #1c1c1e !important; }
    
    /* 2. Custom Badge Styles */
    .badge-base {
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-right: 6px;
        display: inline-block;
        white-space: nowrap;
    }
    .badge-blue { background-color: #e3f2fd; color: #1565c0; border: 1px solid #bbdefb; }
    .badge-red { background-color: #ffebee; color: #c62828; border: 1px solid #ffcdd2; }
    .badge-gray { background-color: #f5f5f5; color: #616161; border: 1px solid #e0e0e0; }
    .badge-green { background-color: #e8f5e9; color: #2e7d32; border: 1px solid #c8e6c9; }
    
    /* 3. Question Text Styles */
    .q-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #212121 !important;
        margin: 12px 0 8px 0;
        line-height: 1.5;
    }
    .q-body {
        font-size: 0.95rem;
        color: #424242 !important;
        background-color: #fafafa;
        padding: 12px;
        border-radius: 8px;
        line-height: 1.6;
        white-space: pre-wrap;
        border: 1px solid #f0f0f0;
    }
    
    /* 4. Dashed Divider */
    .dashed-line {
        border-top: 2px dashed #e0e0e0;
        margin: 16px 0;
    }

    /* 5. Sidebar & Buttons */
    div.stButton > button { border-radius: 10px; font-weight: 600; border: none; box-shadow: none; background-color: #f1f3f5; height: 2.8rem; }
    div.stButton > button:hover { background-color: #e9ecef; transform: scale(0.99); }
    div.stButton > button[kind="primary"] { background-color: #007aff; color: white; }
    div.stButton > button[kind="primary"] p { color: #ffffff !important; }
    
    /* 6. Layout Utils */
    .block-container { padding-top: 1.5rem !important; }
    header[data-testid="stHeader"] { display: none; }
    
    /* 7. Expander Styling Override */
    .streamlit-expanderHeader {
        font-size: 0.9rem;
        font-weight: 600;
        color: #555;
        background-color: #fff;
        border: 1px solid #eee;
        border-radius: 8px;
    }
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
    "last_page_sig": None, "last_ai_sig": None, "last_ai_text": "", "last_related": [],
    "tr_res": None # [Restored] Transcript Result Cache
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
        st.markdown(
            """
            <div style="text-align: center;">
                <div style="font-size: 5rem; margin-bottom: 10px;">🩺</div>
                <h1 style="font-weight: 800; margin-bottom: 0; color: #1c1c1e;">Med-Study OS</h1>
                <p style="color: #8e8e93; margin-bottom: 30px;">스마트한 의대 족보 분석기</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        with st.container(border=True):
            st.markdown("#### 로그인")
            username = st.text_input("아이디", placeholder="admin")
            password = st.text_input("비밀번호", type="password", placeholder="1234")
            
            if st.button("앱 시작하기", type="primary", use_container_width=True):
                if password == "1234":
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("비밀번호가 틀렸습니다. (Demo: 1234)")
            st.markdown("<div style='text-align:center; margin-top:15px; font-size:0.8rem; color:#c7c7cc;'>Demo Access: admin / 1234</div>", unsafe_allow_html=True)

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
    except Exception as e:
        return [], []

def get_best_model(models, keywords):
    if not models: return None
    for k in keywords:
        found = [m for m in models if k in m]
        if found: return found[0]
    return models[0]

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
    
    max_retries = 5
    base_wait = 2
    
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
                if "429" in err_msg or "Resource exhausted" in err_msg:
                    wait_time = base_wait * (2 ** attempt) + random.randint(1, 3)
                    if status_placeholder:
                        status_placeholder.caption(f"⚠️ {model_name} 사용량 초과. {wait_time}초 대기 중...")
                    time.sleep(wait_time)
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

def generate_with_fallback(prompt: str, model_names: list[str]):
    ensure_configured()
    target_model = st.session_state.best_text_model or "gemini-1.5-flash"
    candidates = [target_model]
    if model_names: candidates.extend(model_names)
    candidates = list(dict.fromkeys(candidates))
    
    last_err = None
    config = genai.GenerationConfig(temperature=0.3)
    
    for name in candidates:
        try:
            model = genai.GenerativeModel(name, generation_config=config)
            res = model.generate_content(prompt)
            if res.text: return res.text, name
        except Exception as e: 
            last_err = e
            continue
    raise Exception(f"AI 응답 실패: {str(last_err)}")

def transcribe_image_to_text(image, api_key, model_name=None):
    try:
        genai.configure(api_key=api_key)
        target_model = model_name if model_name else "gemini-1.5-flash"
        model = genai.GenerativeModel(target_model)
        response = model.generate_content([
            "Extract all text from this image exactly as is. Organize by question number if possible.",
            image
        ])
        return response.text
    except Exception:
        return None

def transcribe_audio_gemini(audio_bytes, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([
            "Please transcribe the following audio file into text accurately. Do not add any conversational text, just the transcription.",
            {"mime_type": "audio/wav", "data": audio_bytes}
        ])
        return response.text
    except Exception as e:
        return None

# [Metadata Parser]
def parse_metadata_from_filename(filename):
    """파일명에서 연도, 학기, 시험 종류 추출"""
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

# --- Prompts ---
def build_overview_prompt(first_page_text, subject):
    return f"과목: {subject}\n내용: {first_page_text[:1500]}\n이 강의의 핵심 목표와 족보 기반 공부 전략 3가지를 요약해줘."

def build_page_analysis_prompt(lecture_text, related_jokbo, subject):
    jokbo_ctx = "\n".join([f"- {r['content']['text'][:300]}" for r in related_jokbo[:3]])
    return f"""
    너는 의대 수석 조교다. 학생이 공부 중인 페이지와 관련된 족보를 분석해라.
    과목: {subject}
    
    [관련 족보]
    {jokbo_ctx}
    
    [강의 내용]
    {lecture_text[:1500]}
    
    다음 3가지를 명확히 구분하여 출력하라.
    [SECTION: DIRECTION] 이 페이지에서 시험에 나올만한 핵심 포인트 (1~2문장)
    [SECTION: TWIN_Q] 족보와 유사한 변형 문제 (객관식) 하나 만들기
    [SECTION: EXPLANATION] 위 변형 문제의 정답 및 상세 해설
    """

def build_chat_prompt(history, context_text, related_jokbo, question):
    jokbo_ctx = "\n".join([f"- {r['content']['text'][:300]}" for r in related_jokbo[:3]])
    return f"질문: {question}\n강의내용: {context_text[:1000]}\n족보: {jokbo_ctx}\n답변해주세요."

# [Restored] Transcript Prompt
def build_transcript_prompt(chunks, related_packs, subject):
    packed = ""
    for idx, (chunk, rel) in enumerate(zip(chunks, related_packs), 1):
        if not has_jokbo_evidence(rel): continue
        ctx = "\n".join([f"- {r['content']['text'][:200]}" for r in rel[:2]])
        packed += f"\n(구간 {idx})\n[강의] {chunk}\n[족보근거] {ctx}\n"
    
    if not packed: return "족보와 관련된 특별한 내용은 발견되지 않았습니다. 일반적인 요약입니다."
    
    return f"""
    당신은 의대 조교입니다. 강의 전사 내용을 족보 기반으로 분석하여 요약하세요.
    과목: {subject}
    
    {packed}
    
    출력 형식:
    [족보 적중 노트]
    1. 핵심 주제 (관련 족보 연계)
    2. 교수님이 강조한 내용
    """

def chunk_transcript(text: str, max_chars: int = 900):
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    for p in parts:
        if len(p) <= max_chars: chunks.append(p)
        else:
            for i in range(0, len(p), max_chars):
                chunks.append(p[i:i+max_chars])
    return chunks

def extract_text_from_pdf(uploaded_file):
    try:
        data = uploaded_file.getvalue()
        doc = fitz.open(stream=data, filetype="pdf")
        return doc
    except: return None

# --- Stat Helpers ---
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
        if subj not in stats: stats[subj] = {"count": 0, "last_updated": "방금 전"}
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
            st.markdown(f"<div class='sidebar-subject'>📘 {s}</div>", unsafe_allow_html=True)
    else:
        st.caption("등록된 과목이 없습니다.")
    st.divider()

    st.markdown("### ⚙️ 설정")
    with st.container(border=True):
        api_key_input = st.text_input("Gemini API Key", type="password", key="api_key_input")
        if api_key_input: st.session_state.api_key = api_key_input.strip()
            
        if st.button("🔄 모델 목록 불러오기 (연결 테스트)", use_container_width=True):
            if not st.session_state.api_key:
                st.error("API Key를 입력하세요.")
            else:
                with st.spinner("사용 가능한 모델 찾는 중..."):
                    t_mods, e_mods = list_available_models(st.session_state.api_key)
                    if t_mods and e_mods:
                        st.session_state.api_key_ok = True
                        st.session_state.text_models = t_mods
                        st.session_state.embedding_models = e_mods
                        st.session_state.best_text_model = get_best_model(t_mods, ["flash", "pro"])
                        st.session_state.best_embedding_model = get_best_model(e_mods, ["text-embedding-004", "004"])
                        st.success(f"✅ 연결 성공! ({st.session_state.best_text_model})")
                    else:
                        st.error("🚫 사용 가능한 모델을 찾을 수 없습니다.")
            
    st.markdown("### 📊 DB 현황")
    with st.container(border=True):
        st.metric("총 학습 페이지", len(st.session_state.db))
        if st.button("DB 초기화", use_container_width=True):
            st.session_state.db = []
            st.rerun()

# --- Main Content ---
st.title("Med-Study OS")

tab1, tab2, tab3 = st.tabs(["📂 족보 관리", "📖 강의 분석", "🎙️ 강의 녹음/분석"])

# --- TAB 1: 족보 관리 ---
with tab1:
    if st.session_state.subject_detail_view:
        target_subj = st.session_state.subject_detail_view
        c_back, c_title = st.columns([1, 5])
        with c_back:
            if st.button("← 목록"):
                st.session_state.subject_detail_view = None
                st.rerun()
        with c_title: st.markdown(f"### 📂 {target_subj} 파일 목록")
        st.divider()
        file_map = get_subject_files(target_subj)
        for fname, count in file_map.items():
            meta = parse_metadata_from_filename(fname)
            with st.container(border=True):
                c1, c2 = st.columns([5, 1])
                c1.markdown(f"**📄 {fname}**")
                c1.markdown(f"<span class='badge-base badge-blue'>{meta}</span>", unsafe_allow_html=True)
                c2.caption(f"{count} pages")
    else:
        col_upload, col_list = st.columns([1, 2])
        with col_upload:
            with st.container(border=True):
                st.markdown("#### ➕ 족보 추가")
                up_subj = st.selectbox("과목", ["해부학", "생리학", "약리학", "직접입력"], key="up_subj")
                if up_subj == "직접입력":
                    up_subj_custom = st.text_input("과목명 입력", placeholder="예: 병리학")
                    final_subj = up_subj_custom if up_subj_custom else "기타"
                else: final_subj = up_subj
                
                files = st.file_uploader("PDF 선택", accept_multiple_files=True, type="pdf")
                
                if st.button("학습 시작", type="primary", use_container_width=True):
                    if not st.session_state.api_key_ok: st.error("왼쪽 설정에서 모델 연결을 먼저 해주세요!")
                    elif not files: st.warning("파일을 선택해주세요.")
                    else:
                        prog_bar = st.progress(0)
                        with st.expander("📝 처리 로그", expanded=True):
                            log_c = st.empty()
                            logs = []
                            def log(m):
                                logs.append(m)
                                log_c.markdown("\n".join([f"- {l}" for l in logs[-5:]]))

                            new_db = []
                            for i, f in enumerate(files):
                                try:
                                    log(f"📂 {f.name} 분석 시작...")
                                    doc = fitz.open(stream=f.getvalue(), filetype="pdf")
                                    
                                    for p_idx, page in enumerate(doc):
                                        text = page.get_text().strip()
                                        # OCR Fallback
                                        if len(text) < 50:
                                            try:
                                                pix = page.get_pixmap()
                                                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                                                ocr_text = transcribe_image_to_text(img, st.session_state.api_key, st.session_state.best_text_model)
                                                if ocr_text: text = ocr_text
                                            except: pass

                                        emb, err = get_embedding_robust(text, st.empty())
                                        if emb:
                                            new_db.append({
                                                "page": p_idx + 1, "text": text, "source": f.name,
                                                "embedding": emb, "subject": final_subj
                                            })
                                        elif err != "text_too_short":
                                            log(f"⚠️ P.{p_idx+1} 실패 ({err})")
                                    
                                    log(f"✅ {f.name} 완료")
                                except Exception as e:
                                    log(f"❌ 오류: {str(e)}")
                                prog_bar.progress((i + 1) / len(files))
                            
                            if new_db:
                                st.session_state.db.extend(new_db)
                                st.success(f"{len(new_db)} 페이지 학습 완료!")
                                time.sleep(1)
                                st.rerun()
                            else: st.warning("저장된 데이터 없음")

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
                                if c1.button(f"## {s}", key=f"v_{s}"):
                                    st.session_state.subject_detail_view = s
                                    st.rerun()
                                if c2.button("✏️", key=f"e_{s}"): pass
                                st.markdown(f"**{stats[s]['count']}** pages")

# --- TAB 2: 강의 분석 (Photo-Like Card UI) ---
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
        if c_back.button("← 뒤로"):
            st.session_state.t2_selected_subject = None
            st.rerun()
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
                    if c1.button("◀"): st.session_state.current_page = max(0, st.session_state.current_page-1)
                    c2.markdown(f"<div style='text-align:center;'><b>Page {st.session_state.current_page+1}</b></div>", unsafe_allow_html=True)
                    if c3.button("▶"): st.session_state.current_page = min(len(doc)-1, st.session_state.current_page+1)
                    
                    page = doc.load_page(st.session_state.current_page)
                    pix = page.get_pixmap(dpi=150)
                    st.image(Image.frombytes("RGB", [pix.width, pix.height], pix.samples), use_container_width=True)
                    
                    p_text = page.get_text().strip()
                    if len(p_text) < 50:
                        try:
                            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            with st.spinner("이미지 텍스트 인식 중..."):
                                ocr_res = transcribe_image_to_text(img, st.session_state.api_key, st.session_state.best_text_model)
                                if ocr_res: p_text = ocr_res
                        except: pass

            with c_ai:
                ai_tab1, ai_tab2 = st.tabs(["📝 족보 매칭", "💬 질의응답"])
                
                with ai_tab1:
                    if not p_text:
                        st.info("텍스트를 인식할 수 없습니다.")
                    else:
                        psig = hash(p_text)
                        if psig != st.session_state.last_page_sig:
                            st.session_state.last_page_sig = psig
                            sub_db = filter_db_by_subject(target_subj, st.session_state.db)
                            st.session_state.last_related = find_relevant_jokbo(p_text, sub_db, top_k=10)
                            st.session_state.last_ai_sig = None
                        
                        rel = st.session_state.last_related
                        
                        if has_jokbo_evidence(rel):
                            if st.session_state.api_key_ok:
                                aisig = (psig, target_subj)
                                if aisig != st.session_state.last_ai_sig:
                                    with st.spinner("AI 분석 중..."):
                                        prmt = build_page_analysis_prompt(p_text, rel, target_subj)
                                        raw, _ = generate_with_fallback(prmt, st.session_state.text_models)
                                        parsed = {"DIRECTION": "", "TWIN_Q": "", "EXPLANATION": ""}
                                        parts = raw.split("[SECTION:")
                                        for p in parts:
                                            if "DIRECTION]" in p: parsed["DIRECTION"] = p.replace("DIRECTION]", "").strip()
                                            elif "TWIN_Q]" in p: parsed["TWIN_Q"] = p.replace("TWIN_Q]", "").strip()
                                            elif "EXPLANATION]" in p: parsed["EXPLANATION"] = p.replace("EXPLANATION]", "").strip()
                                        st.session_state.last_ai_text = parsed
                                        st.session_state.last_ai_sig = aisig
                            
                            res_ai = st.session_state.last_ai_text
                            
                            high_rel_count = len([r for r in rel if r['score'] > 0.82])
                            
                            for i, r in enumerate(rel[:2]):
                                score = r['score']
                                src = r['content'].get('source', 'Unknown')
                                txt = r['content'].get('text', '')[:250]
                                meta = parse_metadata_from_filename(src)
                                
                                freq_html = ""
                                if i == 0 and high_rel_count >= 2:
                                    freq_html = f"<span class='badge-base badge-red'>★ 중요 ({high_rel_count}회 출제)</span>"
                                elif score > 0.88:
                                    freq_html = "<span class='badge-base badge-red'>★ 매우 유사</span>"
                                
                                with st.container(border=True):
                                    st.markdown(f"<div><span class='badge-base badge-blue'>기출</span>{freq_html}<span class='badge-base badge-gray'>{meta}</span></div>", unsafe_allow_html=True)
                                    st.markdown(f"<div class='q-title'>Q. 다음 중... (자동요약)</div>", unsafe_allow_html=True)
                                    st.markdown(f"<div class='q-body'>{txt}...</div>", unsafe_allow_html=True)
                                    st.markdown("<div class='dashed-line'></div>", unsafe_allow_html=True)
                                    
                                    c1, c2, c3 = st.columns(3)
                                    with c1:
                                        with st.expander("📝 정답/해설"):
                                            if i == 0 and isinstance(res_ai, dict): st.write(res_ai.get("EXPLANATION", "생성 중..."))
                                            else: st.caption("AI 해설 미제공")
                                    with c2:
                                        with st.expander("🎯 출제포인트"):
                                            if i == 0 and isinstance(res_ai, dict): st.write(res_ai.get("DIRECTION", "생성 중..."))
                                            else: st.caption("내용 없음")
                                    with c3:
                                        with st.expander("🔄 쌍둥이문제"):
                                            if i == 0 and isinstance(res_ai, dict): st.info(res_ai.get("TWIN_Q", "생성 중..."))
                                            else: st.caption("내용 없음")
                        else:
                            st.info("관련 기출 문제가 없습니다.")

                with ai_tab2:
                    for msg in st.session_state.chat_history:
                        with st.chat_message(msg["role"]): st.markdown(msg["content"])
                    if q := st.chat_input("질문 입력..."):
                        if st.session_state.api_key_ok:
                            st.session_state.chat_history.append({"role":"user", "content":q})
                            with st.chat_message("user"): st.markdown(q)
                            with st.chat_message("assistant"):
                                with st.spinner("답변 생성 중..."):
                                    prmt = build_chat_prompt(st.session_state.chat_history, p_text, rel, q)
                                    ans, _ = generate_with_fallback(prmt, st.session_state.text_models)
                                    st.markdown(ans)
                                    st.session_state.chat_history.append({"role":"assistant", "content":ans})

# --- TAB 3: 강의 녹음/분석 (Restored) ---
with tab3:
    with st.container(border=True):
        st.markdown("#### 🎙️ 강의 녹음/분석")
        c_in, c_out = st.columns(2)
        
        with c_in:
            sub_t3 = st.selectbox("과목", ["전체"] + sorted({x.get("subject", "") for x in st.session_state.db}), key="t3_s")
            t3_mode = st.radio("입력 방식", ["🎤 직접 녹음", "📂 파일 업로드 / 텍스트"], horizontal=True, label_visibility="collapsed")
            target_text = ""
            
            if t3_mode == "🎤 직접 녹음":
                audio_value = st.audio_input("녹음 시작")
                if audio_value and st.button("분석 실행", type="primary", key="btn_mic"):
                    if not st.session_state.api_key_ok: st.error("API Key 필요")
                    else:
                        with st.spinner("음성을 텍스트로 변환 중..."):
                            transcript = transcribe_audio_gemini(audio_value.getvalue(), st.session_state.api_key)
                            if transcript:
                                st.session_state.transcribed_text = transcript
                                target_text = transcript
            else:
                f_txt = st.file_uploader("전사 파일(.txt)", type="txt", key="t3_f")
                area_txt = st.text_area("직접 입력", height=150, placeholder="강의 내용을 붙여넣으세요...")
                if st.button("분석 실행", type="primary", key="btn_txt"):
                    target_text = (f_txt.getvalue().decode() if f_txt else area_txt).strip()
            
            if target_text:
                if not st.session_state.api_key_ok: st.error("API Key 필요")
                else:
                    with st.spinner("족보 매칭 및 분석 중..."):
                        sdb = filter_db_by_subject(sub_t3, st.session_state.db)
                        # Chunking & Retrieval
                        chunks = chunk_transcript(target_text)[:10] # Limit to 10 chunks for speed
                        rels = [find_relevant_jokbo(c, sdb, top_k=3) for c in chunks]
                        
                        # Generate RAG Summary
                        pmt = build_transcript_prompt(chunks, rels, sub_t3)
                        res, _ = generate_with_fallback(pmt, st.session_state.text_models)
                        st.session_state.tr_res = res
                    st.success("분석 완료!")

        with c_out:
            st.caption("분석 결과")
            if "tr_res" in st.session_state and st.session_state.tr_res:
                with st.container(border=True):
                    st.markdown("##### 📝 족보 기반 요약 노트")
                    st.info(st.session_state.tr_res)
                
                if st.session_state.transcribed_text:
                    with st.expander("🗣️ 변환된 전체 스크립트 보기"):
                        st.text(st.session_state.transcribed_text)
            else:
                st.markdown("""<div style="height: 300px; background: #f9f9f9; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: #aaa;">왼쪽에서 녹음 또는 텍스트를 입력하세요.</div>""", unsafe_allow_html=True)
