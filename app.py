# app.py (UI: Clean Badge Style / Logic: Auto-Generation + Strict Filtering)
import time
import re
import random
import json
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
    /* 1. Force Light Mode & Colors */
    .stApp { background-color: #f8f9fa; } 
    h1, h2, h3, h4, h5, h6, p, span, div, label, .stMarkdown { color: #1c1c1e !important; }
    .gray-text, .text-sm, .login-desc, small { color: #8e8e93 !important; }
    
    /* Button Text Colors */
    div.stButton > button p { color: #007aff !important; }
    div.stButton > button[kind="primary"] p { color: #ffffff !important; }

    /* 2. Input Styles */
    div[data-baseweb="input"] { background-color: #ffffff !important; border: 1px solid #d1d1d6 !important; color: #1c1c1e !important; }
    div[data-baseweb="input"] input { color: #1c1c1e !important; }
    
    /* 3. Layout Adjustments */
    .block-container { 
        padding-top: 1rem !important; 
        padding-bottom: 2rem !important; 
        padding-left: 1rem !important; 
        padding-right: 1rem !important; 
        max-width: 100% !important;
    }
    header[data-testid="stHeader"] { display: none; }

    /* 4. Tab Styles */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: transparent; padding: 4px; border-radius: 10px; margin-bottom: 15px; }
    .stTabs [data-baseweb="tab"] { height: 40px; border-radius: 20px; padding: 0 20px; background-color: #ffffff; border: 1px solid #e0e0e0; font-weight: 600; color: #8e8e93 !important; flex-grow: 0; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
    .stTabs [aria-selected="true"] { background-color: #007aff !important; color: #ffffff !important; box-shadow: 0 4px 8px rgba(0,122,255,0.2); border: none; }

    /* 5. Card Containers */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 20px; 
        border: 1px solid #edf2f7; 
        box-shadow: 0 4px 20px rgba(0,0,0,0.03); 
        background-color: white;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        padding: 20px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.06);
        border-color: #007aff;
    }

    /* 6. Buttons */
    div.stButton > button { border-radius: 12px; font-weight: 600; border: none; box-shadow: none; background-color: #f2f2f7; transition: all 0.2s; height: 3rem; }
    div.stButton > button:hover { background-color: #e5e5ea; transform: scale(0.98); }
    div.stButton > button[kind="primary"] { background-color: #007aff; box-shadow: 0 4px 10px rgba(0,122,255,0.2); }
    div.stButton > button[kind="primary"]:hover { background-color: #0062cc; box-shadow: 0 6px 14px rgba(0,122,255,0.3); }

    /* 7. Subject Title Button */
    div.stButton > button h2 {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        margin: 0 !important;
        padding: 5px 0 !important;
        color: #1c1c1e !important;
        line-height: 1.2 !important;
    }

    /* 8. Login & Misc */
    .login-logo { font-size: 5rem; margin-bottom: 10px; animation: bounce 2s infinite; }
    @keyframes bounce { 0%, 20%, 50%, 80%, 100% {transform: translateY(0);} 40% {transform: translateY(-20px);} 60% {transform: translateY(-10px);} }
    .text-bold { font-weight: 700; color: #1c1c1e !important; }
    div[data-testid="stFileUploader"] { padding: 20px; border: 2px dashed #d1d1d6; border-radius: 16px; background-color: #fafafa; }
    
    /* 9. Chat Messages */
    .stChatMessage { background-color: #f9f9f9; border-radius: 16px; padding: 15px; margin-bottom: 10px; border: 1px solid #f0f0f0; }
    div[data-testid="stChatMessageContent"] p { font-size: 0.95rem; line-height: 1.5; }
    
    /* 10. Jokbo Items (Updated) */
    .jokbo-item {
        background-color: #fffde7;
        border: 1px solid #fff59d;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02);
    }
    
    /* 11. Answer Box */
    .answer-box {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 15px;
        margin-top: 15px;
        border-radius: 8px;
        font-size: 0.95rem;
    }
    
    /* 12. Twin Problem Box */
    .twin-box {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
        padding: 15px;
        margin-top: 15px;
        border-radius: 8px;
        font-size: 0.95rem;
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
    # Interactive Parsing
    "parsed_items": {}, "twin_items": {},
    # Hot Page Navigation
    "hot_pages": [], "hot_pages_analyzed": False
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
                <div class="login-logo">🩺</div>
                <h1 style="font-weight: 800; margin-bottom: 0; color: #1c1c1e;">Med-Study OS</h1>
                <p class="login-desc" style="color: #8e8e93; margin-bottom: 30px;">당신의 스마트한 의대 학습 파트너</p>
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

def get_embedding_robust(text: str, status_placeholder=None):
    text = (text or "").strip()
    if len(text) < 30: 
        return None, "text_too_short"
        
    text = text[:10000]
    ensure_configured()
    
    if not st.session_state.embedding_models:
        _, embs = list_available_models(st.session_state.api_key)
        st.session_state.embedding_models = embs
    
    candidates = st.session_state.embedding_models
    if not candidates:
        return None, "No embedding models available."
        
    sorted_candidates = sorted(candidates, key=lambda x: 0 if 'text-embedding-004' in x else 1)
    
    max_retries = 3
    
    for model_name in sorted_candidates[:2]:
        for attempt in range(max_retries):
            try:
                time.sleep(0.5) 
                if "004" in model_name:
                    res = genai.embed_content(model=model_name, content=text, task_type="retrieval_document")
                else:
                    res = genai.embed_content(model=model_name, content=text)
                    
                if res and "embedding" in res:
                    return res["embedding"], None
            except Exception:
                time.sleep(1)
                    
    return None, "Embedding Failed"

def filter_db_by_subject(subject: str, db: list[dict]):
    if not db: return []
    if subject in ["전체", "ALL", ""]: return db
    return [x for x in db if x.get("subject") == subject]

def find_relevant_jokbo(query_text: str, db: list[dict], top_k: int = 5):
    if not db: return []
    query_emb, _ = get_embedding_robust(query_text)
    if not query_emb: return []
    
    valid_items = [item for item in db if item.get("embedding")]
    if not valid_items: return []
    db_embs = [item["embedding"] for item in valid_items]
    
    if len(db_embs) == 0: return []
    
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
    return f"AI 응답 실패: {str(last_err)}", "Error"

def transcribe_audio_gemini(audio_bytes, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([
            "Please transcribe the following audio file into text accurately.",
            {"mime_type": "audio/wav", "data": audio_bytes}
        ])
        return response.text
    except Exception as e:
        return None

def transcribe_image_to_text(image, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([
            "Extract all text from this image exactly as is.",
            image
        ])
        return response.text
    except Exception:
        return None

# ==========================================
# 4. New Logic: Relevance Badges & Parsing
# ==========================================

def get_relevance_badge(score):
    if score >= 0.85:
        return "🔥🔥 강력 추천 (적중 예상)", "red"
    elif score >= 0.75:
        return "🔥 추천 (밀접 관련)", "orange"
    else:
        return "✅ 참고 (관련 있음)", "green"

def split_jokbo_text(text):
    if not text: return []
    pattern = r'(?:\n|^)\s*(?=\d+[\.\)])'
    parts = re.split(pattern, text)
    questions = [p.strip() for p in parts if p.strip()]
    return questions

def parse_raw_jokbo_llm(raw_text):
    prompt = f"""
    You are an expert exam data parser.
    Analyze the following raw text which may contain a mix of questions, choices, answers, and explanations.
    Structure it into a clean JSON object.
    
    [Raw Text]
    {raw_text}
    
    [Requirements]
    1. Extract 'question', 'choices', 'answer', 'explanation', 'type'.
    2. Return ONLY the JSON object.
    """
    
    try:
        res_text, _ = generate_with_fallback(prompt, st.session_state.text_models)
        clean_text = re.sub(r"```json|```", "", res_text).strip()
        parsed = json.loads(clean_text)
        return {"success": True, "data": parsed}
    except Exception as e:
        return {"success": False, "error": str(e)}

def generate_twin_problem_llm(parsed_data, subject):
    data = parsed_data["data"]
    prompt = f"""
    Create a 'Twin Problem' (Similar but different question) for medical students.
    Subject: {subject}
    Original Question: {data.get('question')}
    Original Answer: {data.get('answer')}
    
    Output Format:
    **[변형 문제]** Question...
    **[정답 및 해설]** Answer & Logic...
    """
    
    try:
        res_text, _ = generate_with_fallback(prompt, st.session_state.text_models)
        return res_text
    except Exception as e:
        return f"문제 생성 실패: {str(e)}"

# --- Prompt Builders ---
def build_chat_prompt(history: list, context_text: str, related_jokbo: list, question: str):
    jokbo_ctx = "\n".join([f"- {r['content']['text'][:300]}" for r in related_jokbo[:3]])
    return f"""
    당신은 의대 조교입니다. 학생의 질문에 답변해주세요.
    [현재 보고 있는 강의 내용] {context_text[:1000]}
    [관련 족보/기출 내용] {jokbo_ctx}
    [학생 질문] {question}
    답변은 친절하고 명확하게, 족보 내용이 있다면 그것을 근거로 설명해주세요.
    """

def build_transcript_prompt(chunks: list[str], related_packs: list[list[dict]], subject: str):
    packed = ""
    for idx, (chunk, rel) in enumerate(zip(chunks, related_packs), 1):
        if not rel: continue
        ctx = "\n".join([f"- {r['content']['text'][:200]}" for r in rel[:2]])
        packed += f"\n(구간 {idx})\n[강의] {chunk}\n[족보근거] {ctx}\n"
    if not packed: return "족보와 관련된 내용이 없습니다."
    return f"""
    당신은 의대 조교입니다. 강의 전사 내용을 족보 기반으로 요약하세요.
    과목: {subject}
    {packed}
    출력: [족보 적중 노트] 형식으로 요약.
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

def get_subject_stats():
    stats = {}
    for item in st.session_state.db:
        subj = item.get("subject", "기타")
        if subj not in stats:
            stats[subj] = {"count": 0, "last_updated": "방금 전"}
        stats[subj]["count"] += 1
    return stats

def get_subject_files(subject):
    files = {}
    for item in st.session_state.db:
        if item.get("subject") == subject:
            src = item.get("source", "Unknown")
            files[src] = files.get(src, 0) + 1
    return files


# ==========================================
# 5. Main App UI
# ==========================================

if not st.session_state.logged_in:
    login()
    st.stop()

# --- 사이드바 ---
with st.sidebar:
    st.markdown("### 👤 내 프로필")
    with st.container(border=True):
        col_p1, col_p2 = st.columns([1, 3])
        with col_p1: st.markdown("## 👨‍⚕️")
        with col_p2:
            st.markdown("**Student Admin**")
            st.caption("본과 2학년")
        if st.button("로그아웃", use_container_width=True): logout()

    st.markdown("### ⚙️ 설정")
    with st.container(border=True):
        api_key_input = st.text_input("Gemini API Key", type="password", key="api_key_input")
        if api_key_input:
            st.session_state.api_key = api_key_input.strip()
            
        if st.button("🔄 모델 목록 불러오기", use_container_width=True):
            if not st.session_state.api_key:
                st.error("API Key를 입력하세요.")
            else:
                with st.spinner("연결 테스트 중..."):
                    t_mods, e_mods = list_available_models(st.session_state.api_key)
                    if t_mods and e_mods:
                        st.session_state.api_key_ok = True
                        st.session_state.text_models = t_mods
                        st.session_state.embedding_models = e_mods
                        st.session_state.best_text_model = get_best_model(t_mods, ["flash", "pro"])
                        st.success(f"✅ 연결 성공!")
                    else:
                        st.error("🚫 연결 실패")
            
    st.markdown("### 📊 DB 현황")
    with st.container(border=True):
        st.metric("총 학습 페이지", len(st.session_state.db))
        if st.button("DB 초기화", use_container_width=True):
            st.session_state.db = []
            st.rerun()

# --- 메인 콘텐츠 ---
st.title("Med-Study OS")

tab1, tab2, tab3 = st.tabs(["📂 족보 관리", "📖 강의 분석", "🎙️ 강의 녹음/분석"])

# --- TAB 1: 족보 관리 ---
with tab1:
    if st.session_state.subject_detail_view:
        target_subj = st.session_state.subject_detail_view
        c_back, c_title = st.columns([1, 5])
        with c_back:
            if st.button("← 목록", use_container_width=True):
                st.session_state.subject_detail_view = None
                st.rerun()
        with c_title: st.markdown(f"### 📂 {target_subj} - 파일 목록")
        st.divider()
        file_map = get_subject_files(target_subj)
        if not file_map: st.info("이 과목에 등록된 파일이 없습니다.")
        else:
            for fname, count in file_map.items():
                with st.container(border=True):
                    c_f1, c_f2 = st.columns([4, 1])
                    with c_f1: st.markdown(f"**📄 {fname}**")
                    with c_f2: st.caption(f"{count} pages")
    else:
        col_upload, col_list = st.columns([1, 2])
        with col_upload:
            with st.container(border=True):
                st.markdown("#### ➕ 족보 추가")
                st.caption("PDF 파일을 업로드하여 AI 학습")
                up_subj = st.selectbox("과목", ["해부학", "생리학", "약리학", "직접입력"], key="up_subj")
                if up_subj == "직접입력":
                    up_subj_custom = st.text_input("과목명 입력", placeholder="예: 병리학")
                    final_subj = up_subj_custom if up_subj_custom else "기타"
                else: final_subj = up_subj
                
                files = st.file_uploader("PDF 선택", accept_multiple_files=True, type="pdf", label_visibility="collapsed")
                
                if st.button("학습 시작", type="primary", use_container_width=True):
                    if not st.session_state.api_key_ok: st.error("왼쪽 설정에서 '모델 목록 불러오기'를 먼저 해주세요!")
                    elif not files: st.warning("파일을 선택해주세요.")
                    else:
                        prog_bar = st.progress(0)
                        
                        with st.expander("📝 처리 로그 보기", expanded=True):
                            log_container = st.empty()
                            logs = []
                            def log(msg):
                                logs.append(msg)
                                log_container.markdown("\n".join([f"- {l}" for l in logs[-5:]]))

                            new_db = []
                            total_files = len(files)
                            
                            for i, f in enumerate(files):
                                try:
                                    log(f"📂 **{f.name}** 분석 시작...")
                                    doc = fitz.open(stream=f.getvalue(), filetype="pdf")
                                    total_pages = len(doc)
                                    
                                    for p_idx, page in enumerate(doc):
                                        text = page.get_text().strip()
                                        
                                        if len(text) < 30:
                                            try:
                                                pix = page.get_pixmap()
                                                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                                                ocr_text = transcribe_image_to_text(img, st.session_state.api_key)
                                                if ocr_text: text = ocr_text
                                            except Exception: pass

                                        emb, err_msg = get_embedding_robust(text, status_placeholder=st.empty())
                                        
                                        if emb:
                                            new_db.append({
                                                "page": p_idx + 1, "text": text, "source": f.name,
                                                "embedding": emb, "subject": final_subj
                                            })
                                        
                                    log(f"✅ **{f.name}** 완료")
                                except Exception as e:
                                    log(f"❌ 오류: {str(e)}")
                                prog_bar.progress((i + 1) / total_files)
                            
                            if new_db:
                                st.session_state.db.extend(new_db)
                                st.success(f"🎉 학습 완료! (총 {len(new_db)} 페이지)")
                                time.sleep(1.5)
                                st.rerun()
                        
        with col_list:
            st.markdown("#### 📚 내 학습 데이터")
            stats = get_subject_stats()
            if not stats: st.info("등록된 족보가 없습니다.")
            subjects = sorted(stats.keys())
            
            for i in range(0, len(subjects), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(subjects):
                        subj_name = subjects[i+j]
                        subj_data = stats[subj_name]
                        with cols[j]:
                            with st.container(border=True):
                                if st.button(f"## {subj_name}", key=f"btn_view_{subj_name}", use_container_width=True):
                                    st.session_state.subject_detail_view = subj_name
                                    st.rerun()
                                st.caption(f"{subj_data['count']} pages analyzed")

# --- TAB 2: 강의 분석 (Clean UI + Auto Gen) ---
with tab2:
    if st.session_state.t2_selected_subject is None:
        st.markdown("#### 📖 학습할 과목을 선택하세요")
        stats = get_subject_stats()
        subjects = sorted(stats.keys())
        if not subjects: st.info("데이터가 없습니다. 족보 관리 탭에서 추가하세요.")
        else:
             cols = st.columns(3)
             for i, subj in enumerate(subjects):
                 with cols[i % 3]:
                     if st.button(f"## {subj}\n\n📄 {stats[subj]['count']} pages", key=f"t2_sel_{subj}", use_container_width=True):
                         st.session_state.t2_selected_subject = subj
                         st.rerun()
    else:
        target_subj = st.session_state.t2_selected_subject
        c_back, c_header = st.columns([1, 5])
        with c_back:
            if st.button("← 목록", key="t2_back_btn"):
                st.session_state.t2_selected_subject = None
                st.rerun()
        with c_header: st.markdown(f"#### 📖 {target_subj} - 실시간 강의 분석")
        
        with st.expander("📂 강의 PDF 파일 업로드 / 변경", expanded=(st.session_state.lecture_doc is None)):
            l_file = st.file_uploader("PDF 파일 선택", type="pdf", key="t2_f", label_visibility="collapsed")
            if l_file:
                if st.session_state.lecture_filename != l_file.name:
                    st.session_state.lecture_doc = fitz.open(stream=l_file.getvalue(), filetype="pdf")
                    st.session_state.lecture_filename = l_file.name
                    st.session_state.current_page = 0
                    st.session_state.last_page_sig = None
                    st.session_state.chat_history = [] 
                    st.session_state.parsed_items = {}
                    st.session_state.twin_items = {}
                    st.session_state.hot_pages = [] 
                    st.rerun()

        if st.session_state.lecture_doc:
            doc = st.session_state.lecture_doc
            
            # --- Hot Page Logic (Simplified) ---
            with st.expander("🔥 족보 적중 페이지 탐색기", expanded=not st.session_state.hot_pages_analyzed):
                if not st.session_state.hot_pages_analyzed:
                    if st.button("🚀 전체 페이지 분석 시작", type="primary"):
                        if not st.session_state.api_key_ok: st.error("API Key 필요")
                        else:
                            sub_db = filter_db_by_subject(target_subj, st.session_state.db)
                            if sub_db:
                                db_embs = [x["embedding"] for x in sub_db if x.get("embedding")]
                                if db_embs:
                                    results = []
                                    prog_bar = st.progress(0)
                                    for p_idx in range(len(doc)):
                                        try:
                                            txt = doc.load_page(p_idx).get_text().strip()
                                            if len(txt) > 30:
                                                emb, _ = get_embedding_robust(txt)
                                                if emb:
                                                    score = max(cosine_similarity([emb], db_embs)[0])
                                                    if score >= 0.75: results.append({"page": p_idx, "score": score})
                                        except: pass
                                        prog_bar.progress((p_idx+1)/len(doc))
                                    st.session_state.hot_pages = sorted(results, key=lambda x: x["score"], reverse=True)[:20]
                                    st.session_state.hot_pages_analyzed = True
                                    st.rerun()
                            else: st.warning("데이터 부족")
                else:
                    cols = st.columns(6)
                    for i, item in enumerate(st.session_state.hot_pages):
                        with cols[i % 6]:
                            if st.button(f"P.{item['page']+1}", key=f"nav_{item['page']}"):
                                st.session_state.current_page = item['page']
                                st.rerun()
                            st.caption(f"{item['score']:.0%}")
            
            st.divider()

            col_view, col_ai = st.columns([1.5, 1.5])
            
            # --- Left: Viewer ---
            with col_view:
                with st.container(border=True):
                    c1, c2, c3 = st.columns([1, 2, 1])
                    with c1:
                        if st.button("◀", use_container_width=True):
                            if st.session_state.current_page > 0: 
                                st.session_state.current_page -= 1
                                st.rerun()
                    with c2:
                        st.markdown(f"<div style='text-align:center; font-weight:bold; padding-top:8px;'>Page {st.session_state.current_page+1} / {len(doc)}</div>", unsafe_allow_html=True)
                    with c3:
                        if st.button("▶", use_container_width=True):
                            if st.session_state.current_page < len(doc)-1: 
                                st.session_state.current_page += 1
                                st.rerun()
                    
                    page = doc.load_page(st.session_state.current_page)
                    pix = page.get_pixmap(dpi=150)
                    pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    p_text = page.get_text() or ""
                    st.image(pil_image, use_container_width=True)

            # --- Right: AI Assistant (Badge + Auto Gen) ---
            with col_ai:
                st.subheader("💡 족보 AI 분석")
                
                if not p_text.strip():
                    st.info("이 페이지에는 분석할 텍스트가 없습니다.")
                else:
                    # 1. Similarity Search (Cache)
                    psig = hash(p_text)
                    if psig != st.session_state.last_page_sig:
                        st.session_state.last_page_sig = psig
                        sub_db = filter_db_by_subject(target_subj, st.session_state.db)
                        # Fetch top 5, but filter strictly
                        raw = find_relevant_jokbo(p_text, sub_db, top_k=5)
                        # STRICT FILTER: Score >= 0.6
                        st.session_state.last_related = [r for r in raw if r['score'] >= 0.60]
                    
                    related = st.session_state.last_related
                    
                    if not related:
                        st.success("✨ 편안하게 공부하세요! 이 페이지와 관련된 기출문제가 없습니다.")
                    else:
                        for i, r in enumerate(related[:3]):
                            badge_text, badge_color = get_relevance_badge(r['score'])
                            content = r['content']
                            raw_txt = content['text']
                            
                            with st.container(border=True):
                                st.markdown(f":{badge_color}[{badge_text}]")
                                
                                split_q = split_jokbo_text(raw_txt)
                                if not split_q: split_q = [raw_txt]
                                
                                for seq_idx, q_txt in enumerate(split_q):
                                    item_id = f"{psig}_{i}_{seq_idx}"
                                    st.markdown(f"**Q.** {q_txt}")
                                    
                                    # [AUTO GENERATION]
                                    if item_id not in st.session_state.parsed_items:
                                        if st.session_state.api_key_ok:
                                            with st.spinner("AI가 정답과 해설을 분석 중..."):
                                                parsed = parse_raw_jokbo_llm(q_txt)
                                                st.session_state.parsed_items[item_id] = parsed
                                                if parsed["success"]:
                                                    twin = generate_twin_problem_llm(parsed, target_subj)
                                                    st.session_state.twin_items[item_id] = twin
                                                else:
                                                    st.session_state.twin_items[item_id] = "분석 실패"
                                    
                                    # Display Results
                                    if item_id in st.session_state.parsed_items:
                                        res = st.session_state.parsed_items[item_id]
                                        if res["success"]:
                                            d = res["data"]
                                            st.markdown(f"""
                                            <div class="answer-box">
                                                <strong>✅ 정답:</strong> {d.get('answer', '정보 없음')}<br>
                                                <strong>💡 해설:</strong> {d.get('explanation', '해설 없음')}
                                            </div>
                                            """, unsafe_allow_html=True)
                                            
                                            with st.expander("🔄 AI 변형 문제 (시험 대비)"):
                                                st.markdown(f"""
                                                <div class="twin-box">
                                                    {st.session_state.twin_items.get(item_id, "")}
                                                </div>
                                                """, unsafe_allow_html=True)

                st.divider()
                if prompt := st.chat_input("질문하기"):
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    chat_prmt = build_chat_prompt(st.session_state.chat_history, p_text, related, prompt)
                    res, _ = generate_with_fallback(chat_prmt, st.session_state.text_models)
                    st.session_state.chat_history.append({"role": "assistant", "content": res})
                    st.rerun()

# --- TAB 3: 강의 녹음 ---
with tab3:
    with st.container(border=True):
        st.markdown("#### 🎙️ 강의 녹음/분석")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("녹음 기능은 데모 버전에서 텍스트 입력으로 대체될 수 있습니다.")
            txt_in = st.text_area("강의 내용 입력 (전사 텍스트)", height=150)
            if st.button("분석 실행", type="primary"):
                if txt_in and st.session_state.api_key_ok:
                    with st.spinner("분석 중..."):
                        sdb = filter_db_by_subject(st.session_state.get("t3_s", "전체"), st.session_state.db)
                        chks = chunk_transcript(txt_in)[:5]
                        rels = [find_relevant_jokbo(c, sdb) for c in chks]
                        pmt = build_transcript_prompt(chks, rels, "일반")
                        res, _ = generate_with_fallback(pmt, st.session_state.text_models)
                        st.session_state.tr_res = res
        with c2:
            if "tr_res" in st.session_state:
                st.info(st.session_state.tr_res)
