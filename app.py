# app.py (Strict Matching + View All Toggle + Clean UI)
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
    
    /* 10. Exam Card Style */
    .exam-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
        position: relative;
    }
    .exam-meta {
        font-size: 0.85rem;
        color: #666;
        margin-bottom: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #f0f0f0;
        padding-bottom: 8px;
    }
    .exam-score-badge {
        background-color: #fff3e0;
        color: #f57c00;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .exam-question {
        font-size: 1.05rem;
        font-weight: 500;
        color: #333;
        line-height: 1.6;
    }

    /* 11. Twin Problem Card Style */
    .twin-card {
        background-color: #f5faff;
        border: 1px solid #bbdefb;
        border-radius: 16px;
        padding: 24px;
        margin-top: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 8px rgba(33,150,243,0.08);
    }
    .twin-badge {
        background-color: #2196f3;
        color: white;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
        margin-bottom: 10px;
        display: inline-block;
    }

    /* 12. Answer/Explanation Box */
    .explanation-box {
        background-color: #f1f8e9;
        border-left: 5px solid #7cb342;
        padding: 15px 20px;
        border-radius: 4px;
        margin-top: 15px;
        margin-bottom: 25px;
    }
    .exp-title {
        font-weight: 800;
        color: #558b2f;
        margin-bottom: 5px;
        font-size: 0.9rem;
    }
    .exp-text {
        font-size: 0.95rem;
        color: #33691e;
        line-height: 1.5;
    }

    /* 13. Sidebar Items */
    .sidebar-subject {
        padding: 10px 15px;
        background-color: white;
        border-radius: 10px;
        margin-bottom: 8px;
        font-weight: 600;
        color: #333;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid #f0f0f0;
        display: flex;
        align-items: center;
        gap: 8px;
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
    # Interactive Parsing & Twin Gen
    "parsed_items": {}, "twin_items": {},
    # Hot Page Navigation
    "hot_pages": [], "hot_pages_analyzed": False, "analyzing_progress": 0,
    # Tab 3 Results
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
    if len(text) < 50: 
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
                    
    return None, "Embedding failed"

def filter_db_by_subject(subject: str, db: list[dict]):
    if not db: return []
    if subject in ["전체", "ALL", ""]: return db
    return [x for x in db if x.get("subject") == subject]

def find_relevant_jokbo(query_text: str, db: list[dict], top_k: int = 5, threshold: float = 0.65):
    if not db: return []
    query_emb, _ = get_embedding_robust(query_text)
    if not query_emb: return []
    
    valid_items = [item for item in db if item.get("embedding")]
    if not valid_items: return []
    db_embs = [item["embedding"] for item in valid_items]
    
    if len(db_embs) == 0: return []
    
    sims = cosine_similarity([query_emb], db_embs)[0]
    top_idxs = np.argsort(sims)[::-1][:top_k]
    
    # [STRICT FILTERING] Threshold set to 0.65 (Adjustable)
    # Only return items that are actually relevant
    results = [{"score": float(sims[i]), "content": valid_items[i]} for i in top_idxs if sims[i] > threshold]
    return results

def generate_with_fallback(prompt: str, model_names: list[str]):
    ensure_configured()
    target_model = st.session_state.best_text_model or "gemini-1.5-flash"
    candidates = [target_model]
    if model_names: candidates.extend(model_names)
    candidates = list(dict.fromkeys(candidates))
    config = genai.GenerationConfig(temperature=0.3)
    
    for name in candidates:
        try:
            model = genai.GenerativeModel(name, generation_config=config)
            res = model.generate_content(prompt)
            if res.text: return res.text, name
        except Exception: 
            continue
    raise Exception("AI 응답 실패")

def transcribe_audio_gemini(audio_bytes, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([
            "Please transcribe the following audio file into text accurately.",
            {"mime_type": "audio/wav", "data": audio_bytes}
        ])
        return response.text
    except Exception: return None

def transcribe_image_to_text(image, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(["Extract text.", image])
        return response.text
    except Exception: return None

# ==========================================
# 4. LLM Logic (Parser & Generator)
# ==========================================

def split_jokbo_text(text):
    if not text: return []
    pattern = r'(?:\n|^)\s*(?=\d+[\.\)])'
    parts = re.split(pattern, text)
    questions = [p.strip() for p in parts if p.strip()]
    return questions

def parse_raw_jokbo_llm(raw_text):
    prompt = f"""
    Analyze this exam question text. Structure it into JSON.
    [Text] {raw_text}
    [Format] JSON with keys: question, choices(list), answer, explanation.
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
    Create a 'Twin Problem' (similar logic, different values/scenario) for:
    Subject: {subject}
    Original: {json.dumps(data, ensure_ascii=False)}
    Output Format (Markdown):
    ### Question
    ...
    ### Answer
    ...
    ### Explanation
    ...
    """
    try:
        res_text, _ = generate_with_fallback(prompt, st.session_state.text_models)
        return res_text
    except Exception as e:
        return f"문제 생성 실패: {str(e)}"

# ==========================================
# 5. Prompts
# ==========================================
def build_chat_prompt(history, context, related, q):
    ctx = "\n".join([f"- {r['content']['text'][:200]}" for r in related[:2]])
    return f"""
    [강의] {context[:800]}
    [족보] {ctx}
    [질문] {q}
    답변해.
    """

def build_transcript_prompt(chunks, related_packs, subject):
    packed = ""
    for idx, (chunk, rel) in enumerate(zip(chunks, related_packs), 1):
        if not rel or rel[0]["score"] < 0.6: continue
        ctx = "\n".join([f"- {r['content']['text'][:200]}" for r in rel[:2]])
        packed += f"\n(구간 {idx})\n[강의] {chunk}\n[족보근거] {ctx}\n"
    return f"강의 전사 내용을 족보 기반으로 요약하세요.\n{packed}"

def chunk_transcript(text):
    return [text[i:i+900] for i in range(0, len(text), 900)]

def get_subject_stats():
    stats = {}
    for item in st.session_state.db:
        subj = item.get("subject", "기타")
        if subj not in stats: stats[subj] = {"count": 0, "last_updated": "방금 전"}
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
# 6. Main App UI
# ==========================================

if not st.session_state.logged_in:
    login()
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 👤 내 프로필")
    with st.container(border=True):
        st.markdown("## 👨‍⚕️ **Student Admin**")
        if st.button("로그아웃", use_container_width=True): logout()

    st.markdown("### 📚 내 학습 과목")
    my_subjects = sorted({x.get("subject", "기타") for x in st.session_state.db})
    if my_subjects:
        for s in my_subjects:
            st.markdown(f"<div class='sidebar-subject'>📘 {s}</div>", unsafe_allow_html=True)
    else: st.caption("등록된 과목 없음")
    st.divider()

    st.markdown("### ⚙️ 설정")
    with st.container(border=True):
        api_key_input = st.text_input("Gemini API Key", type="password", key="api_key_input")
        if api_key_input: st.session_state.api_key = api_key_input.strip()
        if st.button("🔄 연결 테스트", use_container_width=True):
            if st.session_state.api_key:
                t_mods, e_mods = list_available_models(st.session_state.api_key)
                if t_mods:
                    st.session_state.api_key_ok = True
                    st.session_state.text_models = t_mods
                    st.session_state.embedding_models = e_mods
                    st.session_state.best_text_model = get_best_model(t_mods, ["flash", "pro"])
                    st.success("연결 성공!")
                else: st.error("모델 검색 실패")

    st.markdown("### 📊 DB 현황")
    with st.container(border=True):
        st.metric("총 학습 페이지", len(st.session_state.db))
        if st.button("DB 초기화"):
            st.session_state.db = []
            st.rerun()

# --- Main ---
st.title("Med-Study OS")
tab1, tab2, tab3 = st.tabs(["📂 족보 관리", "📖 강의 분석", "🎙️ 강의 녹음"])

# --- TAB 1: 족보 관리 ---
with tab1:
    col_u, col_l = st.columns([1, 2])
    with col_u:
        with st.container(border=True):
            st.markdown("#### ➕ 족보 추가")
            subj = st.text_input("과목명", "해부학")
            files = st.file_uploader("PDF 업로드", accept_multiple_files=True, type="pdf")
            if st.button("학습 시작", type="primary", use_container_width=True):
                if not st.session_state.api_key_ok: st.error("API Key 필요")
                elif files:
                    prog = st.progress(0)
                    for i, f in enumerate(files):
                        doc = fitz.open(stream=f.getvalue(), filetype="pdf")
                        for p_idx, page in enumerate(doc):
                            txt = page.get_text().strip()
                            if len(txt) < 50:
                                try:
                                    pix = page.get_pixmap()
                                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                                    ocr = transcribe_image_to_text(img, st.session_state.api_key)
                                    if ocr: txt = ocr
                                except: pass
                            emb, _ = get_embedding_robust(txt)
                            if emb:
                                st.session_state.db.append({
                                    "page": p_idx+1, "text": txt, "source": f.name, "embedding": emb, "subject": subj
                                })
                        prog.progress((i+1)/len(files))
                    st.success("학습 완료!")
                    time.sleep(1)
                    st.rerun()
    with col_l:
        st.markdown("#### 📚 학습된 데이터")
        stats = get_subject_stats()
        cols = st.columns(3)
        for i, (s, d) in enumerate(stats.items()):
            with cols[i%3]:
                st.metric(label=s, value=f"{d['count']} pages")

# --- TAB 2: 강의 분석 (Clean & Strict) ---
with tab2:
    if st.session_state.t2_selected_subject is None:
        st.markdown("#### 📖 학습할 과목 선택")
        subjects = sorted(list({x["subject"] for x in st.session_state.db}))
        if not subjects: st.info("족보 데이터가 없습니다.")
        cols = st.columns(3)
        for i, s in enumerate(subjects):
            with cols[i%3]:
                if st.button(f"📘 {s}", use_container_width=True):
                    st.session_state.t2_selected_subject = s
                    st.rerun()
    else:
        # Header
        c_bk, c_hd = st.columns([1, 5])
        if c_bk.button("← 뒤로"): 
            st.session_state.t2_selected_subject = None
            st.rerun()
        c_hd.markdown(f"#### 📖 {st.session_state.t2_selected_subject} - 자동 분석 모드")

        # PDF Upload
        with st.expander("📂 강의 PDF 파일 열기", expanded=not st.session_state.lecture_doc):
            l_file = st.file_uploader("강의록 선택", type="pdf", key="l_pdf")
            if l_file and l_file.name != st.session_state.lecture_filename:
                # Reset State for new file
                st.session_state.lecture_doc = fitz.open(stream=l_file.getvalue(), filetype="pdf")
                st.session_state.lecture_filename = l_file.name
                st.session_state.current_page = 0
                st.session_state.last_page_sig = None
                st.session_state.parsed_items = {}
                st.session_state.twin_items = {}
                st.session_state.hot_pages = []
                st.session_state.hot_pages_analyzed = False
                st.rerun()

        if st.session_state.lecture_doc:
            doc = st.session_state.lecture_doc
            target_subj = st.session_state.t2_selected_subject

            # --- Hot Page Analysis (Background) ---
            if not st.session_state.hot_pages_analyzed:
                with st.spinner("🚀 파일 로드 완료. 적중 페이지 분석 중..."):
                    sub_db = filter_db_by_subject(target_subj, st.session_state.db)
                    if sub_db:
                        valid_items = [x for x in sub_db if x.get("embedding")]
                        db_embs = [x["embedding"] for x in valid_items]
                        results = []
                        for p_idx in range(len(doc)):
                            try:
                                page = doc.load_page(p_idx)
                                txt = page.get_text().strip()
                                if len(txt) > 30:
                                    emb, _ = get_embedding_robust(txt)
                                    if emb:
                                        sims = cosine_similarity([emb], db_embs)[0]
                                        max_score = max(sims)
                                        if max_score >= 0.75: # High threshold
                                            results.append({"page": p_idx, "score": max_score})
                            except: pass
                        st.session_state.hot_pages = sorted(results, key=lambda x: x["score"], reverse=True)[:20]
                    st.session_state.hot_pages_analyzed = True
                st.rerun()

            # --- Hot Page Navigation (Hidden by default to be cleaner) ---
            with st.expander(f"🔥 적중 페이지 탐색기 ({len(st.session_state.hot_pages)} pages found)", expanded=False):
                if st.session_state.hot_pages:
                    cols = st.columns(10)
                    for i, item in enumerate(st.session_state.hot_pages):
                        if i < 20: 
                            with cols[i%10]:
                                if st.button(f"P.{item['page']+1}", key=f"nav_{item['page']}", help=f"적중률 {item['score']:.0%}"):
                                    st.session_state.current_page = item['page']
                                    st.rerun()
                else:
                    st.caption("적중률 높은 페이지가 없습니다.")

            st.divider()

            # --- Main Viewer ---
            col_view, col_ai = st.columns([1, 1])
            
            # LEFT: Viewer
            with col_view:
                c1, c2, c3 = st.columns([1, 2, 1])
                if c1.button("◀ Prev"):
                    if st.session_state.current_page > 0:
                        st.session_state.current_page -= 1
                        st.rerun()
                c2.markdown(f"<div style='text-align:center;'>Page {st.session_state.current_page+1}</div>", unsafe_allow_html=True)
                if c3.button("Next ▶"):
                    if st.session_state.current_page < len(doc)-1:
                        st.session_state.current_page += 1
                        st.rerun()
                
                page = doc.load_page(st.session_state.current_page)
                pix = page.get_pixmap(dpi=150)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                p_text = page.get_text().strip()
                st.image(img, use_container_width=True)

            # RIGHT: Analysis
            with col_ai:
                ai_tab_match, ai_tab_chat = st.tabs(["📝 족보 매칭", "💬 AI 튜터"])
                
                with ai_tab_match:
                    # 1. Calculate Relevant Items
                    if not p_text:
                        st.info("텍스트가 없는 페이지입니다.")
                    else:
                        psig = hash(p_text)
                        if psig != st.session_state.last_page_sig:
                            st.session_state.last_page_sig = psig
                            sub_db = filter_db_by_subject(target_subj, st.session_state.db)
                            # STRICT THRESHOLD: 0.65
                            st.session_state.last_related = find_relevant_jokbo(p_text, sub_db, threshold=0.65)
                        
                        rel = st.session_state.last_related
                        
                        # 2. Toggle to Show ALL
                        show_all = st.checkbox("📂 전체 족보 보기 (View All Questions)", value=False)
                        
                        if show_all:
                            # Show all items for this subject
                            all_db = filter_db_by_subject(target_subj, st.session_state.db)
                            st.markdown(f"**전체 문항 ({len(all_db)}개)**")
                            for item in all_db:
                                raw_txt = item['text']
                                questions = split_jokbo_text(raw_txt)
                                if not questions: questions = [raw_txt]
                                for q in questions:
                                    st.markdown(f"""
                                    <div style="border:1px solid #ddd; padding:10px; border-radius:8px; margin-bottom:5px; background:white;">
                                        <small>{item['source']} P.{item['page']}</small><br>
                                        {q[:200]}...
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                            # 3. Default View: Only Relevant Items
                            if not rel:
                                st.info("💡 이 페이지와 직접 연관된 족보 문항이 없습니다.")
                                st.caption("(우측 상단 '전체 족보 보기'를 체크하면 모든 문제를 볼 수 있습니다.)")
                            else:
                                st.success(f"🔥 **{len(rel)}개의 관련 족보 문항이 발견되었습니다.**")
                                
                                for i, r in enumerate(rel): 
                                    content = r['content']
                                    score = r['score']
                                    raw_txt = content['text']
                                    
                                    questions = split_jokbo_text(raw_txt)
                                    if not questions: questions = [raw_txt]
                                    
                                    for q_idx, q_txt in enumerate(questions):
                                        item_id = f"{psig}_{i}_{q_idx}"
                                        
                                        # Display Card
                                        st.markdown(f"""
                                        <div class="exam-card">
                                            <div class="exam-meta">
                                                <span><span class="exam-score-badge">유사도 {score:.0%}</span> &nbsp; {content['source']} (P.{content['page']})</span>
                                            </div>
                                            <div class="exam-question">
                                                {q_txt[:500] + ('...' if len(q_txt)>500 else '')}
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)

                                        # Instant Auto-Analysis
                                        if item_id not in st.session_state.parsed_items:
                                            with st.spinner(f"⚡ 문항 분석 중..."):
                                                parsed = parse_raw_jokbo_llm(q_txt)
                                                st.session_state.parsed_items[item_id] = parsed
                                                if parsed["success"]:
                                                    twin = generate_twin_problem_llm(parsed, target_subj)
                                                    st.session_state.twin_items[item_id] = twin
                                                st.rerun()

                                        # Render Results
                                        if item_id in st.session_state.parsed_items:
                                            parsed = st.session_state.parsed_items[item_id]
                                            if parsed["success"]:
                                                d = parsed["data"]
                                                t_ans, t_twin = st.tabs(["💡 정답 및 해설", "🧩 쌍둥이(변형) 문제"])
                                                
                                                with t_ans:
                                                    st.markdown(f"""
                                                    <div class="explanation-box">
                                                        <div class="exp-title">✅ 정답</div>
                                                        <div class="exp-text">{d.get('answer','정보 없음')}</div>
                                                        <br>
                                                        <div class="exp-title">📘 상세 해설</div>
                                                        <div class="exp-text">{d.get('explanation','정보 없음')}</div>
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                                
                                                with t_twin:
                                                    twin_content = st.session_state.twin_items.get(item_id, "생성 실패")
                                                    st.markdown(f"""
                                                    <div class="twin-card">
                                                        <div class="twin-badge">TWIN PROBLEM</div>
                                                        <div class="exam-question">
                                                            {twin_content}
                                                        </div>
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                            else:
                                                st.error("분석 실패")

                # --- Chat Tab ---
                with ai_tab_chat:
                    st.caption("질문하기")
                    for msg in st.session_state.chat_history:
                        with st.chat_message(msg["role"]): st.markdown(msg["content"])
                    if prompt := st.chat_input("질문 입력..."):
                        if st.session_state.api_key_ok:
                            st.session_state.chat_history.append({"role": "user", "content": prompt})
                            with st.chat_message("user"): st.markdown(prompt)
                            with st.chat_message("assistant"):
                                with st.spinner("..."):
                                    p_context = p_text if p_text else "No text"
                                    rel_context = st.session_state.last_related
                                    chat_prmt = build_chat_prompt(st.session_state.chat_history, p_context, rel_context, prompt)
                                    response_text, _ = generate_with_fallback(chat_prmt, st.session_state.text_models)
                                    st.markdown(response_text)
                                    st.session_state.chat_history.append({"role": "assistant", "content": response_text})

# --- TAB 3: 녹음 (Restored) ---
with tab3:
    with st.container(border=True):
        st.markdown("#### 🎙️ 강의 녹음/분석")
        c_in, c_out = st.columns(2)
        with c_in:
            sub_t3 = st.selectbox("과목", ["전체"] + sorted({x.get("subject", "") for x in st.session_state.db}), key="t3_s")
            t3_mode = st.radio("입력 방식", ["🎤 직접 녹음", "📂 파일 업로드"], horizontal=True)
            target_text = ""
            
            if t3_mode == "🎤 직접 녹음":
                audio_value = st.audio_input("녹음 시작")
                if audio_value and st.button("🚀 분석"):
                    if st.session_state.api_key_ok:
                        with st.spinner("변환 중..."):
                            transcript = transcribe_audio_gemini(audio_value.getvalue(), st.session_state.api_key)
                            if transcript:
                                st.session_state.transcribed_text = transcript
                                target_text = transcript
            else:
                f_txt = st.file_uploader("전사 파일", type="txt")
                if f_txt and st.button("분석"):
                    target_text = f_txt.getvalue().decode().strip()
            
            if target_text and st.session_state.api_key_ok:
                with st.spinner("분석 중..."):
                    sdb = filter_db_by_subject(sub_t3, st.session_state.db)
                    chks = chunk_transcript(target_text)[:10]
                    rels = [find_relevant_jokbo(c, sdb, threshold=0.6) for c in chks]
                    pmt = build_transcript_prompt(chks, rels, sub_t3)
                    res, _ = generate_with_fallback(pmt, st.session_state.text_models)
                    st.session_state.tr_res = res
                st.success("완료!")

        with c_out:
            st.caption("결과")
            if st.session_state.tr_res:
                st.info(st.session_state.tr_res)
                if st.session_state.transcribed_text:
                    with st.expander("전체 텍스트"): st.text(st.session_state.transcribed_text)
