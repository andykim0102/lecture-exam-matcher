# app.py
import time
import re
import streamlit as st
import google.generativeai as genai
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 0. Page config & Custom CSS
# ==========================================
st.set_page_config(page_title="Med-Study OS", layout="wide", page_icon="🩺")

# 실제 앱 느낌을 위한 커스텀 CSS 주입
st.markdown("""
<style>
    /* 전체 폰트 적용 */
    html, body, [class*="css"]  {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* 로그인 박스 스타일 */
    .login-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding-top: 50px;
    }
    
    /* 탭 스타일 개선 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 4px 4px 0 0;
        padding: 0 20px;
        background-color: #f8f9fa;
        border: none;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 2px solid #4b89dc;
        color: #4b89dc;
    }
    
    /* 버튼 스타일 */
    div.stButton > button {
        border-radius: 6px;
        height: 2.8rem;
        font-weight: 600;
        border: 1px solid #e0e0e0;
    }
    
    /* 사이드바 프로필 영역 */
    .profile-box {
        padding: 20px;
        background-color: #e3f2fd;
        border-radius: 10px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 15px;
    }
    .profile-text h4 {
        margin: 0;
        color: #1565c0;
        font-size: 1rem;
    }
    .profile-text p {
        margin: 0;
        color: #5c6bc0;
        font-size: 0.8rem;
    }
    
    /* 메인 헤더 */
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #333;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 1. Session state initialization
# ==========================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "db" not in st.session_state:
    st.session_state.db = []

if "api_key" not in st.session_state:
    st.session_state.api_key = None

if "api_key_ok" not in st.session_state:
    st.session_state.api_key_ok = False

if "text_models" not in st.session_state:
    st.session_state.text_models = []

if "best_text_model" not in st.session_state:
    st.session_state.best_text_model = None

if "lecture_doc" not in st.session_state:
    st.session_state.lecture_doc = None

if "lecture_filename" not in st.session_state:
    st.session_state.lecture_filename = None

if "current_page" not in st.session_state:
    st.session_state.current_page = 0

# caches
if "last_page_sig" not in st.session_state:
    st.session_state.last_page_sig = None

if "last_ai_sig" not in st.session_state:
    st.session_state.last_ai_sig = None

if "last_ai_text" not in st.session_state:
    st.session_state.last_ai_text = ""

if "last_related" not in st.session_state:
    st.session_state.last_related = []


# ==========================================
# 2. Login Logic
# ==========================================
def login():
    col1, col2, col3 = st.columns([1, 1.5, 1])
    
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 30px;">
                <div style="font-size: 4rem; margin-bottom: 10px;">🩺</div>
                <h1 style="color: #2c3e50;">Med-Study OS</h1>
                <p style="color: #7f8c8d;">의대생을 위한 스마트 학습 어시스턴트</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        with st.form("login_form"):
            st.markdown("##### 🔐 로그인")
            username = st.text_input("아이디", placeholder="admin")
            password = st.text_input("비밀번호", type="password", placeholder="1234")
            
            st.markdown("<br>", unsafe_allow_html=True)
            submit = st.form_submit_button("Start Learning", type="primary")
            
            if submit:
                # 데모용 하드코딩
                if password == "1234":
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("비밀번호가 틀렸습니다. (Demo: 1234)")
        
        st.markdown(
            "<div style='text-align:center; margin-top:15px; color:#aaa; font-size:0.85rem;'>Demo Access: admin / 1234</div>", 
            unsafe_allow_html=True
        )

def logout():
    st.session_state.logged_in = False
    st.rerun()


# ==========================================
# 3. Main App Logic
# ==========================================

# 로그인 체크
if not st.session_state.logged_in:
    login()
    st.stop()

# --- 로그인 이후 UI ---

# 사이드바
with st.sidebar:
    st.markdown(
        """
        <div class="profile-box">
            <div style="font-size: 2.2rem;">👨‍⚕️</div>
            <div class="profile-text">
                <h4>Student Admin</h4>
                <p>본과 2학년 · 학습중</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if st.button("로그아웃", type="secondary"):
        logout()

    st.markdown("---")
    st.caption("⚙️ SYSTEM SETTINGS")

    api_key = st.text_input("Gemini API Key", type="password", key="api_key_input")
    if api_key:
        try:
            st.session_state.api_key = api_key
            genai.configure(api_key=api_key)
            available_models = list_text_models(api_key)
            if not available_models:
                st.session_state.api_key_ok = False
                st.error("사용 가능한 모델 없음")
            else:
                st.session_state.api_key_ok = True
                st.session_state.text_models = available_models
                st.session_state.best_text_model = pick_best_text_model(available_models)
                st.success(f"연결됨: {st.session_state.best_text_model}")
        except Exception as e:
            st.session_state.api_key_ok = False
            st.error("API Key 오류")
    else:
        st.info("AI 기능을 위해 키를 입력하세요.")

    st.markdown("---")
    
    # DB 현황
    subjects_in_db = sorted({x.get("subject", "") for x in st.session_state.db if x.get("subject")})
    st.caption("📚 DATABASE STATUS")
    col_db1, col_db2 = st.columns(2)
    col_db1.metric("총 페이지", len(st.session_state.db))
    col_db2.metric("과목 수", len(subjects_in_db))
    
    if st.button("DB 초기화 (Reset)", key="reset_db_btn"):
        st.session_state.db = []
        st.rerun()


# 메인 헤더
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<div class="main-header">Med-Study Dashboard</div>', unsafe_allow_html=True)
    st.caption("강의 자료와 족보 데이터를 연결하여 학습 효율을 극대화하세요.")


# Settings & Helpers
JOKBO_THRESHOLD = 0.72

def has_jokbo_evidence(related: list[dict]) -> bool:
    return bool(related) and related[0]["score"] >= JOKBO_THRESHOLD

def ensure_configured():
    if st.session_state.get("api_key"):
        genai.configure(api_key=st.session_state["api_key"])

def extract_text_from_pdf(uploaded_file):
    data = uploaded_file.getvalue()
    doc = fitz.open(stream=data, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text() or ""
        if text.strip():
            pages.append({"page": i + 1, "text": text, "source": uploaded_file.name})
    return pages

def get_embedding(text: str):
    text = (text or "").strip()
    if not text: return []
    text = text[:12000]
    ensure_configured()
    try:
        return genai.embed_content(model="models/text-embedding-004", content=text, task_type="retrieval_document")["embedding"]
    except:
        try:
            return genai.embed_content(model="models/embedding-001", content=text, task_type="retrieval_document")["embedding"]
        except:
            return []

def filter_db_by_subject(subject: str, db: list[dict]):
    if not db: return []
    if subject in ["전체", "ALL", ""]: return db
    return [x for x in db if x.get("subject") == subject]

def find_relevant_jokbo(query_text: str, db: list[dict], top_k: int = 5):
    if not db: return []
    query_emb = get_embedding(query_text)
    if not query_emb: return []
    valid_items = [item for item in db if item.get("embedding")]
    if not valid_items: return []
    db_embs = [item["embedding"] for item in valid_items]
    sims = cosine_similarity([query_emb], db_embs)[0]
    top_idxs = np.argsort(sims)[::-1][:top_k]
    return [{"score": float(sims[i]), "content": valid_items[i]} for i in top_idxs]

@st.cache_data(show_spinner=False)
def list_text_models(api_key: str):
    genai.configure(api_key=api_key)
    models = genai.list_models()
    return [m.name for m in models if "generateContent" in getattr(m, "supported_generation_methods", [])]

def pick_best_text_model(model_names: list[str]):
    if not model_names: return None
    flash = [m for m in model_names if "flash" in m.lower()]
    return flash[0] if flash else model_names[0]

def generate_with_fallback(prompt: str, model_names: list[str]):
    ensure_configured()
    for name in model_names:
        try:
            model = genai.GenerativeModel(name)
            res = model.generate_content(prompt)
            if res.text: return res.text, name
        except: continue
    raise Exception("All models failed")

def build_ta_prompt(lecture_text: str, related: list[dict], subject: str):
    ctx = "\n".join([f"- [{r['content']['source']} p{r['content']['page']}] {r['content']['text'][:400]}" for r in related[:3]])
    return f"""
    당신은 의대 조교입니다. 학생이 공부 중인 강의 내용과 관련된 족보(기출) 내용을 바탕으로 핵심을 짚어주세요.
    과목: {subject}
    
    [관련 족보 내용]
    {ctx}
    
    [현재 강의 내용]
    {lecture_text}
    
    출력 형식:
    1. 💡 한줄 요약: (족보와 연관된 핵심 내용 한 문장)
    2. 🎯 출제 포인트 TOP 3:
       - (포인트 1)
       - (포인트 2)
       - (포인트 3)
    3. 📝 암기 키워드: (콤마로 구분)
    """

def build_transcript_prompt(chunks: list[str], related_packs: list[list[dict]], subject: str):
    packed = ""
    for idx, (chunk, rel) in enumerate(zip(chunks, related_packs), 1):
        if not has_jokbo_evidence(rel): continue
        ctx = "\n".join([f"- {r['content']['text'][:200]}" for r in rel[:2]])
        packed += f"\n(구간 {idx})\n[강의] {chunk}\n[족보근거] {ctx}\n"
    
    if not packed: return "족보와 관련된 내용이 없습니다."
    
    return f"""
    당신은 의대 조교입니다. 다음은 강의 녹취록의 일부입니다. 족보(기출)에 근거하여 중요한 부분만 요약 노트로 만들어주세요.
    과목: {subject}
    
    {packed}
    
    출력 형식:
    [족보 적중 노트]
    1. (주제)
       - 내용 요약
       - 관련 기출 포인트
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


# ==========================================
# Tabs Logic
# ==========================================
st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(
    ["📂 족보 학습 (Upload)", "📖 강의 분석 (Viewer)", "🎙️ 전사 분석 (Transcript)"]
)

# --- TAB 1: Upload ---
with tab1:
    st.markdown("#### 📂 과목별 족보 데이터 구축")
    st.info("💡 족보 파일을 업로드하여 AI에게 학습시킵니다. 과목별로 분리하여 관리할 수 있습니다.")

    c1, c2 = st.columns([1, 2])
    with c1:
        subject_for_upload = st.selectbox("과목 선택", ["해부학", "생리학", "약리학", "기타(직접입력)"], index=1)
    with c2:
        subject_custom = st.text_input("과목명 직접 입력", disabled=(subject_for_upload != "기타(직접입력)"), placeholder="예: 병리학")

    subject_final = subject_custom.strip() if subject_for_upload == "기타(직접입력)" else subject_for_upload
    if not subject_final: subject_final = "기타"

    files = st.file_uploader("족보 PDF 파일 선택 (다중 선택 가능)", type="pdf", accept_multiple_files=True)

    col_a, col_b = st.columns([1, 2])
    with col_a:
        max_pages = st.number_input("파일당 최대 학습 페이지", 1, 500, 60)
    with col_b:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 DB 학습 시작", type="primary"):
            if not st.session_state.api_key_ok:
                st.error("API Key 설정이 필요합니다.")
            elif not files:
                st.warning("파일을 업로드해주세요.")
            else:
                bar = st.progress(0)
                status = st.empty()
                new_db = []
                for i, f in enumerate(files):
                    status.text(f"Processing: {f.name}...")
                    pages = extract_text_from_pdf(f)[:int(max_pages)]
                    for p in pages:
                        emb = get_embedding(p["text"])
                        if emb:
                            p["embedding"] = emb
                            p["subject"] = subject_final
                            new_db.append(p)
                        time.sleep(0.5)
                    bar.progress((i+1)/len(files))
                
                st.session_state.db.extend(new_db)
                status.success("✅ 학습 완료!")
                st.toast(f"{len(new_db)} 페이지 학습 완료", icon="🎉")
                time.sleep(1)
                st.rerun()

# --- TAB 2: Viewer ---
with tab2:
    st.markdown("#### 📖 실시간 강의 분석")
    if not st.session_state.db:
        st.warning("⚠️ 먼저 [족보 학습] 탭에서 DB를 구축해주세요.")
    
    subjects = sorted({x.get("subject", "") for x in st.session_state.db})
    subj_opts = ["전체"] + (subjects if subjects else [])
    
    c_sel, c_up = st.columns([1, 2])
    subj_pick = c_sel.selectbox("분석 과목", subj_opts, key="t2_sub")
    lec_file = c_up.file_uploader("강의 PDF 업로드", type="pdf")
    
    debug_show = st.toggle("매칭 근거 보기", False)
    st.markdown("---")

    if lec_file:
        if st.session_state.lecture_filename != lec_file.name:
            st.session_state.lecture_doc = fitz.open(stream=lec_file.getvalue(), filetype="pdf")
            st.session_state.lecture_filename = lec_file.name
            st.session_state.current_page = 0
            st.session_state.last_page_sig = None
        
        doc = st.session_state.lecture_doc
        col_view, col_right = st.columns([1.2, 1])
        
        with col_view:
            c_prev, c_page, c_next = st.columns([1, 2, 1])
            if c_prev.button("◀", key="prev"):
                if st.session_state.current_page > 0: st.session_state.current_page -= 1
            c_page.markdown(f"<center>{st.session_state.current_page+1} / {len(doc)}</center>", unsafe_allow_html=True)
            if c_next.button("▶", key="next"):
                if st.session_state.current_page < len(doc)-1: st.session_state.current_page += 1
            
            page = doc.load_page(st.session_state.current_page)
            pix = page.get_pixmap(dpi=150)
            st.image(Image.frombytes("RGB", [pix.width, pix.height], pix.samples), use_container_width=True)
            page_text = page.get_text() or ""
            
        with col_right:
            st.markdown("### 🧑‍🏫 AI 조교")
            if not st.session_state.db:
                st.error("DB 없음")
            elif not page_text.strip():
                st.info("텍스트 없음")
            else:
                p_sig = hash(page_text)
                if p_sig != st.session_state.last_page_sig:
                    st.session_state.last_page_sig = p_sig
                    db_sub = filter_db_by_subject(subj_pick, st.session_state.db)
                    st.session_state.last_related = find_relevant_jokbo(page_text, db_sub)
                    st.session_state.last_ai_sig = None
                
                rel = st.session_state.last_related
                if not has_jokbo_evidence(rel):
                    st.info("관련 족보 내용이 없습니다.")
                else:
                    ai_sig = (p_sig, subj_pick)
                    if ai_sig != st.session_state.last_ai_sig and st.session_state.api_key_ok:
                        with st.spinner("분석 중..."):
                            prompt = build_ta_prompt(page_text, rel, subj_pick)
                            res, _ = generate_with_fallback(prompt, st.session_state.text_models)
                            st.session_state.last_ai_text = res
                            st.session_state.last_ai_sig = ai_sig
                    
                    st.markdown(f"""<div style="background:#f8f9fa;padding:15px;border-radius:8px;border-left:4px solid #4b89dc;">
                    {st.session_state.last_ai_text}</div>""", unsafe_allow_html=True)
                    
                    if debug_show:
                        st.caption("근거:")
                        for r in rel[:3]: st.text(f"[{r['score']:.2f}] {r['content']['source']}")
    else:
        st.info("강의 PDF를 업로드하세요.")

# --- TAB 3: Transcript ---
with tab3:
    st.markdown("#### 🎙️ 강의 녹음/전사 분석")
    
    c_sub, c_dummy = st.columns([1, 2])
    subj_pick_t3 = c_sub.selectbox("분석 과목", ["전체"] + sorted({x.get("subject", "") for x in st.session_state.db}), key="t3_sub")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        txt_file = st.file_uploader("전사 파일(.txt)", type="txt")
        raw_txt = st.text_area("또는 텍스트 입력", height=200)
        if st.button("✨ 분석 시작", type="primary"):
            target_txt = (txt_file.getvalue().decode() if txt_file else raw_txt).strip()
            if not target_txt:
                st.error("내용을 입력하세요.")
            elif not st.session_state.api_key_ok:
                st.error("API Key 확인 필요")
            else:
                db_sub = filter_db_by_subject(subj_pick_t3, st.session_state.db)
                chunks = chunk_transcript(target_txt)[:10]
                rels = []
                for ch in chunks:
                    rels.append(find_relevant_jokbo(ch, db_sub, top_k=3))
                
                prompt = build_transcript_prompt(chunks, rels, subj_pick_t3)
                res, _ = generate_with_fallback(prompt, st.session_state.text_models)
                st.session_state.tr_res = res
                
    with col_t2:
        if "tr_res" in st.session_state:
            st.markdown(f"""<div style="background:#fff;padding:20px;border:1px solid #ddd;border-radius:8px;">
            {st.session_state.tr_res}</div>""", unsafe_allow_html=True)
        else:
            st.info("왼쪽에서 내용을 입력하고 분석을 시작하세요.")
