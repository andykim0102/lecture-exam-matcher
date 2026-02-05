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
    /* 전체 폰트 및 배경 설정 */
    html, body, [class*="css"]  {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        background-color: #f8f9fc;
    }
    
    /* 메인 컨테이너 패딩 조절 */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
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
        gap: 8px;
        background-color: transparent;
        padding-bottom: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        border-radius: 8px;
        padding: 0 24px;
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        font-weight: 600;
        color: #666;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        margin-right: 5px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4b89dc !important;
        color: #ffffff !important;
        border: none;
    }
    
    /* 과목 카드 스타일 (Tab 1) */
    .subject-card {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #eee;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        transition: transform 0.2s, box-shadow 0.2s;
        text-align: center;
        height: 100%;
    }
    .subject-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.08);
        border-color: #4b89dc;
    }
    .subject-icon { font-size: 2rem; margin-bottom: 10px; }
    .subject-title { font-size: 1.1rem; font-weight: 700; color: #333; margin-bottom: 5px; }
    .subject-count { font-size: 0.9rem; color: #888; background: #f1f3f5; padding: 4px 10px; border-radius: 12px; display: inline-block; }

    /* 콘텐츠 패널 (Tab 2, 3 - 투명도 개선) */
    .content-panel {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        border: 1px solid #f0f0f0;
        margin-bottom: 20px;
    }
    
    /* 버튼 스타일 */
    div.stButton > button {
        border-radius: 8px;
        height: 3rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }
    
    /* 사이드바 프로필 영역 */
    .profile-box {
        padding: 20px;
        background-color: #ffffff;
        border-radius: 12px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #f0f0f0;
    }
    .profile-text h4 { margin: 0; color: #1565c0; font-size: 1rem; font-weight: 700; }
    .profile-text p { margin: 0; color: #5c6bc0; font-size: 0.8rem; }
    
    /* 메인 헤더 */
    .main-header { font-size: 2rem; font-weight: 800; color: #2c3e50; margin-bottom: 5px; letter-spacing: -0.5px; }
    .sub-header { color: #7f8c8d; font-size: 1rem; margin-bottom: 25px; }
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
    col1, col2, col3 = st.columns([1, 1.2, 1])
    
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 30px; background: white; padding: 40px; border-radius: 20px; box-shadow: 0 10px 25px rgba(0,0,0,0.08);">
                <div style="font-size: 4rem; margin-bottom: 10px;">🩺</div>
                <h1 style="color: #2c3e50; font-weight: 800;">Med-Study OS</h1>
                <p style="color: #95a5a6;">의대생을 위한 스마트 학습 어시스턴트</p>
                <div style="margin-top: 30px;"></div>
            """, 
            unsafe_allow_html=True
        )
        
        # 폼은 HTML 블록 밖에서 별도로 렌더링 (Streamlit 제약)
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
        
        st.markdown("</div>", unsafe_allow_html=True)
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

# AI Helpers
@st.cache_data(show_spinner=False)
def list_text_models(api_key: str):
    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        return [m.name for m in models if "generateContent" in getattr(m, "supported_generation_methods", [])]
    except Exception as e:
        return []

def pick_best_text_model(model_names: list[str]):
    if not model_names: return None
    flash = [m for m in model_names if "flash" in m.lower()]
    return flash[0] if flash else model_names[0]

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

    api_key_input = st.text_input("Gemini API Key", type="password", key="api_key_input")
    
    if api_key_input:
        api_key = api_key_input.strip()
        try:
            st.session_state.api_key = api_key
            genai.configure(api_key=api_key)
            
            available_models = list_text_models(api_key)
            if not available_models:
                st.session_state.api_key_ok = False
                st.error("API 연결 실패: 유효하지 않은 키이거나 모델 목록 권한이 없습니다.")
            else:
                st.session_state.api_key_ok = True
                st.session_state.text_models = available_models
                st.session_state.best_text_model = pick_best_text_model(available_models)
                st.success(f"연결됨: {st.session_state.best_text_model}")
        except Exception as e:
            st.session_state.api_key_ok = False
            st.error(f"오류 발생: {str(e)}")
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
st.markdown('<div class="main-header">Med-Study Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">강의 자료와 족보 데이터를 연결하여 학습 효율을 극대화하세요.</div>', unsafe_allow_html=True)


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

def generate_with_fallback(prompt: str, model_names: list[str]):
    ensure_configured()
    candidates = model_names if model_names else ["gemini-1.5-flash", "gemini-pro"]
    last_err = None
    for name in candidates:
        try:
            model = genai.GenerativeModel(name)
            res = model.generate_content(prompt)
            if res.text: return res.text, name
        except Exception as e: 
            last_err = e
            continue
    raise Exception(f"모든 모델 시도 실패: {str(last_err)}")

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
    ["📂 족보 학습 (Jokbo DB)", "📖 강의 분석 (Lecture)", "🎙️ 전사 분석 (Transcript)"]
)

# --- TAB 1: Upload & DB Management ---
with tab1:
    st.markdown("#### 📂 내 학습 데이터베이스")
    
    # DB 통계 및 카드형 UI 표시
    subjects = sorted({x.get("subject", "기타") for x in st.session_state.db})
    
    if not subjects:
        st.info("아직 학습된 족보 데이터가 없습니다. 아래에서 파일을 업로드하여 과목을 추가하세요.")
    else:
        # 과목별 페이지 수 계산
        subj_counts = {}
        for x in st.session_state.db:
            s = x.get("subject", "기타")
            subj_counts[s] = subj_counts.get(s, 0) + 1
            
        # 카드 그리드 렌더링
        cols = st.columns(4)
        for i, subj in enumerate(subjects):
            with cols[i % 4]:
                st.markdown(
                    f"""
                    <div class="subject-card">
                        <div class="subject-icon">📚</div>
                        <div class="subject-title">{subj}</div>
                        <div class="subject-count">{subj_counts[subj]} pages</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
    
    st.markdown("---")
    
    # 업로드 섹션 디자인 개선
    st.markdown("##### ➕ 새로운 족보 추가하기")
    with st.container():
        st.markdown('<div class="content-panel" style="padding: 20px;">', unsafe_allow_html=True)
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
            if st.button("🚀 학습 시작 (Upload)", type="primary"):
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
        st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 2: Viewer ---
with tab2:
    st.markdown("#### 📖 실시간 강의 분석")
    if not st.session_state.db:
        st.warning("⚠️ 먼저 [족보 학습] 탭에서 DB를 구축해주세요.")
    
    subjects = sorted({x.get("subject", "") for x in st.session_state.db})
    subj_opts = ["전체"] + (subjects if subjects else [])
    
    # 상단 컨트롤 바
    with st.container():
        st.markdown('<div class="content-panel" style="padding: 15px; margin-bottom: 10px;">', unsafe_allow_html=True)
        c_sel, c_up = st.columns([1, 2])
        subj_pick = c_sel.selectbox("분석 과목", subj_opts, key="t2_sub")
        lec_file = c_up.file_uploader("강의 PDF 업로드", type="pdf")
        debug_show = st.toggle("매칭 근거 보기 (Debug)", False)
        st.markdown('</div>', unsafe_allow_html=True)

    if lec_file:
        if st.session_state.lecture_filename != lec_file.name:
            st.session_state.lecture_doc = fitz.open(stream=lec_file.getvalue(), filetype="pdf")
            st.session_state.lecture_filename = lec_file.name
            st.session_state.current_page = 0
            st.session_state.last_page_sig = None
        
        doc = st.session_state.lecture_doc
        
        # 메인 뷰어 영역 (흰색 패널로 감싸기)
        st.markdown('<div class="content-panel">', unsafe_allow_html=True)
        col_view, col_right = st.columns([1.2, 1])
        
        with col_view:
            st.markdown("##### 📄 PDF Viewer")
            c_prev, c_page, c_next = st.columns([1, 2, 1])
            if c_prev.button("◀ Prev", key="prev"):
                if st.session_state.current_page > 0: st.session_state.current_page -= 1
            c_page.markdown(f"<center>{st.session_state.current_page+1} / {len(doc)}</center>", unsafe_allow_html=True)
            if c_next.button("Next ▶", key="next"):
                if st.session_state.current_page < len(doc)-1: st.session_state.current_page += 1
            
            page = doc.load_page(st.session_state.current_page)
            pix = page.get_pixmap(dpi=150)
            st.image(Image.frombytes("RGB", [pix.width, pix.height], pix.samples), use_container_width=True)
            page_text = page.get_text() or ""
            
        with col_right:
            st.markdown("##### 🧑‍🏫 AI 조교 브리핑")
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
                    st.info("💡 이 페이지와 관련된 족보 내용이 없습니다.")
                    st.caption("가볍게 읽고 넘어가셔도 좋습니다.")
                else:
                    ai_sig = (p_sig, subj_pick)
                    if ai_sig != st.session_state.last_ai_sig and st.session_state.api_key_ok:
                        with st.spinner("AI가 족보를 분석 중입니다..."):
                            prompt = build_ta_prompt(page_text, rel, subj_pick)
                            res, _ = generate_with_fallback(prompt, st.session_state.text_models)
                            st.session_state.last_ai_text = res
                            st.session_state.last_ai_sig = ai_sig
                    
                    st.markdown(f"""
                        <div style="background:#f8f9fa; padding:15px; border-radius:8px; border-left:4px solid #4b89dc; font-size:0.95rem; line-height:1.6;">
                        {st.session_state.last_ai_text}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if debug_show:
                        st.divider()
                        st.caption("🔍 근거 자료:")
                        for r in rel[:3]: st.text(f"[{r['score']:.2f}] {r['content']['source']}")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("상단에서 강의 PDF 파일을 업로드해주세요.")

# --- TAB 3: Transcript ---
with tab3:
    st.markdown("#### 🎙️ 강의 녹음/전사 분석")
    
    # 흰색 패널로 전체 감싸기
    st.markdown('<div class="content-panel">', unsafe_allow_html=True)
    
    c_sub, c_dummy = st.columns([1, 2])
    subj_pick_t3 = c_sub.selectbox("분석 과목", ["전체"] + sorted({x.get("subject", "") for x in st.session_state.db}), key="t3_sub")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("##### 1. 전사 텍스트 입력")
        txt_file = st.file_uploader("전사 파일(.txt)", type="txt")
        raw_txt = st.text_area("또는 텍스트 직접 입력", height=200, placeholder="강의 내용을 여기에 붙여넣으세요...")
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✨ 족보 매칭 분석 시작", type="primary"):
            target_txt = (txt_file.getvalue().decode() if txt_file else raw_txt).strip()
            if not target_txt:
                st.error("내용을 입력하세요.")
            elif not st.session_state.api_key_ok:
                st.error("API Key 확인 필요")
            else:
                with st.spinner("전사 텍스트를 분석하고 족보와 대조 중..."):
                    db_sub = filter_db_by_subject(subj_pick_t3, st.session_state.db)
                    chunks = chunk_transcript(target_txt)[:10]
                    rels = []
                    for ch in chunks:
                        rels.append(find_relevant_jokbo(ch, db_sub, top_k=3))
                    
                    prompt = build_transcript_prompt(chunks, rels, subj_pick_t3)
                    res, _ = generate_with_fallback(prompt, st.session_state.text_models)
                    st.session_state.tr_res = res
                st.success("분석 완료!")
                
    with col_t2:
        st.markdown("##### 2. 족보 포인트 요약 노트")
        if "tr_res" in st.session_state:
            st.markdown(f"""
            <div style="background:#ffffff; padding:20px; border:1px solid #eee; border-radius:8px; min-height:300px;">
            {st.session_state.tr_res}
            </div>""", unsafe_allow_html=True)
        else:
            st.info("왼쪽에서 텍스트를 입력하고 분석 버튼을 눌러주세요.")
            
    st.markdown('</div>', unsafe_allow_html=True)
