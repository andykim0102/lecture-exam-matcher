# app.py
import time
import re
import random  # For simulating update times
import streamlit as st
import google.generativeai as genai
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_drawable_canvas import st_canvas  # ✏️ 필기 기능을 위한 라이브러리
import base64
from io import BytesIO

# ==========================================
# 🚑 Monkey Patch for streamlit-drawable-canvas
# Streamlit 1.40+ 호환성 문제 해결을 위한 패치
# ==========================================
import streamlit.elements.image as st_image
if not hasattr(st_image, 'image_to_url'):
    def image_to_url(image, width, clamp, channels, output_format, image_id, allow_emoji=False):
        """
        Streamlit 1.40+에서 사라진 image_to_url을 대체하는 더미 함수.
        이미지가 이미 URL(문자열) 형태인 경우 그대로 반환하여 canvas가 처리하게 함.
        """
        if isinstance(image, str):
            return image
        return "" # Fallback
    st_image.image_to_url = image_to_url

# ==========================================
# 0. Page config & Custom CSS
# ==========================================
st.set_page_config(page_title="Med-Study OS", layout="wide", page_icon="🩺")

# 실제 앱 느낌을 위한 커스텀 CSS 주입
st.markdown("""
<style>
    /* 1. 강제 라이트 모드 적용 */
    .stApp { background-color: #f8f9fa; } 
    h1, h2, h3, h4, h5, h6, p, span, div, label, .stMarkdown { color: #1c1c1e !important; }
    .gray-text, .text-sm, .login-desc, small { color: #8e8e93 !important; }
    
    /* 버튼 텍스트 색상 복구 */
    div.stButton > button p { color: #007aff !important; }
    div.stButton > button[kind="primary"] p { color: #ffffff !important; }

    /* 2. 입력창 스타일 */
    div[data-baseweb="input"] { background-color: #ffffff !important; border: 1px solid #d1d1d6 !important; color: #1c1c1e !important; }
    div[data-baseweb="input"] input { color: #1c1c1e !important; }
    
    /* 3. 레이아웃 조정 (Full Width) */
    .block-container { 
        padding-top: 1.5rem; 
        padding-bottom: 2rem; 
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important; /* 꽉 차게 설정 */
    }

    /* 4. 탭 스타일링 */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: transparent; padding: 4px; border-radius: 10px; margin-bottom: 25px; }
    .stTabs [data-baseweb="tab"] { height: 40px; border-radius: 20px; padding: 0 20px; background-color: #ffffff; border: 1px solid #e0e0e0; font-weight: 600; color: #8e8e93 !important; flex-grow: 0; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
    .stTabs [aria-selected="true"] { background-color: #007aff !important; color: #ffffff !important; box-shadow: 0 4px 8px rgba(0,122,255,0.2); border: none; }

    /* 5. 카드 컨테이너 */
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

    /* 6. 버튼 스타일 */
    div.stButton > button { border-radius: 12px; font-weight: 600; border: none; box-shadow: none; background-color: #f2f2f7; transition: all 0.2s; height: 3rem; }
    div.stButton > button:hover { background-color: #e5e5ea; transform: scale(0.98); }
    div.stButton > button[kind="primary"] { background-color: #007aff; box-shadow: 0 4px 10px rgba(0,122,255,0.2); }
    div.stButton > button[kind="primary"]:hover { background-color: #0062cc; box-shadow: 0 6px 14px rgba(0,122,255,0.3); }

    /* 7. 과목 카드 제목 버튼 스타일 */
    div.stButton > button h2 {
        font-size: 2rem !important;
        font-weight: 800 !important;
        margin: 0 !important;
        padding: 10px 0 !important;
        color: #1c1c1e !important;
        line-height: 1.2 !important;
    }

    /* 8. 로그인 & 기타 */
    .login-logo { font-size: 5rem; margin-bottom: 10px; animation: bounce 2s infinite; }
    @keyframes bounce { 0%, 20%, 50%, 80%, 100% {transform: translateY(0);} 40% {transform: translateY(-20px);} 60% {transform: translateY(-10px);} }
    .text-bold { font-weight: 700; color: #1c1c1e !important; }
    div[data-testid="stFileUploader"] { padding: 20px; border: 2px dashed #d1d1d6; border-radius: 16px; background-color: #fafafa; }
    div[data-baseweb="toast"] div { color: #ffffff !important; }
    
    /* 9. 채팅 메시지 스타일 */
    .stChatMessage { background-color: #f9f9f9; border-radius: 16px; padding: 15px; margin-bottom: 10px; border: 1px solid #f0f0f0; }
    div[data-testid="stChatMessageContent"] p { font-size: 0.95rem; line-height: 1.5; }
    
    /* 10. 족보 아이템 스타일 */
    .jokbo-item {
        background-color: #fffde7;
        border: 1px solid #fff59d;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02);
    }
    .jokbo-source {
        font-size: 0.8rem;
        color: #f57f17;
        margin-bottom: 6px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* 11. 사이드바 과목 리스트 스타일 */
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
    .sidebar-icon { font-size: 1.1rem; }
    
    /* 12. 캔버스 툴바 스타일 */
    div[data-testid="stExpander"] { background-color: white; border-radius: 12px; border: 1px solid #eee; }
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

# For Edit Mode in Tab 1
if "edit_target_subject" not in st.session_state:
    st.session_state.edit_target_subject = None

# For Detail View in Tab 1
if "subject_detail_view" not in st.session_state:
    st.session_state.subject_detail_view = None

# For Subject Selection in Tab 2
if "t2_selected_subject" not in st.session_state:
    st.session_state.t2_selected_subject = None

# For Audio Analysis
if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""

# For Chat History in Tab 2
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# caches
if "last_page_sig" not in st.session_state:
    st.session_state.last_page_sig = None

if "last_ai_sig" not in st.session_state:
    st.session_state.last_ai_sig = None

if "last_ai_text" not in st.session_state:
    st.session_state.last_ai_text = ""

if "last_related" not in st.session_state:
    st.session_state.last_related = []

# ✏️ [NEW] 필기 데이터 저장용 (페이지별 저장)
if "drawings" not in st.session_state:
    st.session_state.drawings = {}


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
def rename_subject(old_name, new_name):
    count = 0
    for item in st.session_state.db:
        if item.get("subject") == old_name:
            item["subject"] = new_name
            count += 1
    return count

def get_subject_stats():
    stats = {}
    for item in st.session_state.db:
        subj = item.get("subject", "기타")
        if subj not in stats:
            rand_min = random.randint(1, 59)
            stats[subj] = {"count": 0, "last_updated": f"{rand_min}분 전"}
        stats[subj]["count"] += 1
    return stats

def get_subject_files(subject):
    files = {}
    for item in st.session_state.db:
        if item.get("subject") == subject:
            src = item.get("source", "Unknown")
            files[src] = files.get(src, 0) + 1
    return files

# AI & PDF Helpers
def has_jokbo_evidence(related: list[dict]) -> bool:
    return bool(related) and related[0]["score"] >= 0.72

def ensure_configured():
    if st.session_state.get("api_key"):
        genai.configure(api_key=st.session_state["api_key"])

def list_text_models(api_key: str):
    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        return [m.name for m in models if "generateContent" in getattr(m, "supported_generation_methods", [])]
    except Exception:
        return []

def pick_best_text_model(model_names: list[str]):
    if not model_names: return None
    flash = [m for m in model_names if "flash" in m.lower()]
    return flash[0] if flash else model_names[0]

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
    raise Exception(f"AI 응답 실패: {str(last_err)}")

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
        st.error(f"음성 인식 실패: {e}")
        return None

def pil_to_base64(image):
    """PIL 이미지를 Base64 URL로 변환 (canvas background용)"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# --- New Prompts for Specialized Analysis ---

def build_overview_prompt(first_page_text, subject):
    """강의록 첫 페이지용: 전체 공부 방향성 제시"""
    return f"""
    너는 의대 수석 조교다. 지금 학생이 '{subject}' 강의록의 첫 페이지(표지/목차)를 보고 있다.
    이 강의록 전체를 공부할 때 어떤 마음가짐과 전략을 가져야 하는지, 족보(기출) 패턴을 고려하여 조언해라.
    
    [강의록 첫 페이지 내용]
    {first_page_text[:1500]}
    
    출력 형식:
    1. 🏁 이 강의의 핵심 목표 (한 줄)
    2. 🚩 족보 기반 공부 전략 (3가지 포인트)
    3. ⚠️ 주의해야 할 점
    """

def build_page_analysis_prompt(lecture_text, related_jokbo, subject):
    """일반 페이지용: 방향성, 쌍둥이문제, 해설 생성"""
    jokbo_ctx = "\n".join([f"- {r['content']['text'][:300]}" for r in related_jokbo[:3]])
    
    return f"""
    너는 의대 조교다. 현재 강의록 페이지와 연관된 족보(기출)를 분석해라.
    과목: {subject}
    
    [관련 족보/기출 내용]
    {jokbo_ctx}
    
    [현재 강의 내용]
    {lecture_text[:1500]}
    
    다음 3가지 섹션으로 나누어 출력하라. 각 섹션 헤더를 정확히 지킬 것.
    
    [SECTION: DIRECTION]
    이 페이지 공부 방향성을 한 문단으로 요약. (어떤 키워드가 족보에 자주 나오는지 등)
    
    [SECTION: TWIN_Q]
    위 족보 문제와 유사한 '쌍둥이 문제(변형 문제)'를 1개 만들어라. (객관식 또는 단답형)
    
    [SECTION: EXPLANATION]
    위 쌍둥이 문제의 정답과, 왜 그것이 답인지에 대한 족보 기반 해설.
    """

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
        if not has_jokbo_evidence(rel): continue
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

# ==========================================
# 4. Main App UI
# ==========================================

# 로그인 체크
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

    # --- NEW: 내 학습 과목 리스트 ---
    st.markdown("### 📚 내 학습 과목")
    my_subjects = sorted({x.get("subject", "기타") for x in st.session_state.db})
    if my_subjects:
        for s in my_subjects:
            st.markdown(
                f"""
                <div class="sidebar-subject">
                    <span class="sidebar-icon">📘</span> {s}
                </div>
                """, 
                unsafe_allow_html=True
            )
    else:
        st.caption("아직 등록된 과목이 없습니다.")
        st.caption("'족보 관리' 탭에서 추가해주세요.")
    st.divider()
    # ------------------------------

    st.markdown("### ⚙️ 설정")
    with st.container(border=True):
        api_key_input = st.text_input("Gemini API Key", type="password", key="api_key_input")
        if api_key_input:
            api_key = api_key_input.strip()
            try:
                st.session_state.api_key = api_key
                genai.configure(api_key=api_key)
                models = list_text_models(api_key)
                if models:
                    st.session_state.api_key_ok = True
                    st.session_state.text_models = models
                    st.session_state.best_text_model = pick_best_text_model(models)
                    st.success(f"연결됨: {st.session_state.best_text_model}")
                else: st.error("모델 권한 없음")
            except Exception as e: st.error(f"키 오류: {e}")
        else: st.warning("API Key 입력 필요")
            
    st.markdown("### 📊 DB 현황")
    with st.container(border=True):
        st.metric("총 학습 페이지", len(st.session_state.db))
        if st.button("DB 초기화", use_container_width=True):
            st.session_state.db = []
            st.rerun()

# --- 메인 콘텐츠 ---
st.title("Med-Study OS")

# 탭 구성
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
                    if not st.session_state.api_key_ok: st.error("API Key 필요")
                    elif not files: st.warning("파일 필요")
                    else:
                        prog = st.progress(0)
                        new_db = []
                        for i, f in enumerate(files):
                            pgs = extract_text_from_pdf(f)
                            for p in pgs:
                                emb = get_embedding(p["text"])
                                if emb:
                                    p["embedding"] = emb
                                    p["subject"] = final_subj
                                    new_db.append(p)
                            prog.progress((i+1)/len(files))
                        st.session_state.db.extend(new_db)
                        st.toast("학습 완료!", icon="🎉")
                        time.sleep(1)
                        st.rerun()
        with col_list:
            st.markdown("#### 📚 내 학습 데이터")
            stats = get_subject_stats()
            if not stats: st.info("등록된 족보가 없습니다. 왼쪽에서 추가해주세요.")
            subjects = sorted(stats.keys())
            for i in range(0, len(subjects), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(subjects):
                        subj_name = subjects[i+j]
                        subj_data = stats[subj_name]
                        with cols[j]:
                            with st.container(border=True):
                                c_head_1, c_head_2 = st.columns([4, 1])
                                is_editing = (st.session_state.edit_target_subject == subj_name)
                                with c_head_1:
                                    if is_editing: new_name_input = st.text_input("새 이름", value=subj_name, key=f"edit_in_{subj_name}", label_visibility="collapsed")
                                    else:
                                        # 과목명 클릭 시 상세 보기로 이동 (버튼 스타일) - h2 태그로 꽉 차게
                                        if st.button(f"## {subj_name}", key=f"btn_view_{subj_name}", help="클릭하여 파일 목록 보기"):
                                            st.session_state.subject_detail_view = subj_name
                                            st.rerun()
                                with c_head_2:
                                    if is_editing:
                                        if st.button("💾", key=f"save_{subj_name}"):
                                            if new_name_input and new_name_input != subj_name:
                                                rename_subject(subj_name, new_name_input)
                                            st.session_state.edit_target_subject = None
                                            st.rerun()
                                    else:
                                        if st.button("✏️", key=f"edit_btn_{subj_name}"):
                                            st.session_state.edit_target_subject = subj_name
                                            st.rerun()
                                if not is_editing:
                                    st.markdown("---")
                                    st.markdown(f"**⚡ 분석된 패턴:** {subj_data['count']}건")
                                    st.markdown(f"<span class='gray-text'>🕒 최근 업데이트: {subj_data['last_updated']}</span>", unsafe_allow_html=True)

# --- TAB 2: 강의 분석 (UI 개선 & Chat & Canvas) ---
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
                     # 과목 선택 카드도 크게
                     btn_label = f"## {subj}\n\n📄 {stats[subj]['count']} pages"
                     if st.button(btn_label, key=f"t2_sel_{subj}", use_container_width=True):
                         st.session_state.t2_selected_subject = subj
                         st.rerun()
    else:
        target_subj = st.session_state.t2_selected_subject
        c_back, c_header = st.columns([1, 5])
        with c_back:
            if st.button("← 과목 변경", key="t2_back_btn"):
                st.session_state.t2_selected_subject = None
                st.rerun()
        with c_header: st.markdown(f"#### 📖 {target_subj} - 실시간 강의 분석")
        
        # 파일 업로드 영역
        with st.expander("📂 강의 PDF 파일 업로드 / 변경", expanded=(st.session_state.lecture_doc is None)):
            l_file = st.file_uploader("PDF 파일 선택", type="pdf", key="t2_f", label_visibility="collapsed")
            if l_file:
                if st.session_state.lecture_filename != l_file.name:
                    st.session_state.lecture_doc = fitz.open(stream=l_file.getvalue(), filetype="pdf")
                    st.session_state.lecture_filename = l_file.name
                    st.session_state.current_page = 0
                    st.session_state.last_page_sig = None
                    st.session_state.chat_history = [] 

        if st.session_state.lecture_doc:
            doc = st.session_state.lecture_doc
            
            # 메인 레이아웃: 왼쪽(뷰어) / 오른쪽(AI)
            col_view, col_ai = st.columns([1.8, 1.2])
            
            # 1. 왼쪽: PDF 뷰어 (with 필기 모드)
            with col_view:
                with st.container(border=True):
                    # 네비게이션 & 필기 모드 토글
                    c1, c2, c3, c4 = st.columns([1, 2, 1, 1.5])
                    with c1:
                        if st.button("◀", use_container_width=True):
                            if st.session_state.current_page > 0: 
                                st.session_state.current_page -= 1
                                st.session_state.chat_history = [] 
                    with c2:
                        st.markdown(f"<div style='text-align:center; font-weight:bold; padding-top:8px;'>Page {st.session_state.current_page+1} / {len(doc)}</div>", unsafe_allow_html=True)
                    with c3:
                        if st.button("▶", use_container_width=True):
                            if st.session_state.current_page < len(doc)-1: 
                                st.session_state.current_page += 1
                                st.session_state.chat_history = [] 
                    with c4:
                        use_annotation = st.toggle("🖊️ 필기 모드", value=False)
                    
                    # 이미지 렌더링
                    page = doc.load_page(st.session_state.current_page)
                    pix = page.get_pixmap(dpi=150) # 화면 표시용 적정 DPI
                    pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    p_text = page.get_text() or ""
                    
                    if use_annotation:
                        # 캔버스 툴바 (색상, 굵기)
                        with st.expander("🎨 펜 설정", expanded=True):
                            col_tool1, col_tool2, col_tool3 = st.columns([1, 1, 1])
                            with col_tool1:
                                drawing_mode = st.selectbox(
                                    "도구", ("freedraw", "line", "rect", "circle", "transform"),
                                    label_visibility="collapsed"
                                )
                            with col_tool2:
                                stroke_width = st.slider("굵기", 1, 25, 3, label_visibility="collapsed")
                            with col_tool3:
                                stroke_color = st.color_picker("색상", "#FF0000", label_visibility="collapsed")
                        
                        # 캔버스 ID 생성 (페이지별 고유)
                        canvas_key = f"canvas_{st.session_state.lecture_filename}_{st.session_state.current_page}"
                        initial_drawing = st.session_state.drawings.get(canvas_key)
                        
                        # 🖼️ 이미지를 Base64로 변환하여 전달 (Patch 적용됨)
                        bg_image_url = pil_to_base64(pil_image)

                        canvas_result = st_canvas(
                            fill_color="rgba(255, 165, 0, 0.3)",
                            stroke_width=stroke_width,
                            stroke_color=stroke_color,
                            background_image=bg_image_url,
                            update_streamlit=True,
                            height=pil_image.height,
                            width=pil_image.width,
                            drawing_mode=drawing_mode,
                            key=canvas_key,
                            initial_drawing=initial_drawing
                        )
                        
                        # 필기 데이터 저장 (페이지 전환 후에도 유지)
                        if canvas_result.json_data is not None:
                            st.session_state.drawings[canvas_key] = canvas_result.json_data
                            
                    else:
                        st.image(pil_image, use_container_width=True)

            # 2. 오른쪽: AI 조교 (분석 & 채팅)
            with col_ai:
                with st.container(border=True):
                    # 탭 분리: 자동 분석 vs 채팅
                    ai_tab1, ai_tab2 = st.tabs(["📝 족보 분석", "💬 질의응답"])
                    
                    # 공통 분석 로직
                    if not p_text.strip():
                        analysis_ready = False
                        with ai_tab1: st.caption("텍스트가 없는 이미지 페이지입니다.")
                    else:
                        analysis_ready = True
                        psig = hash(p_text)
                        
                        # 페이지 변경 시 AI 분석 트리거
                        if psig != st.session_state.last_page_sig:
                            st.session_state.last_page_sig = psig
                            sub_db = filter_db_by_subject(target_subj, st.session_state.db)
                            st.session_state.last_related = find_relevant_jokbo(p_text, sub_db)
                            st.session_state.last_ai_sig = None
                        
                        rel = st.session_state.last_related
                    
                    # --- Tab A: 족보 분석 ---
                    with ai_tab1:
                        if analysis_ready:
                            # 1. 첫 페이지면 전체 방향성 (Overview)
                            if st.session_state.current_page == 0:
                                st.markdown("##### 🏁 전체 강의 학습 전략")
                                aisig = ("overview", target_subj, psig)
                                if aisig != st.session_state.last_ai_sig and st.session_state.api_key_ok:
                                    with st.spinner("강의 전체 방향성 분석 중..."):
                                        prmt = build_overview_prompt(p_text, target_subj)
                                        res, _ = generate_with_fallback(prmt, st.session_state.text_models)
                                        st.session_state.last_ai_text = res
                                        st.session_state.last_ai_sig = aisig
                                st.markdown(st.session_state.last_ai_text)
                            
                            # 2. 일반 페이지 분석
                            else:
                                if has_jokbo_evidence(rel):
                                    # 섹션 1: 족보 문항 원문 (가장 중요) - 항상 표시
                                    st.markdown("##### 🔥 관련 족보 문항")
                                    for r in rel[:2]:
                                        score = r['score']
                                        src = r['content'].get('source', 'Unknown')
                                        txt = r['content'].get('text', '')[:200]
                                        st.markdown(f"""
                                        <div class="jokbo-item">
                                            <div class="jokbo-source">출처: {src} (유사도 {score:.2f})</div>
                                            {txt}...
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    # AI 분석 실행
                                    aisig = (psig, target_subj)
                                    if aisig != st.session_state.last_ai_sig and st.session_state.api_key_ok:
                                        with st.spinner("족보 기반 심층 분석 중..."):
                                            prmt = build_page_analysis_prompt(p_text, rel, target_subj)
                                            raw_res, _ = generate_with_fallback(prmt, st.session_state.text_models)
                                            
                                            # 결과 파싱 (간단히 섹션별로 나눔)
                                            parts = raw_res.split("[SECTION:")
                                            parsed = {"DIRECTION": "", "TWIN_Q": "", "EXPLANATION": ""}
                                            for p in parts:
                                                if "DIRECTION]" in p: parsed["DIRECTION"] = p.replace("DIRECTION]", "").strip()
                                                elif "TWIN_Q]" in p: parsed["TWIN_Q"] = p.replace("TWIN_Q]", "").strip()
                                                elif "EXPLANATION]" in p: parsed["EXPLANATION"] = p.replace("EXPLANATION]", "").strip()
                                            
                                            st.session_state.last_ai_text = parsed
                                            st.session_state.last_ai_sig = aisig
                                    
                                    # 섹션 2, 3, 4: 심화 분석 (접기/펼치기)
                                    res_dict = st.session_state.last_ai_text
                                    if isinstance(res_dict, dict):
                                        # 공부 방향성 (Expander)
                                        with st.expander("🧭 공부 방향성 보기"):
                                            st.info(res_dict.get("DIRECTION", "분석 중..."))
                                        
                                        # 쌍둥이 문제 (Expander)
                                        with st.expander("🧩 쌍둥이 문제 만들기"):
                                            st.markdown(res_dict.get("TWIN_Q", "생성 중..."))
                                            
                                        # 해설 (Expander)
                                        with st.expander("✅ 해설 및 정답"):
                                            st.markdown(res_dict.get("EXPLANATION", "생성 중..."))
                                    else:
                                        st.write(res_dict) # Fallback if parsing fails
                                        
                                else:
                                    st.info("💡 이 페이지와 직접 연관된 족보 내용은 없습니다.")
                                    st.caption("가볍게 훑고 넘어가셔도 좋습니다.")
                        else:
                            st.info("분석할 텍스트가 없습니다.")

                    # --- Tab B: 채팅 ---
                    with ai_tab2:
                        for msg in st.session_state.chat_history:
                            with st.chat_message(msg["role"]):
                                st.markdown(msg["content"])
                        
                        if prompt := st.chat_input("질문하세요 (예: 이거 시험에 나와?)"):
                            if not st.session_state.api_key_ok: st.error("API Key 필요")
                            else:
                                st.session_state.chat_history.append({"role": "user", "content": prompt})
                                with st.chat_message("user"): st.markdown(prompt)
                                
                                with st.chat_message("assistant"):
                                    with st.spinner("생각 중..."):
                                        if analysis_ready:
                                            chat_prmt = build_chat_prompt(st.session_state.chat_history, p_text, rel, prompt)
                                            response_text, _ = generate_with_fallback(chat_prmt, st.session_state.text_models)
                                        else: response_text = "이 페이지에는 텍스트가 없어 답변하기 어렵습니다."
                                        st.markdown(response_text)
                                        st.session_state.chat_history.append({"role": "assistant", "content": response_text})

        else:
            st.markdown("""
                <div style="height: 400px; display: flex; align-items: center; justify-content: center; color: #ccc; border: 2px dashed #eee; border-radius: 12px; margin-top: 20px;">
                    <h3>상단에서 강의 PDF 파일을 업로드해주세요 📂</h3>
                </div>
            """, unsafe_allow_html=True)


# --- TAB 3: 강의 녹음/분석 ---
with tab3:
    with st.container(border=True):
        st.markdown("#### 🎙️ 강의 녹음/분석")
        st.caption("강의를 바로 녹음하거나 녹음 파일을 업로드하면, AI가 족보 내용과 매칭하여 요약해줍니다.")
        
        c_in, c_out = st.columns(2)
        with c_in:
            sub_t3 = st.selectbox("과목", ["전체"] + sorted({x.get("subject", "") for x in st.session_state.db}), key="t3_s")
            t3_mode = st.radio("입력 방식", ["🎤 직접 녹음", "📂 파일 업로드 / 텍스트"], horizontal=True, label_visibility="collapsed")
            target_text = ""
            
            if t3_mode == "🎤 직접 녹음":
                audio_value = st.audio_input("녹음 시작 버튼을 누르세요")
                if audio_value:
                    st.success("녹음 완료! (분석 준비됨)")
                    if st.button("🚀 녹음 내용 분석하기", type="primary", use_container_width=True, key="btn_audio_analyze"):
                        if not st.session_state.api_key_ok: st.error("API Key 필요")
                        else:
                            with st.spinner("음성을 텍스트로 변환 중..."):
                                transcript = transcribe_audio_gemini(audio_value.getvalue(), st.session_state.api_key)
                                if transcript:
                                    st.session_state.transcribed_text = transcript
                                    target_text = transcript
                                else: st.error("변환 실패")
            else:
                f_txt = st.file_uploader("전사 파일(.txt)", type="txt", key="t3_f")
                area_txt = st.text_area("직접 입력", height=200, placeholder="강의 내용을 입력하세요...")
                if st.button("분석 실행", type="primary", use_container_width=True):
                    target_text = (f_txt.getvalue().decode() if f_txt else area_txt).strip()
            
            if target_text:
                if not st.session_state.api_key_ok: st.error("API Key 필요")
                else:
                    with st.spinner("족보 데이터와 대조하여 분석 중..."):
                        sdb = filter_db_by_subject(sub_t3, st.session_state.db)
                        chks = chunk_transcript(target_text)[:10]
                        rels = [find_relevant_jokbo(c, sdb, top_k=3) for c in chks]
                        pmt = build_transcript_prompt(chks, rels, sub_t3)
                        res, _ = generate_with_fallback(pmt, st.session_state.text_models)
                        st.session_state.tr_res = res
                    st.success("분석 완료!")

        with c_out:
            st.caption("분석 결과")
            if "tr_res" in st.session_state:
                st.info(st.session_state.tr_res)
                if st.session_state.transcribed_text:
                    with st.expander("📝 변환된 전체 텍스트 보기"):
                        st.text(st.session_state.transcribed_text)
            else:
                st.markdown("""<div style="height: 300px; background: #f9f9f9; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: #aaa;">결과가 여기에 표시됩니다.</div>""", unsafe_allow_html=True)
