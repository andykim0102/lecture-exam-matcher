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

# ==========================================
# 0. Page config & Custom CSS
# ==========================================
st.set_page_config(page_title="Med-Study OS", layout="wide", page_icon="🩺")

# 실제 앱 느낌을 위한 커스텀 CSS 주입
st.markdown("""
<style>
    /* 1. 기본 배경 및 폰트 설정 (Pure White App Style) */
    .stApp {
        background-color: #ffffff;
    }
    html, body, [class*="css"]  {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #1c1c1e;
    }
    
    /* 2. 상단 여백 제거하여 앱 헤더처럼 보이게 하기 */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* 3. 탭 스타일링 (iOS Segmented Control 느낌) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background-color: #f2f2f7;
        padding: 4px;
        border-radius: 10px;
        margin-bottom: 25px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 36px;
        border-radius: 7px;
        padding: 0 20px;
        background-color: transparent;
        border: none;
        font-weight: 500;
        color: #8e8e93;
        flex-grow: 1; /* 탭 균등 분할 */
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #000000 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        font-weight: 600;
    }

    /* 4. 카드 컨테이너 스타일 (Streamlit Native Container with Border) */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px;
        border: 1px solid #f0f0f0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.03);
        background-color: white;
    }

    /* 5. 버튼 스타일 (앱 버튼처럼) */
    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        border: none;
        box-shadow: none;
        background-color: #f2f2f7;
        color: #007aff; /* iOS Blue */
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        background-color: #e5e5ea;
        transform: scale(0.98);
    }
    /* Primary 버튼 강조 */
    div.stButton > button[kind="primary"] {
        background-color: #007aff;
        color: white;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #0062cc;
    }

    /* 6. 로그인 화면 스타일 */
    .login-logo { font-size: 5rem; margin-bottom: 10px; animation: bounce 2s infinite; }
    @keyframes bounce { 0%, 20%, 50%, 80%, 100% {transform: translateY(0);} 40% {transform: translateY(-20px);} 60% {transform: translateY(-10px);} }
    
    /* 7. 텍스트 유틸리티 */
    .text-sm { font-size: 0.85rem; color: #8e8e93; }
    .text-bold { font-weight: 700; color: #1c1c1e; }
    .badge { 
        background-color: #eef2ff; color: #4b89dc; 
        padding: 4px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; 
    }

    /* 8. 파일 업로더 깔끔하게 */
    div[data-testid="stFileUploader"] {
        padding: 15px;
        border: 1px dashed #d1d1d6;
        border-radius: 12px;
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

# For Edit Mode in Tab 1
if "edit_target_subject" not in st.session_state:
    st.session_state.edit_target_subject = None

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
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        st.markdown("<div style='height: 15vh;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="text-align: center;">
                <div class="login-logo">🩺</div>
                <h1 style="font-weight: 800; margin-bottom: 0;">Med-Study OS</h1>
                <p style="color: #8e8e93; margin-bottom: 30px;">당신의 스마트한 의대 학습 파트너</p>
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
            
            st.markdown(
                "<div style='text-align:center; margin-top:15px; font-size:0.8rem; color:#c7c7cc;'>Demo Access: admin / 1234</div>", 
                unsafe_allow_html=True
            )

def logout():
    st.session_state.logged_in = False
    st.rerun()


# ==========================================
# 3. Helpers & Data Logic
# ==========================================
def rename_subject(old_name, new_name):
    """DB 내의 모든 해당 과목명을 변경"""
    count = 0
    for item in st.session_state.db:
        if item.get("subject") == old_name:
            item["subject"] = new_name
            count += 1
    return count

def get_subject_stats():
    """과목별 통계 데이터 생성 (패턴 수, 업데이트 시간 시뮬레이션)"""
    stats = {}
    for item in st.session_state.db:
        subj = item.get("subject", "기타")
        if subj not in stats:
            # 시뮬레이션용 랜덤 시간 (실제 앱에선 timestamp 필드 필요)
            rand_min = random.randint(1, 59)
            stats[subj] = {"count": 0, "last_updated": f"{rand_min}분 전"}
        stats[subj]["count"] += 1
    return stats

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

# Prompts
def build_ta_prompt(lecture_text: str, related: list[dict], subject: str):
    ctx = "\n".join([f"- [{r['content']['source']} p{r['content']['page']}] {r['content']['text'][:400]}" for r in related[:3]])
    return f"""
    당신은 의대 조교입니다. 학생이 공부 중인 강의 내용과 관련된 족보(기출) 내용을 바탕으로 핵심을 짚어주세요.
    과목: {subject}
    [관련 족보 내용] {ctx}
    [현재 강의 내용] {lecture_text}
    출력 형식:
    1. 💡 한줄 요약: (족보와 연관된 핵심 내용 한 문장)
    2. 🎯 출제 포인트 TOP 3: (짧게)
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
        with col_p1:
            st.markdown("## 👨‍⚕️")
        with col_p2:
            st.markdown("**Student Admin**")
            st.caption("본과 2학년")
        
        if st.button("로그아웃", use_container_width=True):
            logout()

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
                else:
                    st.error("모델 권한 없음")
            except Exception as e:
                st.error(f"키 오류: {e}")
        else:
            st.warning("API Key 입력 필요")
            
    st.markdown("### 📊 DB 현황")
    with st.container(border=True):
        st.metric("총 학습 페이지", len(st.session_state.db))
        if st.button("DB 초기화", use_container_width=True):
            st.session_state.db = []
            st.rerun()

# --- 메인 콘텐츠 ---
st.title("Med-Study OS")

# 탭 구성
tab1, tab2, tab3 = st.tabs(["📂 족보 관리", "📖 강의 분석", "🎙️ 전사 분석"])

# --- TAB 1: 족보 관리 (카드 UI + 수정 기능) ---
with tab1:
    col_upload, col_list = st.columns([1, 2])
    
    # 1-1. 업로드 패널 (왼쪽)
    with col_upload:
        with st.container(border=True):
            st.markdown("#### ➕ 족보 추가")
            st.caption("PDF 파일을 업로드하여 AI 학습")
            
            up_subj = st.selectbox("과목", ["해부학", "생리학", "약리학", "직접입력"], key="up_subj")
            if up_subj == "직접입력":
                up_subj_custom = st.text_input("과목명 입력", placeholder="예: 병리학")
                final_subj = up_subj_custom if up_subj_custom else "기타"
            else:
                final_subj = up_subj
                
            files = st.file_uploader("PDF 선택", accept_multiple_files=True, type="pdf", label_visibility="collapsed")
            max_p = st.number_input("학습 페이지 수 제한", 1, 500, 60)
            
            if st.button("학습 시작", type="primary", use_container_width=True):
                if not st.session_state.api_key_ok:
                    st.error("API Key 필요")
                elif not files:
                    st.warning("파일 필요")
                else:
                    prog = st.progress(0)
                    new_db = []
                    for i, f in enumerate(files):
                        pgs = extract_text_from_pdf(f)[:max_p]
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

    # 1-2. 과목 카드 리스트 (오른쪽)
    with col_list:
        st.markdown("#### 📚 내 학습 데이터")
        stats = get_subject_stats()
        
        if not stats:
            st.info("등록된 족보가 없습니다. 왼쪽에서 추가해주세요.")
        
        # Grid Layout for Cards
        subjects = sorted(stats.keys())
        
        # 2열 그리드로 표시
        for i in range(0, len(subjects), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(subjects):
                    subj_name = subjects[i+j]
                    subj_data = stats[subj_name]
                    
                    with cols[j]:
                        # 카드 컨테이너
                        with st.container(border=True):
                            # 헤더: 과목명 + 수정 버튼
                            c_head_1, c_head_2 = st.columns([4, 1])
                            
                            # 수정 모드인지 확인
                            is_editing = (st.session_state.edit_target_subject == subj_name)
                            
                            with c_head_1:
                                if is_editing:
                                    new_name_input = st.text_input("새 이름", value=subj_name, key=f"edit_in_{subj_name}", label_visibility="collapsed")
                                else:
                                    st.markdown(f"### {subj_name}")
                            
                            with c_head_2:
                                if is_editing:
                                    if st.button("💾", key=f"save_{subj_name}"):
                                        if new_name_input and new_name_input != subj_name:
                                            rename_subject(subj_name, new_name_input)
                                            st.session_state.edit_target_subject = None
                                            st.toast("수정 완료!")
                                            st.rerun()
                                        else:
                                            st.session_state.edit_target_subject = None
                                            st.rerun()
                                else:
                                    if st.button("✏️", key=f"edit_btn_{subj_name}"):
                                        st.session_state.edit_target_subject = subj_name
                                        st.rerun()

                            if not is_editing:
                                st.markdown("---")
                                st.markdown(f"**⚡ 분석된 패턴:** {subj_data['count']}건")
                                st.markdown(f"<span class='text-sm'>🕒 최근 업데이트: {subj_data['last_updated']}</span>", unsafe_allow_html=True)


# --- TAB 2: 강의 분석 ---
with tab2:
    with st.container(border=True):
        st.markdown("#### 📖 실시간 강의 분석")
        
        c_tool, c_view = st.columns([1, 2])
        with c_tool:
            if not st.session_state.db:
                st.warning("DB 데이터가 없습니다.")
            
            all_subjs = sorted({x.get("subject", "") for x in st.session_state.db})
            s_pick = st.selectbox("과목 선택", ["전체"] + all_subjs, key="t2_s")
            l_file = st.file_uploader("강의 PDF", type="pdf", key="t2_f")
            
            st.info("PDF를 업로드하면 AI가 족보와 매칭되는 내용을 실시간으로 분석해줍니다.")

        with c_view:
            if l_file:
                if st.session_state.lecture_filename != l_file.name:
                    st.session_state.lecture_doc = fitz.open(stream=l_file.getvalue(), filetype="pdf")
                    st.session_state.lecture_filename = l_file.name
                    st.session_state.current_page = 0
                    st.session_state.last_page_sig = None
                
                doc = st.session_state.lecture_doc
                
                # 뷰어 컨트롤
                col_nav, col_dummy = st.columns([2, 1])
                with col_nav:
                    c1, c2, c3 = st.columns([1, 2, 1])
                    if c1.button("◀", use_container_width=True):
                        if st.session_state.current_page > 0: st.session_state.current_page -= 1
                    c2.markdown(f"<div style='text-align:center; padding-top:5px;'>Page {st.session_state.current_page+1} / {len(doc)}</div>", unsafe_allow_html=True)
                    if c3.button("▶", use_container_width=True):
                        if st.session_state.current_page < len(doc)-1: st.session_state.current_page += 1
                
                # PDF & AI Analysis
                c_pdf, c_ai = st.columns(2)
                
                with c_pdf:
                    page = doc.load_page(st.session_state.current_page)
                    pix = page.get_pixmap(dpi=150)
                    st.image(Image.frombytes("RGB", [pix.width, pix.height], pix.samples), use_container_width=True)
                    p_text = page.get_text() or ""
                
                with c_ai:
                    with st.container(border=True):
                        st.markdown("**🤖 조교 분석**")
                        if not p_text.strip():
                            st.caption("텍스트 없음")
                        else:
                            # Analysis Logic
                            psig = hash(p_text)
                            if psig != st.session_state.last_page_sig:
                                st.session_state.last_page_sig = psig
                                sub_db = filter_db_by_subject(s_pick, st.session_state.db)
                                st.session_state.last_related = find_relevant_jokbo(p_text, sub_db)
                                st.session_state.last_ai_sig = None
                            
                            rel = st.session_state.last_related
                            if has_jokbo_evidence(rel):
                                aisig = (psig, s_pick)
                                if aisig != st.session_state.last_ai_sig and st.session_state.api_key_ok:
                                    with st.spinner("분석 중..."):
                                        prmt = build_ta_prompt(p_text, rel, s_pick)
                                        res, _ = generate_with_fallback(prmt, st.session_state.text_models)
                                        st.session_state.last_ai_text = res
                                        st.session_state.last_ai_sig = aisig
                                
                                st.markdown(st.session_state.last_ai_text)
                            else:
                                st.info("관련 족보 없음")
            else:
                st.markdown(
                    """
                    <div style="height: 300px; display: flex; align-items: center; justify-content: center; color: #ccc; border: 2px dashed #eee; border-radius: 12px;">
                        PDF 파일을 선택하면 여기에 표시됩니다.
                    </div>
                    """, unsafe_allow_html=True
                )


# --- TAB 3: 전사 분석 ---
with tab3:
    with st.container(border=True):
        st.markdown("#### 🎙️ 강의 녹음/전사 분석")
        
        c_in, c_out = st.columns(2)
        
        with c_in:
            st.caption("텍스트 입력")
            sub_t3 = st.selectbox("과목", ["전체"] + sorted({x.get("subject", "") for x in st.session_state.db}), key="t3_s")
            f_txt = st.file_uploader("전사 파일(.txt)", type="txt", key="t3_f")
            area_txt = st.text_area("직접 입력", height=200, placeholder="강의 내용을 입력하세요...")
            
            if st.button("분석 실행", type="primary", use_container_width=True):
                target = (f_txt.getvalue().decode() if f_txt else area_txt).strip()
                if not target:
                    st.error("내용 필요")
                elif not st.session_state.api_key_ok:
                    st.error("API Key 필요")
                else:
                    with st.spinner("분석 중..."):
                        sdb = filter_db_by_subject(sub_t3, st.session_state.db)
                        chks = chunk_transcript(target)[:10]
                        rels = [find_relevant_jokbo(c, sdb, top_k=3) for c in chks]
                        pmt = build_transcript_prompt(chks, rels, sub_t3)
                        res, _ = generate_with_fallback(pmt, st.session_state.text_models)
                        st.session_state.tr_res = res
                    st.success("완료!")
        
        with c_out:
            st.caption("분석 결과")
            if "tr_res" in st.session_state:
                st.info(st.session_state.tr_res)
            else:
                st.markdown(
                    """
                    <div style="height: 300px; background: #f9f9f9; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: #aaa;">
                        결과가 여기에 표시됩니다.
                    </div>
                    """, unsafe_allow_html=True
                )
