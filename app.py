# app.py (UI: Yellow Box / Logic: Smart Model Discovery)
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

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; } 
    h1, h2, h3, h4, h5, h6, p, span, div, label, .stMarkdown { color: #1c1c1e !important; }
    .gray-text, .text-sm, .login-desc, small { color: #8e8e93 !important; }
    div.stButton > button p { color: #007aff !important; }
    div.stButton > button[kind="primary"] p { color: #ffffff !important; }
    div[data-baseweb="input"] { background-color: #ffffff !important; border: 1px solid #d1d1d6 !important; color: #1c1c1e !important; }
    div[data-baseweb="input"] input { color: #1c1c1e !important; }
    .block-container { padding: 1rem 2rem !important; max-width: 100% !important; }
    header[data-testid="stHeader"] { display: none; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: transparent; padding: 4px; border-radius: 10px; margin-bottom: 15px; }
    .stTabs [data-baseweb="tab"] { height: 40px; border-radius: 20px; padding: 0 20px; background-color: #ffffff; border: 1px solid #e0e0e0; font-weight: 600; color: #8e8e93 !important; flex-grow: 0; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
    .stTabs [aria-selected="true"] { background-color: #007aff !important; color: #ffffff !important; box-shadow: 0 4px 8px rgba(0,122,255,0.2); border: none; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 20px; border: 1px solid #edf2f7; box-shadow: 0 4px 20px rgba(0,0,0,0.03); 
        background-color: white; padding: 20px; transition: transform 0.2s ease;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover { transform: translateY(-2px); border-color: #007aff; }
    div.stButton > button { border-radius: 12px; font-weight: 600; border: none; background-color: #f2f2f7; height: 3rem; }
    div.stButton > button:hover { background-color: #e5e5ea; transform: scale(0.98); }
    div.stButton > button[kind="primary"] { background-color: #007aff; box-shadow: 0 4px 10px rgba(0,122,255,0.2); }
    div.stButton > button[kind="primary"]:hover { background-color: #0062cc; }
    .login-logo { font-size: 5rem; margin-bottom: 10px; animation: bounce 2s infinite; }
    @keyframes bounce { 0%, 20%, 50%, 80%, 100% {transform: translateY(0);} 40% {transform: translateY(-20px);} 60% {transform: translateY(-10px);} }
    .jokbo-item { background-color: #fffde7; border: 1px solid #fff59d; border-radius: 12px; padding: 16px; margin-bottom: 12px; }
    .jokbo-source { font-size: 0.8rem; color: #f57f17; margin-bottom: 6px; font-weight: 800; }
    .sidebar-subject { padding: 10px 15px; background-color: white; border-radius: 10px; margin-bottom: 8px; font-weight: 600; color: #333; border: 1px solid #f0f0f0; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. Session State
# ==========================================
defaults = {
    "logged_in": False, "db": [], "api_key": None, "api_key_ok": False,
    "text_models": [], "embedding_models": [], "best_text_model": None, "best_embedding_model": None,
    "lecture_doc": None, "lecture_filename": None, "current_page": 0,
    "edit_target_subject": None, "subject_detail_view": None, "t2_selected_subject": None,
    "transcribed_text": "", "chat_history": [],
    "last_page_sig": None, "last_ai_sig": None, "last_ai_text": "", "last_related": []
}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

# ==========================================
# 2. Login
# ==========================================
def login():
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("<div style='height: 15vh;'></div>", unsafe_allow_html=True)
        st.markdown("""<div style="text-align: center;"><div class="login-logo">🩺</div><h1 style="color:#1c1c1e;">Med-Study OS</h1></div>""", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("#### 로그인")
            if st.button("앱 시작하기 (Demo)", type="primary", use_container_width=True):
                st.session_state.logged_in = True
                st.rerun()

def logout():
    st.session_state.logged_in = False
    st.rerun()

# ==========================================
# 3. Helpers & Smart Model Logic
# ==========================================
def ensure_configured():
    if st.session_state.get("api_key"):
        genai.configure(api_key=st.session_state["api_key"])

@st.cache_data(show_spinner=False)
def list_available_models(api_key: str):
    """API 키로 사용 가능한 텍스트 및 임베딩 모델을 자동으로 찾습니다."""
    try:
        genai.configure(api_key=api_key)
        all_models = list(genai.list_models())
        
        text_mods = [m.name for m in all_models if "generateContent" in getattr(m, "supported_generation_methods", [])]
        embed_mods = [m.name for m in all_models if "embedContent" in getattr(m, "supported_generation_methods", [])]
        
        return text_mods, embed_mods
    except Exception as e:
        return [], []

def get_best_model(models, keywords):
    """키워드가 포함된 최신 모델을 우선 선택합니다."""
    if not models: return None
    for k in keywords:
        found = [m for m in models if k in m]
        if found: return found[0]
    return models[0]

# --- Smart Robust Embedding ---
def get_embedding_robust(text: str, status_placeholder=None):
    """
    1. 사용 가능한 모델 리스트에서 임베딩 모델을 찾음.
    2. Rate Limit(429) 발생 시 지능적으로 대기.
    3. 없는 모델(404)은 시도하지 않음.
    """
    text = (text or "").strip()
    if len(text) < 50: return None, "text_too_short"
    ensure_configured()
    
    # 세션에 저장된 임베딩 모델 리스트 활용 (없으면 다시 검색)
    if not st.session_state.embedding_models:
        _, embs = list_available_models(st.session_state.api_key)
        st.session_state.embedding_models = embs
    
    # 우선순위: text-embedding-004 > 004 > embedding-001 순으로 검색
    candidates = st.session_state.embedding_models
    if not candidates:
        return None, "No embedding models available for this API key."
        
    # 우선순위 정렬
    sorted_candidates = sorted(candidates, key=lambda x: 0 if 'text-embedding-004' in x else 1)
    
    max_retries = 5
    base_wait = 3
    last_error_msg = ""

    # 모델 하나씩 시도 (보통 첫번째에서 성공해야 함)
    for model_name in sorted_candidates[:2]: # 상위 2개만 시도
        for attempt in range(max_retries):
            try:
                # API 호출 속도 조절
                time.sleep(1.5) 
                
                # 모델에 따른 파라미터 조정
                if "004" in model_name:
                    res = genai.embed_content(model=model_name, content=text, task_type="retrieval_document")
                else:
                    res = genai.embed_content(model=model_name, content=text)
                    
                if res and "embedding" in res:
                    return res["embedding"], None # 성공
            
            except Exception as e:
                err_msg = str(e)
                last_error_msg = f"{model_name}: {err_msg}"
                
                if "429" in err_msg or "Resource exhausted" in err_msg:
                    wait_time = base_wait * (2 ** attempt) + random.randint(1, 3)
                    if status_placeholder:
                        status_placeholder.warning(f"⚠️ 사용량 많음 ({model_name}). {wait_time}초 대기 중... ({attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                elif "404" in err_msg or "Not Found" in err_msg:
                    # 모델이 없으면 즉시 다음 모델로
                    break
                else:
                    time.sleep(1)
                    
    return None, f"Fail: {last_error_msg}"

def extract_text_from_pdf(uploaded_file):
    try:
        data = uploaded_file.getvalue()
        doc = fitz.open(stream=data, filetype="pdf")
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text() or ""
            pages.append({"page": i + 1, "text": text, "source": uploaded_file.name})
        return pages
    except: return []

def find_relevant_jokbo(query_text, db, top_k=5):
    if not db: return []
    q_emb, _ = get_embedding_robust(query_text)
    if not q_emb: return []
    
    valid = [x for x in db if x.get("embedding")]
    if not valid: return []
    
    sims = cosine_similarity([q_emb], [x["embedding"] for x in valid])[0]
    idxs = np.argsort(sims)[::-1][:top_k]
    return [{"score": float(sims[i]), "content": valid[i]} for i in idxs]

def generate_with_fallback(prompt, model_names):
    ensure_configured()
    # 텍스트 모델도 리스트에서 선택
    target_model = st.session_state.best_text_model or "gemini-1.5-flash"
    try:
        model = genai.GenerativeModel(target_model)
        res = model.generate_content(prompt)
        return res.text, target_model
    except Exception as e:
        raise Exception(f"AI Error ({target_model}): {str(e)}")

# --- Prompts (Same as before) ---
def build_overview_prompt(txt, subj): return f"너는 의대 수석 조교다. '{subj}' 강의록 첫 페이지를 보고 핵심 목표, 족보 기반 공부 전략 3가지, 주의점을 요약해라.\n[내용]\n{txt[:1500]}"
def build_page_analysis_prompt(txt, rel, subj): 
    jokbo = "\n".join([f"- {r['content']['text'][:300]}" for r in rel[:3]])
    return f"의대 조교로서 분석해라. 과목:{subj}\n[관련족보]\n{jokbo}\n[강의내용]\n{txt[:1500]}\n출력형식:\n[SECTION: DIRECTION] 공부방향, 키워드\n[SECTION: TWIN_Q] 족보 변형 문제 1개\n[SECTION: EXPLANATION] 정답 및 해설"
def build_chat_prompt(hist, ctx, rel, q): return f"의대 조교입니다. 강의내용: {ctx[:1000]}\n관련족보: {rel}\n질문: {q}\n답변해주세요."
def build_transcript_prompt(chunks, packs, subj): return f"강의 내용을 족보 기반으로 요약하세요. 과목:{subj}\n(생략)"
def chunk_transcript(text): return [text[i:i+900] for i in range(0, len(text), 900)]
def get_subject_stats(): return {item.get("subject", "기타"): {"count": 0} for item in st.session_state.db} # Simplified
def get_subject_files(subj): 
    files = {}
    for x in st.session_state.db:
        if x.get("subject") == subj: files[x.get("source")] = files.get(x.get("source"), 0) + 1
    return files

# ==========================================
# 4. Main UI
# ==========================================
if not st.session_state.logged_in:
    login()
    st.stop()

with st.sidebar:
    st.markdown("### 👤 내 프로필")
    with st.container(border=True):
        st.markdown("**Student Admin** (본과 2학년)")
        if st.button("로그아웃", use_container_width=True): logout()
    
    st.markdown("### ⚙️ 설정 (필수)")
    with st.container(border=True):
        api_key = st.text_input("Gemini API Key", type="password", key="api_key_input")
        if api_key:
            st.session_state.api_key = api_key
            
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
                        
                        # Best model selection
                        st.session_state.best_text_model = get_best_model(t_mods, ["flash", "pro"])
                        st.session_state.best_embedding_model = get_best_model(e_mods, ["text-embedding-004", "004"])
                        
                        st.success(f"✅ 연결 성공!")
                        st.caption(f"텍스트 모델: {st.session_state.best_text_model}")
                        st.caption(f"임베딩 모델: {st.session_state.best_embedding_model}")
                    else:
                        st.error("🚫 사용 가능한 모델을 찾을 수 없습니다. (API Key 권한 확인)")

    st.markdown("### 📊 DB 현황")
    st.metric("총 학습 페이지", len(st.session_state.db))
    if st.button("DB 초기화"): 
        st.session_state.db = []
        st.rerun()

st.title("Med-Study OS")
tab1, tab2, tab3 = st.tabs(["📂 족보 관리", "📖 강의 분석", "🎙️ 강의 녹음/분석"])

with tab1:
    col_up, col_list = st.columns([1, 2])
    with col_up:
        with st.container(border=True):
            st.markdown("#### ➕ 족보 학습")
            subj = st.text_input("과목명", value="직접입력")
            files = st.file_uploader("PDF 업로드", accept_multiple_files=True, type="pdf")
            
            if st.button("학습 시작", type="primary", use_container_width=True):
                if not st.session_state.api_key_ok: st.error("왼쪽 설정에서 '모델 목록 불러오기'를 먼저 해주세요!")
                elif not files: st.warning("파일을 선택하세요.")
                else:
                    bar = st.progress(0)
                    log_area = st.empty()
                    
                    # Log display
                    logs = []
                    def log(m): 
                        logs.append(m)
                        log_area.markdown("\n".join([f"- {l}" for l in logs[-5:]]))

                    new_data = []
                    for i, f in enumerate(files):
                        log(f"📂 {f.name} 처리 중...")
                        pages = extract_text_from_pdf(f)
                        
                        success = 0
                        for p in pages:
                            emb, err = get_embedding_robust(p["text"], st.empty())
                            if emb:
                                p["embedding"] = emb
                                p["subject"] = subj
                                new_data.append(p)
                                success += 1
                            elif "429" in str(err):
                                log(f"⚠️ {f.name} 일부 페이지 스킵 (사용량 초과)")
                        
                        log(f"✅ {f.name}: {success}페이지 학습 완료")
                        bar.progress((i+1)/len(files))
                    
                    if new_data:
                        st.session_state.db.extend(new_data)
                        st.success(f"총 {len(new_data)}페이지 저장 완료!")
                        time.sleep(1)
                        st.rerun()

    with col_list:
        st.markdown("#### 📚 학습된 데이터")
        db_subjs = sorted({x["subject"] for x in st.session_state.db})
        if not db_subjs: st.info("데이터가 없습니다.")
        for s in db_subjs:
            cnt = len([x for x in st.session_state.db if x["subject"] == s])
            with st.container(border=True):
                c1, c2 = st.columns([5, 1])
                c1.markdown(f"**📘 {s}** ({cnt} pages)")
                if c2.button("보기", key=f"v_{s}"):
                    st.session_state.t2_selected_subject = s
                    st.rerun()

# (Tab 2, Tab 3 omitted for brevity, logic remains similar but uses session state models)
with tab2:
    st.markdown("#### 📖 실시간 강의 분석")
    if not st.session_state.t2_selected_subject:
        st.info("족보 관리 탭에서 과목의 [보기] 버튼을 눌러주세요.")
    else:
        st.markdown(f"**선택된 과목: {st.session_state.t2_selected_subject}**")
        l_file = st.file_uploader("강의 PDF 열기", type="pdf", key="l_pdf")
        if l_file and l_file.name != st.session_state.lecture_filename:
            st.session_state.lecture_doc = fitz.open(stream=l_file.getvalue(), filetype="pdf")
            st.session_state.lecture_filename = l_file.name
            st.session_state.current_page = 0
        
        if st.session_state.lecture_doc:
            doc = st.session_state.lecture_doc
            c_view, c_ai = st.columns([1.5, 1])
            with c_view:
                if st.button("◀ 이전"): st.session_state.current_page = max(0, st.session_state.current_page-1)
                st.image(doc.load_page(st.session_state.current_page).get_pixmap().tobytes(), use_container_width=True)
                if st.button("다음 ▶"): st.session_state.current_page = min(len(doc)-1, st.session_state.current_page+1)
            
            with c_ai:
                txt = doc.load_page(st.session_state.current_page).get_text()
                if st.button("이 페이지 분석 (AI)", type="primary"):
                    if not st.session_state.api_key_ok: st.error("API 연결 필요")
                    else:
                        with st.spinner("분석 중..."):
                            related = find_relevant_jokbo(txt, [x for x in st.session_state.db if x["subject"] == st.session_state.t2_selected_subject])
                            res, _ = generate_with_fallback(build_page_analysis_prompt(txt, related, st.session_state.t2_selected_subject), [])
                            st.markdown(res)
                            if related:
                                with st.expander("참고한 족보"):
                                    for r in related[:2]: st.caption(f"{r['content']['text'][:100]}...")

with tab3:
    st.info("강의 녹음 기능은 Tab 2와 동일한 AI 로직을 사용합니다.")
