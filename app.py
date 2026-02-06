# app.py
# ==============================================================================
#  Med-Study OS: 의대생을 위한 스마트 학습 어시스턴트 (Full Version)
#  기능: 족보 PDF 분석, 실시간 강의 매칭, 음성 녹음 요약, AI 질의응답
#  업데이트: AI JSON 출력 모드 적용, 텍스트 가독성 포매터 추가, UI 전면 개편
# ==============================================================================

import time
import re
import json
import random
import numpy as np
import fitz  # PyMuPDF (PDF 처리 라이브러리)
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

# 프리미엄 디자인 CSS 적용
st.markdown("""
<style>
    /* 전체 폰트 및 배경 설정 */
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6, span, div, label, .stMarkdown {
        color: #2c3e50 !important;
    }
    
    .gray-text {
        color: #8e8e93 !important;
    }

    /* 카드 컨테이너 디자인 (그림자 효과) */
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

    /* 로그인 화면 로고 애니메이션 */
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

    /* 탭(Tab) 스타일 커스터마이징 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        border-radius: 20px;
        padding: 0 24px;
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        font-weight: 600;
        color: #8e8e93 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #007aff !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(0,122,255,0.3);
        border: none;
    }

    /* 배지(Badge) 스타일 */
    .badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 5px 12px;
        border-radius: 99px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-right: 6px;
        margin-bottom: 8px;
        letter-spacing: -0.3px;
        transition: 0.2s;
    }
    
    .badge:hover {
        transform: scale(1.05);
    }
    
    .badge-blue { background-color: #e3f2fd; color: #1565c0; border: 1px solid #bbdefb; }
    .badge-red { background-color: #ffebee; color: #c62828; border: 1px solid #ffcdd2; }
    .badge-gray { background-color: #f5f5f5; color: #616161; border: 1px solid #eeeeee; }
    .badge-green { background-color: #e8f5e9; color: #2e7d32; border: 1px solid #c8e6c9; }
    
    /* 문제 텍스트 스타일 */
    .q-header {
        font-size: 1.1rem;
        font-weight: 800;
        color: #1a1a1a !important;
        margin-top: 8px;
        margin-bottom: 12px;
        line-height: 1.4;
    }
    
    .q-body {
        font-size: 0.95rem;
        color: #495057 !important;
        line-height: 1.8;
        background-color: #fafafa;
        padding: 18px;
        border-radius: 12px;
        margin-bottom: 16px;
        border: 1px solid #f1f3f5;
        white-space: pre-wrap; /* 줄바꿈 유지 */
        font-family: 'Pretendard', sans-serif;
    }

    /* 점선 구분선 */
    .dashed-line {
        border-top: 2px dashed #e0e0e0;
        margin: 20px 0;
        width: 100%;
        height: 0;
    }

    /* 확장(Expander) 및 채팅창 스타일 */
    .streamlit-expanderHeader {
        font-size: 0.9rem;
        font-weight: 600;
        color: #555;
        background-color: #fff;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 10px 16px;
    }
    
    .stChatMessage {
        background-color: #ffffff;
        border: 1px solid #f0f0f0;
        border-radius: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    }
    
    /* 버튼 스타일 */
    div.stButton > button {
        border-radius: 12px;
        font-weight: 600;
        border: none;
        height: 3rem;
        transition: 0.2s;
    }
    
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #007aff 0%, #0062cc 100%);
        box-shadow: 0 4px 12px rgba(0,122,255,0.3);
    }
    
    div.stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 16px rgba(0,122,255,0.4);
        transform: scale(1.01);
    }
    
    /* 파일 업로더 스타일 */
    div[data-testid="stFileUploader"] {
        padding: 20px;
        border: 2px dashed #d1d1d6;
        border-radius: 16px;
        background-color: #ffffff;
    }

    /* 레이아웃 여백 조정 */
    .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    
    header {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 2. 세션 상태 초기화 (Session State Initialization)
# ------------------------------------------------------------------------------
# 로그인 상태
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# 학습 데이터베이스 (벡터 DB 역할)
if "db" not in st.session_state:
    st.session_state.db = []

# API 키 및 모델 설정
if "api_key" not in st.session_state:
    st.session_state.api_key = None

if "api_key_ok" not in st.session_state:
    st.session_state.api_key_ok = False

if "text_models" not in st.session_state:
    st.session_state.text_models = []

if "embedding_models" not in st.session_state:
    st.session_state.embedding_models = []

if "best_text_model" not in st.session_state:
    st.session_state.best_text_model = None

if "best_embedding_model" not in st.session_state:
    st.session_state.best_embedding_model = None

# 강의 PDF 및 분석 상태
if "lecture_doc" not in st.session_state:
    st.session_state.lecture_doc = None

if "lecture_filename" not in st.session_state:
    st.session_state.lecture_filename = None

if "current_page" not in st.session_state:
    st.session_state.current_page = 0

# UI 제어 변수
if "edit_target_subject" not in st.session_state:
    st.session_state.edit_target_subject = None

if "subject_detail_view" not in st.session_state:
    st.session_state.subject_detail_view = None

if "t2_selected_subject" not in st.session_state:
    st.session_state.t2_selected_subject = None

# 녹음 및 텍스트 데이터
if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# AI 분석 결과 캐싱 (중복 호출 방지)
if "last_page_sig" not in st.session_state:
    st.session_state.last_page_sig = None

if "last_ai_sig" not in st.session_state:
    st.session_state.last_ai_sig = None

if "last_ai_data" not in st.session_state:
    st.session_state.last_ai_data = None  # JSON 결과를 저장

if "last_related" not in st.session_state:
    st.session_state.last_related = []

if "tr_res" not in st.session_state:
    st.session_state.tr_res = None


# ------------------------------------------------------------------------------
# 3. 로그인 및 인증 로직
# ------------------------------------------------------------------------------
def login():
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("<div style='height: 15vh;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="text-align: center;">
                <div class="login-logo">🩺</div>
                <h1 style="font-weight: 800; color: #1c1c1e;">Med-Study OS</h1>
                <p class="login-desc" style="color: #8e8e93; font-size: 1.1rem;">
                    의대생을 위한 스마트 족보 분석기
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        with st.container(border=True):
            st.markdown("#### 로그인")
            # username = st.text_input("아이디", placeholder="admin") # 단순화 위해 생략 가능
            password = st.text_input("비밀번호", type="password", placeholder="1234")
            
            if st.button("앱 시작하기", type="primary", use_container_width=True):
                if password == "1234":
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("비밀번호가 틀렸습니다. (Demo: 1234)")
            
            st.markdown("<div style='text-align:center; margin-top:15px; font-size:0.8rem; color:#c7c7cc;'>Demo Access: 1234</div>", unsafe_allow_html=True)

def logout():
    st.session_state.logged_in = False
    st.rerun()


# ------------------------------------------------------------------------------
# 4. 헬퍼 함수 & AI 모델 로직
# ------------------------------------------------------------------------------

def ensure_configured():
    """API 키가 설정되어 있는지 확인하고 Gemini를 설정합니다."""
    if st.session_state.get("api_key"):
        genai.configure(api_key=st.session_state["api_key"])

@st.cache_data(show_spinner=False)
def list_available_models(api_key: str):
    """현재 API 키로 사용 가능한 모델 목록을 가져옵니다."""
    try:
        genai.configure(api_key=api_key)
        all_models = list(genai.list_models())
        
        text_mods = [m.name for m in all_models if "generateContent" in getattr(m, "supported_generation_methods", [])]
        embed_mods = [m.name for m in all_models if "embedContent" in getattr(m, "supported_generation_methods", [])]
        
        return text_mods, embed_mods
    except Exception as e:
        return [], []

def get_best_model(models, keywords):
    """키워드를 기반으로 가장 적합한 모델을 선택합니다."""
    if not models: return None
    for k in keywords:
        found = [m for m in models if k in m]
        if found: return found[0]
    return models[0]

# --- [Text Beautifier] 족보 텍스트 정리 함수 ---
def clean_jokbo_text(text):
    """
    OCR로 읽어온 족보 텍스트의 가독성을 높여줍니다.
    줄바꿈 오류 수정, 보기(①, ②) 정리, 문항 번호 강조 등
    """
    if not text: return ""
    
    # 1. 과도한 줄바꿈 제거 (3개 이상 -> 2개로)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 2. 문항 번호 강조 (예: "1. " -> "**1.** ")
    text = re.sub(r'(?m)^(\d+)\.', r'**\1.**', text)
    
    # 3. 보기 가독성 개선 (①, (1) 등이 앞 문장과 붙어있으면 줄바꿈)
    # 예: "설명이다. ①" -> "설명이다.\n①"
    text = re.sub(r'(?<!\n)(①|②|③|④|⑤|❶|❷|❸|❹|❺|\(1\)|\(2\)|\(3\)|\(4\)|\(5\))', r'\n\1', text)
    
    # 4. 불필요한 PDF 페이지 번호 등 제거 (줄에 숫자만 있는 경우)
    text = re.sub(r'(?m)^\d+\s*$', '', text)
    
    return text.strip()

# --- [Robust Embedding] 견고한 임베딩 함수 ---
def get_embedding_robust(text: str, status_placeholder=None):
    """
    API 제한(Rate Limit)을 고려하여 임베딩을 수행합니다.
    여러 모델을 순차적으로 시도하고, 실패 시 대기 후 재시도합니다.
    """
    text = (text or "").strip()
    if len(text) < 50: 
        return None, "text_too_short"
        
    text = text[:10000] # 길이 제한 안전장치
    ensure_configured()
    
    # 세션에 저장된 임베딩 모델 리스트 활용 (없으면 다시 검색)
    if not st.session_state.embedding_models:
        _, embs = list_available_models(st.session_state.api_key)
        st.session_state.embedding_models = embs
    
    candidates = st.session_state.embedding_models
    if not candidates:
        return None, "No embedding models available."
        
    # 우선순위: text-embedding-004 (최신) > 004 > embedding-001
    sorted_candidates = sorted(candidates, key=lambda x: 0 if 'text-embedding-004' in x else 1)
    
    max_retries = 3
    
    for model_name in sorted_candidates[:2]: # 상위 2개 모델만 시도
        for attempt in range(max_retries):
            try:
                time.sleep(1.2) # API 호출 간격 조절
                
                if "004" in model_name:
                    res = genai.embed_content(model=model_name, content=text, task_type="retrieval_document")
                else:
                    res = genai.embed_content(model=model_name, content=text)
                    
                if res and "embedding" in res:
                    return res["embedding"], None # 성공
            
            except Exception as e:
                err_msg = str(e)
                # Rate Limit 에러 처리
                if "429" in err_msg or "Resource exhausted" in err_msg:
                    wait_time = 2 * (attempt + 1)
                    if status_placeholder:
                        status_placeholder.caption(f"⚠️ {model_name}: 사용량 많음. {wait_time}초 대기...")
                    time.sleep(wait_time)
                # 모델 없음 에러 처리
                elif "404" in err_msg or "Not Found" in err_msg:
                    break 
                else:
                    time.sleep(1)
                    
    return None, "API Error"

def filter_db_by_subject(subject: str, db: list[dict]):
    """선택한 과목의 데이터만 필터링합니다."""
    if not db: return []
    if subject in ["전체", "ALL", ""]: return db
    return [x for x in db if x.get("subject") == subject]

def find_relevant_jokbo(query_text: str, db: list[dict], top_k: int = 10):
    """
    현재 페이지와 가장 유사한 족보를 검색합니다.
    top_k를 10으로 설정하여 빈도 분석의 정확도를 높입니다.
    """
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

# --- [AI Generation - JSON Mode] 스마트 답변 생성 ---
def generate_json_response(prompt: str):
    """
    AI에게 JSON 형식으로 답변을 요청하여, 
    정답/해설/문제 등을 정확하게 파싱합니다. (엉망인 답변 방지)
    """
    ensure_configured()
    target_model = st.session_state.best_text_model or "gemini-1.5-flash"
    
    try:
        # JSON 모드 설정 (Gemini 1.5 Flash 이상 지원)
        config = genai.GenerationConfig(
            temperature=0.3,
            response_mime_type="application/json"
        )
        model = genai.GenerativeModel(target_model, generation_config=config)
        res = model.generate_content(prompt)
        
        # JSON 파싱
        return json.loads(res.text)
    except Exception as e:
        # JSON 모드 실패 시 fallback: 일반 텍스트에서 정규식으로 추출
        try:
            model = genai.GenerativeModel(target_model)
            res = model.generate_content(prompt)
            match = re.search(r'\{.*\}', res.text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            else:
                return {
                    "explanation": res.text, 
                    "direction": "분석 실패 (텍스트 모드)", 
                    "twin_question": "생성 실패"
                }
        except:
            return {
                "explanation": "AI 연결 오류가 발생했습니다.", 
                "direction": "오류", 
                "twin_question": "오류"
            }

def generate_text_response(prompt: str):
    """일반 텍스트 생성 (채팅용)"""
    ensure_configured()
    target_model = st.session_state.best_text_model or "gemini-1.5-flash"
    try:
        model = genai.GenerativeModel(target_model)
        res = model.generate_content(prompt)
        return res.text
    except Exception as e:
        return f"AI 응답 오류: {str(e)}"

# --- [Enhanced OCR] 향상된 이미지 인식 ---
def transcribe_image_to_text(image, api_key):
    """
    이미지에서 텍스트를 추출할 때, 
    '시험지 형식(줄바꿈 등)'을 유지하도록 프롬프트를 강화했습니다.
    """
    try:
        genai.configure(api_key=api_key)
        target_model = "gemini-1.5-flash" # 이미지는 Flash가 빠르고 정확함
        model = genai.GenerativeModel(target_model)
        
        response = model.generate_content([
            "Extract all text from this image exactly as is. Preserve the line breaks for each option (①, ②, etc.). Format it structured like a standard exam paper.",
            image
        ])
        return response.text
    except:
        return None

def transcribe_audio_gemini(audio_bytes, api_key):
    """오디오 파일을 텍스트로 변환합니다."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([
            "Please transcribe the following audio file into text accurately.",
            {"mime_type": "audio/wav", "data": audio_bytes}
        ])
        return response.text
    except:
        return None

# --- [Metadata Parser] 메타데이터 추출 ---
def parse_metadata_from_filename(filename):
    """파일명에서 연도, 학기, 시험 종류를 추출하여 태그로 만듭니다."""
    year = ""
    exam_type = ""
    
    # 연도 추출 (20xx)
    year_match = re.search(r'(20\d{2})', filename)
    if year_match: year = year_match.group(1)
    
    # 시험 종류 추출
    if "중간" in filename: exam_type = "중간"
    elif "기말" in filename: exam_type = "기말"
    elif "모의" in filename: exam_type = "모의"
    elif "국시" in filename: exam_type = "국시"
    
    full_meta = f"{year} {exam_type}".strip()
    return full_meta if full_meta else "기출"

# ------------------------------------------------------------------------------
# 5. 프롬프트 빌더 (Prompt Builders)
# ------------------------------------------------------------------------------

def build_page_analysis_prompt_json(lecture_text, related_jokbo, subject):
    """
    [JSON 모드 전용] 강의 내용과 족보를 분석하여 JSON 형식으로 요청합니다.
    """
    jokbo_ctx = "\n".join([f"- {r['content']['text'][:300]}" for r in related_jokbo[:3]])
    return f"""
    You are a medical tutor. Analyze the lecture content and related exam questions (Jokbo).
    Subject: {subject}
    
    [Related Exam Questions (Reference)]
    {jokbo_ctx}
    
    [Lecture Content (Current Page)]
    {lecture_text[:1500]}
    
    Please output in JSON format with the following keys. 
    The content MUST be in Korean.
    
    {{
        "direction": "Write 1-2 sentences on what key concepts to memorize for the exam based on this page.",
        "twin_question": "Create 1 new multiple-choice question similar to the reference questions. Include options (1~5).",
        "explanation": "Provide the correct answer and a detailed explanation for the twin question."
    }}
    """

def build_overview_prompt(txt, subj):
    return f"과목: {subj}\n내용: {txt[:1500]}\n이 강의의 핵심 목표와 족보 기반 공부 전략 3가지를 요약해줘."

def build_chat_prompt(hist, ctx, rel, q):
    jokbo_ctx = "\n".join([f"- {r['content']['text'][:300]}" for r in rel[:3]])
    return f"질문: {q}\n강의내용: {ctx[:1000]}\n족보: {jokbo_ctx}\n답변해주세요."

def build_transcript_prompt(chunks, packs, subj):
    return f"강의 녹음 내용을 족보와 연결하여 요약해주세요. 과목: {subj}"

def chunk_transcript(text):
    return [text[i:i+900] for i in range(0, len(text), 900)]

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


# ==============================================================================
# 6. 메인 앱 UI 구조 (Main App UI)
# ==============================================================================

if not st.session_state.logged_in:
    login()
    st.stop()

# --- [사이드바] 프로필 및 설정 ---
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

# --- [메인] 탭 구성 ---
st.title("Med-Study OS")
tab1, tab2, tab3 = st.tabs(["📂 족보 관리", "📖 강의 분석", "🎙️ 강의 녹음/분석"])

# ------------------------------------------------------------------------------
# TAB 1: 족보 관리 (파일 업로드 및 학습)
# ------------------------------------------------------------------------------
with tab1:
    # 상세 보기 모드
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
    
    # 기본 목록 모드
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
                        # 학습 로직 시작
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
                                    
                                    # [OCR Fallback] 텍스트가 너무 적으면 이미지 인식 시도
                                    if len(text) < 50:
                                        try:
                                            pix = page.get_pixmap()
                                            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                                            ocr_text = transcribe_image_to_text(img, st.session_state.api_key)
                                            if ocr_text: text = ocr_text; log(f"✨ P.{p_idx+1} OCR 성공")
                                        except: pass
                                    
                                    # [Clean Text] 가독성 향상
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

# ------------------------------------------------------------------------------
# TAB 2: 강의 분석 (핵심 기능 - 프리미엄 카드 UI)
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
        
        # PDF 업로더
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
            
            # [Left] PDF 뷰어
            with c_view:
                with st.container(border=True):
                    c1, c2, c3 = st.columns([1, 2, 1])
                    if c1.button("◀"): st.session_state.current_page = max(0, st.session_state.current_page-1); st.rerun()
                    c2.markdown(f"<div style='text-align:center;'><b>Page {st.session_state.current_page+1}</b></div>", unsafe_allow_html=True)
                    if c3.button("▶"): st.session_state.current_page = min(len(doc)-1, st.session_state.current_page+1); st.rerun()
                    
                    page = doc.load_page(st.session_state.current_page)
                    pix = page.get_pixmap(dpi=150)
                    st.image(Image.frombytes("RGB", [pix.width, pix.height], pix.samples), use_container_width=True)
                    
                    # 텍스트 추출 (뷰어용)
                    p_text = page.get_text().strip()
                    if len(p_text) < 50:
                        try:
                            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            ocr_res = transcribe_image_to_text(img, st.session_state.api_key)
                            if ocr_res: p_text = ocr_res
                        except: pass

            # [Right] AI 분석 패널
            with c_ai:
                ai_tab1, ai_tab2 = st.tabs(["📝 족보 매칭", "💬 질의응답"])
                with ai_tab1:
                    if not p_text: st.info("텍스트 없음")
                    else:
                        psig = hash(p_text)
                        # 페이지가 바뀌면 족보 검색
                        if psig != st.session_state.last_page_sig:
                            st.session_state.last_page_sig = psig
                            sub_db = filter_db_by_subject(target_subj, st.session_state.db)
                            st.session_state.last_related = find_relevant_jokbo(p_text, sub_db, top_k=10)
                            st.session_state.last_ai_sig = None # AI 분석 초기화
                        
                        rel = st.session_state.last_related
                        
                        if has_jokbo_evidence(rel):
                            # AI 분석 (JSON 모드) 수행
                            if st.session_state.api_key_ok:
                                aisig = (psig, target_subj)
                                if aisig != st.session_state.last_ai_sig:
                                    with st.spinner("AI 분석 중... (JSON)"):
                                        prmt = build_page_analysis_prompt_json(p_text, rel, target_subj)
                                        json_res = generate_json_response(prmt)
                                        st.session_state.last_ai_data = json_res
                                        st.session_state.last_ai_sig = aisig
                            
                            res_ai = st.session_state.last_ai_data or {}
                            high_rel_count = len([r for r in rel if r['score'] > 0.82])
                            
                            # 카드 렌더링 (상위 2개)
                            for i, r in enumerate(rel[:2]):
                                score = r['score']
                                src = r['content'].get('source', 'Unknown')
                                txt = r['content'].get('text', '')
                                
                                # 텍스트 정리 (가독성 UP)
                                txt_clean = clean_jokbo_text(txt)[:400]
                                meta = parse_metadata_from_filename(src)
                                
                                # 빈도 배지
                                freq_html = ""
                                if i == 0 and high_rel_count >= 2:
                                    freq_html = f"<span class='badge badge-red'>🔥 {high_rel_count}회 출제</span>"
                                elif score > 0.88:
                                    freq_html = "<span class='badge badge-red'>★ 매우 유사</span>"
                                
                                with st.container(border=True):
                                    # 1. 헤더 (배지)
                                    st.markdown(f"<div><span class='badge badge-blue'>기출</span>{freq_html}<span class='badge badge-gray'>{meta}</span></div>", unsafe_allow_html=True)
                                    
                                    # 2. 질문 본문
                                    st.markdown(f"<div class='q-header'>Q. (자동 추출 문항)</div>", unsafe_allow_html=True)
                                    st.markdown(f"<div class='q-body'>{txt_clean}...</div>", unsafe_allow_html=True)
                                    
                                    # 3. 구분선
                                    st.markdown("<div class='dashed-line'></div>", unsafe_allow_html=True)
                                    
                                    # 4. 기능 버튼 (JSON 데이터 연동)
                                    c1, c2, c3 = st.columns(3)
                                    with c1:
                                        with st.expander("📝 정답/해설"):
                                            # 첫 번째 카드에만 AI 해설 표시 (API 절약 및 중복 방지)
                                            if i==0: st.write(res_ai.get("explanation", "생성 중..."))
                                            else: st.caption("가장 유사한 문제에서 확인하세요.")
                                    with c2:
                                        with st.expander("🎯 출제포인트"):
                                            if i==0: st.write(res_ai.get("direction", "생성 중..."))
                                            else: st.caption("내용 없음")
                                    with c3:
                                        with st.expander("🔄 쌍둥이문제"):
                                            if i==0: st.info(res_ai.get("twin_question", "생성 중..."))
                                            else: st.caption("내용 없음")
                                    
                                    # 5. 전체 보기
                                    with st.expander("🔍 전체 지문 보기"):
                                        st.text(clean_jokbo_text(txt))
                        else: st.info("관련 기출 문제가 없습니다.")

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
# TAB 3: 강의 녹음/분석 (완전한 기능 복구)
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
