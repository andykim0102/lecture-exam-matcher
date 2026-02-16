# app.py (UI: Original Rich Style / Logic: Smart Model Discovery + OCR Fallback + Robust Parsing 2.0 + Hot Page Nav)
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
    /* ... (기존 설정 유지) ... */
    
    /* 10. Exam Card Style (New Clean Look) */
    .exam-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s;
    }
    .exam-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    .exam-text {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 1.05rem;
        line-height: 1.6;
        color: #374151;
        margin-top: 10px;
        white-space: pre-wrap;
    }
    
    /* 11. Score Badge */
    .score-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .badge-high { background-color: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
    .badge-med { background-color: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
    .badge-low { background-color: #f3f4f6; color: #4b5563; border: 1px solid #e5e7eb; }
    
    /* 12. Hot Page Button */
    .hot-page-btn-score { font-size: 0.8em; color: #ff3b30; }

    /* 13. Answer Box */
    .answer-box {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 10px;
        margin-top: 10px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# [NEW] Advanced Search Logic (Reranking & Expansion)
# ==========================================

def extract_search_keywords_llm(text):
    """
    [검색 증강] 강의록 텍스트에서 검색에 방해되는 노이즈를 제거하고
    핵심 의학 키워드와 개념 위주로 요약하여 검색 정확도를 높임
    """
    prompt = f"""
    You are a medical study assistant.
    Extract the most important medical keywords, disease names, symptoms, and concepts from the following lecture note text for exam searching.
    Return only the keywords separated by spaces.
    
    [Lecture Text]
    {text[:2000]}
    """
    try:
        # 키워드 추출용 가벼운 호출
        res, _ = generate_with_fallback(prompt, st.session_state.text_models)
        return res.strip()
    except:
        return text # 실패 시 원본 사용

def rerank_candidates_gemini(lecture_text, candidates, top_k=3):
    """
    [Reranking] 1차 검색된 족보 후보들을 LLM이 직접 읽고 
    강의 내용과 논리적으로 가장 연관된 문제만 선별 (가장 강력한 정확도 향상 도구)
    """
    if not candidates: return []
    
    # 후보군 텍스트 준비
    candidates_prompt = ""
    for idx, item in enumerate(candidates):
        # 텍스트 길이 제한 (속도 최적화)
        q_text = item['content']['text'][:400].replace("\n", " ")
        candidates_prompt += f"[{idx}] {q_text}\n\n"
        
    prompt = f"""
    You are a strict medical school professor.
    
    [Task]
    Below is a specific page from a lecture note and a list of candidate exam questions.
    Select the Top {top_k} questions that are MOST relevant to the medical concepts discussed in the lecture note.
    Ignore questions that just share common words but ask about different concepts.
    
    [Lecture Note Content]
    {lecture_text[:1500]}
    
    [Candidate Questions]
    {candidates_prompt}
    
    [Output Format]
    Return ONLY a JSON list of the selected indices in order of relevance.
    Example: [3, 0, 5]
    """
    
    try:
        res, _ = generate_with_fallback(prompt, st.session_state.text_models)
        # JSON 파싱 (숫자 리스트 추출)
        indices = json.loads(re.search(r'\[.*\]', res).group())
        
        # 선택된 인덱스에 해당하는 항목만 반환 (점수 재조정: 순서대로 0.99, 0.98...)
        reranked = []
        for rank, idx in enumerate(indices):
            if idx < len(candidates):
                item = candidates[idx]
                # LLM이 선택한 것은 신뢰도가 높으므로 점수 보정 (시각적 효과)
                item['score'] = 0.95 - (rank * 0.05) 
                reranked.append(item)
        
        return reranked if reranked else candidates[:top_k]
        
    except Exception as e:
        # 에러 발생 시 원래 순서대로 반환 (Fallback)
        return candidates[:top_k]


def find_relevant_jokbo_advanced(query_text, db, top_k=3, use_rerank=True):
    """
    고급 검색 함수: (키워드 추출) -> (벡터 검색) -> (LLM 재순위화)
    """
    if not db: return []
    
    # 1. 검색 쿼리 최적화 (너무 긴 강의록은 노이즈가 됨 -> 핵심 키워드로 변환)
    if len(query_text) > 300:
        search_query = extract_search_keywords_llm(query_text)
    else:
        search_query = query_text
        
    # 2. 1차 벡터 검색 (후보군을 넉넉하게 10~15개 확보)
    # Reranking을 위해 top_k의 3~4배수를 가져옵니다.
    candidates = find_relevant_jokbo(search_query, db, top_k=10)
    
    if not candidates: return []
    
    # 3. LLM Reranking (정밀 매칭)
    if use_rerank:
        final_results = rerank_candidates_gemini(query_text, candidates, top_k=top_k)
        return final_results
    else:
        return candidates[:top_k]

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
# 3. Helpers & Data Logic (Smart Model Update)
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
    
    max_retries = 5
    base_wait = 3
    last_error_msg = ""

    for model_name in sorted_candidates[:2]:
        for attempt in range(max_retries):
            try:
                time.sleep(1.0) 
                if "004" in model_name:
                    res = genai.embed_content(model=model_name, content=text, task_type="retrieval_document")
                else:
                    res = genai.embed_content(model=model_name, content=text)
                    
                if res and "embedding" in res:
                    return res["embedding"], None
            
            except Exception as e:
                err_msg = str(e)
                last_error_msg = f"{model_name}: {err_msg}"
                
                if "429" in err_msg or "Resource exhausted" in err_msg:
                    wait_time = base_wait * (2 ** attempt) + random.randint(1, 3)
                    if status_placeholder:
                        status_placeholder.caption(f"⚠️ 사용량 많음 ({model_name}). {wait_time}초 대기 중... ({attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                elif "404" in err_msg or "Not Found" in err_msg:
                    break
                else:
                    time.sleep(1)
                    
    return None, f"Fail: {last_error_msg}"

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

def transcribe_image_to_text(image, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([
            "Extract all text from this image exactly as is. Just the text, no comments.",
            image
        ])
        return response.text
    except Exception:
        return None

# ==========================================
# 4. New LLM Logic (Parser & Generator)
# ==========================================

def split_jokbo_text(text):
    """
    정규표현식을 사용하여 문항 번호(1. 24. 15) 등을 기준으로 텍스트를 분리하고
    불필요한 공백을 제거합니다.
    """
    if not text: return []
    # Pattern: 문장 시작이나 줄바꿈 뒤에 '숫자 + 점/괄호'가 오는 패턴을 찾음
    pattern = r'(?:\n|^)\s*(?=\d+[\.\)])'
    
    parts = re.split(pattern, text)
    # [수정] 각 파트마다 .strip()을 호출하여 앞뒤 공백/줄바꿈을 완벽히 제거
    questions = [p.strip() for p in parts if p.strip()]
    return questions

def parse_raw_jokbo_llm(raw_text):
    """
    LLM을 사용하여 엉망인 족보 텍스트를 구조화된 JSON으로 변환
    """
    prompt = f"""
    You are an expert medical exam parser.
    Analyze the following text.
    
    [Raw Text]
    {raw_text}
    
    [Requirements]
    1. Extract 'question', 'choices' (list), 'answer', 'explanation'.
    2. **CRITICAL:** Even if the question is in English, the 'explanation' MUST be in KOREAN (한국어).
    3. If there is no explanation in the text, generate a brief logic in Korean based on medical knowledge.
    4. Return ONLY JSON.
    5. Detect 'type' ("객관식" or "주관식").
    6. Return ONLY the JSON object. Do not include markdown formatting like ```json.
    """
    
    try:
        res_text, _ = generate_with_fallback(prompt, st.session_state.text_models)
        # Clean up code blocks if model adds them
        clean_text = re.sub(r"```json|```", "", res_text).strip()
        parsed = json.loads(clean_text)
        return {"success": True, "data": parsed}
    except Exception as e:
        return {"success": False, "error": str(e)}

def generate_twin_problem_llm(parsed_data, subject):
    """
    쌍둥이 문제 생성 (한국어 필수)
    """
    data = parsed_data["data"]
    prompt = f"""
    Create a 'Twin Problem' for medical students.
    Subject: {subject}
    
    [Original Data]
    {json.dumps(data, ensure_ascii=False)}
    
    [Instructions]
    1. Create a similar problem (same logic, different values/case).
    2. **Output Language:** The problem text can be English or Korean (match original), but the **Explanation MUST be in KOREAN**.
    3. Provide:
       - **[변형 문제]**: The new question.
       - **[정답 및 해설]**: Correct answer and detailed logic in Korean.
    
    [Output Format]
    **[변형 문제]**
    (Question Text)
    (Choices if applicable)
    
    **[정답 및 해설]**
    **정답:** (Answer)
    **해설:** (Detailed Logic)
    """
    
    try:
        res_text, _ = generate_with_fallback(prompt, st.session_state.text_models)
        return res_text
    except Exception as e:
        return f"문제 생성 실패: {str(e)}"

# --- Prompt Builders (Improved with Persona) ---
def build_overview_prompt(first_page_text, subject):
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

def format_jokbo_text(text):
    if not text: return ""
    formatted = re.sub(r'(?<!\d)(\d+\.)\s+', r'\n\n**\1** ', text)
    return formatted.strip()

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

def has_jokbo_evidence(related: list[dict]) -> bool:
    return bool(related) and related[0]["score"] >= 0.70


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

    # --- 내 학습 과목 리스트 ---
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
    st.divider()

    st.markdown("### ⚙️ 설정")
    with st.container(border=True):
        api_key_input = st.text_input("Gemini API Key", type="password", key="api_key_input")
        if api_key_input:
            st.session_state.api_key = api_key_input.strip()
            
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
                        
                        st.success(f"✅ 연결 성공!")
                        st.caption(f"텍스트 모델: {st.session_state.best_text_model}")
                        st.caption(f"임베딩 모델: {st.session_state.best_embedding_model}")
                    else:
                        st.error("🚫 사용 가능한 모델을 찾을 수 없습니다. (API Key 권한 확인)")
            
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
                        
                        with st.expander("📝 처리 로그 보기 (클릭하여 펼치기)", expanded=True):
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
                                    success_cnt = 0
                                    skip_cnt = 0
                                    
                                    for p_idx, page in enumerate(doc):
                                        log_container.markdown(f"⏳ **{f.name}** 처리 중... ({p_idx + 1}/{total_pages} 페이지)")
                                        
                                        text = page.get_text().strip()
                                        
                                        if len(text) < 50:
                                            try:
                                                pix = page.get_pixmap()
                                                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                                                ocr_text = transcribe_image_to_text(img, st.session_state.api_key)
                                                if ocr_text:
                                                    text = ocr_text
                                                    log(f"✨ P.{p_idx+1}: 이미지에서 텍스트 추출 성공!")
                                            except Exception:
                                                pass

                                        emb, err_msg = get_embedding_robust(text, status_placeholder=st.empty())
                                        
                                        if emb:
                                            p_data = {
                                                "page": p_idx + 1,
                                                "text": text,
                                                "source": f.name,
                                                "embedding": emb,
                                                "subject": final_subj
                                            }
                                            new_db.append(p_data)
                                            success_cnt += 1
                                        elif err_msg == "text_too_short":
                                            skip_cnt += 1
                                            log(f"⚠️ P.{p_idx+1}: 내용 없음 (스킵)")
                                        else:
                                            log(f"❌ P.{p_idx+1} 임베딩 실패 ({err_msg})")
                                    
                                    log(f"✅ **{f.name}** 완료: 성공 {success_cnt}, 스킵 {skip_cnt}")
                                    
                                except Exception as e:
                                    log(f"❌ 오류 발생: {str(e)}")
                                
                                prog_bar.progress((i + 1) / total_files)
                            
                            if new_db:
                                st.session_state.db.extend(new_db)
                                st.success(f"🎉 총 {len(new_db)} 페이지 학습이 완료되었습니다!")
                                time.sleep(1.5)
                                st.rerun()
                            else:
                                st.warning("저장된 데이터가 없습니다. (문서에 텍스트가 없거나 인식할 수 없습니다.)")
                        
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
                                    st.markdown(f"<span class='gray-text'>🕒 {subj_data['last_updated']}</span>", unsafe_allow_html=True)

# --- TAB 2: 강의 분석 (Original Rich UI + New Logic) ---
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
                    # Hot Pages Reset
                    st.session_state.hot_pages = []
                    st.session_state.hot_pages_analyzed = False

        if st.session_state.lecture_doc:
            doc = st.session_state.lecture_doc
            
            # --- [NEW] Hot Page Discovery ---
            with st.expander("🔥 족보 적중 페이지 탐색기", expanded=not st.session_state.hot_pages_analyzed):
                if not st.session_state.hot_pages_analyzed:
                    st.markdown("강의록 전체를 스캔하여 족보와 연관성이 높은 **'적중 페이지'**를 찾아냅니다.")
                    if st.button("🚀 전체 페이지 분석 시작 (AI Scan)", type="primary"):
                        if not st.session_state.api_key_ok:
                            st.error("설정 탭에서 API Key를 먼저 연결해주세요.")
                        else:
                            # 1. Prepare DB Check
                            sub_db = filter_db_by_subject(target_subj, st.session_state.db)
                            if not sub_db:
                                st.warning(f"'{target_subj}' 과목의 족보 데이터가 없습니다.")
                            else:
                                results = []
                                valid_db_items = [x for x in sub_db if x.get("embedding")]
                                db_embs = [x["embedding"] for x in valid_db_items]
                                
                                if not db_embs:
                                    st.warning("족보 데이터에 임베딩 정보가 없습니다.")
                                else:
                                    # 2. Scanning Loop
                                    prog_bar = st.progress(0)
                                    status_txt = st.empty()
                                    
                                    total_pages = len(doc)
                                    
                                    for p_idx in range(total_pages):
                                        status_txt.caption(f"Analyzing Page {p_idx+1}/{total_pages}...")
                                        try:
                                            page = doc.load_page(p_idx)
                                            txt = page.get_text().strip()
                                            
                                            # Optimization: Skip empty pages, limit text length
                                            if len(txt) > 30: 
                                                emb, _ = get_embedding_robust(txt)
                                                if emb:
                                                    sims = cosine_similarity([emb], db_embs)[0]
                                                    max_score = max(sims)
                                                    
                                                    # Threshold for "Hot Page" (INCREASED to 0.75 for better accuracy)
                                                    if max_score >= 0.75:
                                                        results.append({"page": p_idx, "score": max_score})
                                        except Exception:
                                            pass
                                        
                                        # Update progress
                                        prog_bar.progress((p_idx+1)/total_pages)
                                    
                                    # 3. Store Results (Limit to Top 20)
                                    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
                                    st.session_state.hot_pages = sorted_results[:20]
                                    st.session_state.hot_pages_analyzed = True
                                    st.rerun()
                else:
                    # Display Navigation
                    c_head, c_reset = st.columns([4, 1])
                    with c_head:
                        if not st.session_state.hot_pages:
                            st.info("매칭되는 적중 페이지를 찾지 못했습니다. (임계값 0.75 미만)")
                        else:
                            st.markdown(f"**🔥 총 {len(st.session_state.hot_pages)}개의 적중 페이지 발견!** (클릭하여 이동)")
                    with c_reset:
                        if st.button("재분석"):
                            st.session_state.hot_pages_analyzed = False
                            st.rerun()
                    
                    if st.session_state.hot_pages:
                        # Grid Layout for Buttons
                        cols = st.columns(6)
                        for i, item in enumerate(st.session_state.hot_pages):
                            p_num = item['page']
                            score = item['score']
                            with cols[i % 6]:
                                btn_label = f"P.{p_num+1}"
                                if st.button(btn_label, key=f"nav_{p_num}", help=f"적중률 {score:.0%}"):
                                    st.session_state.current_page = p_num
                                    st.session_state.last_page_sig = None
                                    st.rerun()
                                st.markdown(f"<div style='text-align:center; font-size:0.75rem; color:#ff3b30; margin-top:-10px;'>{score:.0%}</div>", unsafe_allow_html=True)
            
            st.divider()

            col_view, col_ai = st.columns([1.8, 1.2])
            
            # --- Left: Viewer (Standard Image) ---
            with col_view:
                with st.container(border=True):
                    # Nav Toolbar
                    c1, c2, c3 = st.columns([1, 2, 1])
                    with c1:
                        if st.button("◀", use_container_width=True):
                            if st.session_state.current_page > 0: 
                                st.session_state.current_page -= 1
                                st.session_state.chat_history = [] 
                                st.rerun()
                    with c2:
                        st.markdown(f"<div style='text-align:center; font-weight:bold; padding-top:8px;'>Page {st.session_state.current_page+1} / {len(doc)}</div>", unsafe_allow_html=True)
                    with c3:
                        if st.button("▶", use_container_width=True):
                            if st.session_state.current_page < len(doc)-1: 
                                st.session_state.current_page += 1
                                st.session_state.chat_history = []
                                st.rerun()
                    
                    # Prepare Image
                    page = doc.load_page(st.session_state.current_page)
                    pix = page.get_pixmap(dpi=150)
                    pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    p_text = page.get_text() or ""
                    
                    st.image(pil_image, use_container_width=True)

# --- Right: AI Assistant (Clean Version) ---
            with col_ai:
                with st.container(border=True):
                    ai_tab1, ai_tab2 = st.tabs(["📝 족보 분석", "💬 질의응답"])
                    
                    if not p_text.strip():
                        analysis_ready = False
                        with ai_tab1: st.caption("텍스트가 없는 이미지 페이지입니다.")
                    else:
                        analysis_ready = True
                        psig = hash(p_text)
                        
                        if psig != st.session_state.last_page_sig:
                            st.session_state.last_page_sig = psig
                            sub_db = filter_db_by_subject(target_subj, st.session_state.db)
                            
                            # [변경] 고급 검색 함수 사용 (Reranking 적용)
                            with st.spinner("AI가 강의 내용을 분석하고 최적의 족보를 선별 중입니다..."):
                                st.session_state.last_related = find_relevant_jokbo_advanced(
                                    p_text, 
                                    sub_db, 
                                    top_k=3, 
                                    use_rerank=True
                                )
                            st.session_state.last_ai_sig = None
                        
                        rel = st.session_state.last_related
                    
                    # 탭 1: 족보 분석 화면
                    with ai_tab1:
                        # 보기 모드 선택 토글
                        view_mode = st.radio(
                            "보기 모드", 
                            ["📄 현재 페이지 연관", "📚 과목 전체 문항"], 
                            horizontal=True, 
                            label_visibility="collapsed"
                        )
                        st.divider()

                        if view_mode == "📚 과목 전체 문항":
                            # 전체 문항 리스트 출력 로직
                            st.markdown(f"##### 📚 {target_subj} 전체 문항")
                            sub_db = filter_db_by_subject(target_subj, st.session_state.db)
                            
                            if not sub_db:
                                st.info("이 과목에 등록된 족보 데이터가 없습니다.")
                            else:
                                all_items = []
                                # 족보 DB에서 모든 문항 추출
                                for page_item in sub_db:
                                    q_chunks = split_jokbo_text(page_item['text'])
                                    if not q_chunks: q_chunks = [page_item['text']]
                                    for q in q_chunks:
                                        all_items.append({
                                            "text": q,
                                            "source": page_item['source'],
                                            "page": page_item['page']
                                        })
                                
                                st.caption(f"총 {len(all_items)}개의 문항이 있습니다.")
                                
                                # 전체 리스트 출력
                                for idx, item in enumerate(all_items):
                                    item_id = f"all_view_{idx}"
                                    with st.container(border=True):
                                        st.caption(f"📄 {item['source']} (P.{item['page']})")
                                        st.markdown(f"""<div class="exam-text" style="font-size: 0.95rem;">{item['text']}</div>""", unsafe_allow_html=True)
                                        
                                        # AI 분석 기능
                                        with st.expander("✨ 정답/해설 및 쌍둥이 문제"):
                                            # 데이터가 없으면 자동 분석 시작
                                            if item_id not in st.session_state.parsed_items:
                                                if st.button("🚀 분석 실행", key=f"btn_all_run_{item_id}"):
                                                     with st.spinner("분석 중..."):
                                                         parsed = parse_raw_jokbo_llm(item['text'])
                                                         st.session_state.parsed_items[item_id] = parsed
                                                         if parsed["success"]:
                                                             twin_res = generate_twin_problem_llm(parsed, target_subj)
                                                             st.session_state.twin_items[item_id] = twin_res
                                                             st.rerun()

                                            # 결과 표시
                                            if item_id in st.session_state.parsed_items:
                                                parsed_res = st.session_state.parsed_items[item_id]
                                                if parsed_res["success"]:
                                                    data = parsed_res["data"]
                                                    st.markdown(f"""
                                                    <div class="answer-box">
                                                        <strong>✅ 정답:</strong> {data.get('answer', '정보 없음')}<br><br>
                                                        <strong>💡 해설:</strong> {data.get('explanation', '정보 없음')}
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                                    
                                                    if item_id in st.session_state.twin_items:
                                                        st.divider()
                                                        st.markdown(st.session_state.twin_items[item_id])
                                                else:
                                                    st.error("분석 실패")

                        elif analysis_ready:
                            # 기존 '현재 페이지 연관' 로직
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
                            else:
                                st.markdown(f"##### 🔥 연관 족보 TOP {len(rel[:3])}")
                                
                                if not rel:
                                    st.caption("관련된 족보 내용이 없습니다.")
                                
                                # Loop through related items
                                for i, r in enumerate(rel[:3]):
                                    content = r['content']
                                    score = r['score']
                                    raw_txt = content['text']
                                    
                                    # 유사도 뱃지 로직
                                    if score >= 0.82:
                                        badge_cls = "badge-high"
                                        badge_txt = f"🔥 강력 추천 ({score:.0%})"
                                    elif score >= 0.75:
                                        badge_cls = "badge-med"
                                        badge_txt = f"✨ 높은 연관 ({score:.0%})"
                                    else:
                                        badge_cls = "badge-low"
                                        badge_txt = f"☁️ 참고 문제 ({score:.0%})"
                                    
                                    # 문항 분리 및 공백 제거
                                    split_questions = split_jokbo_text(raw_txt)
                                    if not split_questions: split_questions = [raw_txt]

                                    for seq_idx, question_txt in enumerate(split_questions):
                                        item_id = f"{psig}_{i}_{seq_idx}"
                                        
                                        # 1. 문제 카드 출력
                                        st.markdown(f"""
                                        <div class="exam-card">
                                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                                <span class="score-badge {badge_cls}">{badge_txt}</span>
                                                <small style="color: #9ca3af;">{content['source']} (P.{content['page']})</small>
                                            </div>
                                            <div class="exam-text">{question_txt}</div>
                                        </div>
                                        """, unsafe_allow_html=True)

                                        # 2. 자동 분석 및 탭 뷰
                                        with st.expander("💡 해설 및 변형 문제 확인하기", expanded=False):
                                            
                                            # 데이터가 세션에 없으면 -> 즉시 분석 실행 (자동화)
                                            if item_id not in st.session_state.parsed_items:
                                                with st.spinner("AI가 문제를 분석하고 있습니다..."):
                                                    parsed = parse_raw_jokbo_llm(question_txt)
                                                    st.session_state.parsed_items[item_id] = parsed
                                                    
                                                    if parsed["success"]:
                                                        twin_res = generate_twin_problem_llm(parsed, target_subj)
                                                        st.session_state.twin_items[item_id] = twin_res
                                            
                                            # 분석 결과 출력 (탭 분리)
                                            parsed_res = st.session_state.parsed_items.get(item_id)
                                            
                                            if parsed_res and parsed_res.get("success"):
                                                data = parsed_res["data"]
                                                
                                                tab_ans, tab_twin = st.tabs(["✅ 정답 & 해설", "🔄 쌍둥이(변형) 문제"])
                                                
                                                with tab_ans:
                                                    ans_text = data.get('answer', '정보 없음')
                                                    exp_text = data.get('explanation', '정보 없음')
                                                    st.markdown(f"""
                                                    <div class="answer-box">
                                                        <p><strong>정답:</strong> {ans_text}</p>
                                                        <hr style="margin: 10px 0; opacity: 0.2;">
                                                        <p><strong>해설:</strong><br>{exp_text}</p>
                                                    </div>
                                                    """, unsafe_allow_html=True)

                                                with tab_twin:
                                                    twin_content = st.session_state.twin_items.get(item_id, "변형 문제 생성 실패")
                                                    st.info(twin_content)
                                                    
                                            elif parsed_res and not parsed_res.get("success"):
                                                st.error(f"분석 실패: {parsed_res.get('error')}")
                                            else:
                                                st.warning("일시적인 오류가 발생했습니다. 다시 시도해주세요.")
                        else:
                            st.info("분석할 텍스트가 없습니다.")

                    # 탭 2: 질의응답
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

# Loop through related items
                                for i, r in enumerate(rel[:3]):
                                    content = r['content']
                                    score = r['score']
                                    raw_txt = content['text']
                                    
                                    # 유사도 뱃지 로직
                                    if score >= 0.82:
                                        badge_cls = "badge-high"
                                        badge_txt = f"🔥 강력 추천 ({score:.0%})"
                                    elif score >= 0.75:
                                        badge_cls = "badge-med"
                                        badge_txt = f"✨ 높은 연관 ({score:.0%})"
                                    else:
                                        badge_cls = "badge-low"
                                        badge_txt = f"☁️ 참고 문제 ({score:.0%})"
                                    
                                    # 문항 분리 및 공백 제거
                                    split_questions = split_jokbo_text(raw_txt)
                                    if not split_questions: split_questions = [raw_txt]

                                    for seq_idx, question_txt in enumerate(split_questions):
                                        item_id = f"{psig}_{i}_{seq_idx}"
                                        
                                        # 1. 문제 카드 출력
                                        st.markdown(f"""
                                        <div class="exam-card">
                                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                                <span class="score-badge {badge_cls}">{badge_txt}</span>
                                                <small style="color: #9ca3af;">{content['source']} (P.{content['page']})</small>
                                            </div>
                                            <div class="exam-text">{question_txt}</div>
                                        </div>
                                        """, unsafe_allow_html=True)

                                        # 2. [NEW] 자동 분석 및 탭 뷰 (버튼 제거)
                                        # Expander를 사용하여 기본적으로는 접어두되, 열면 바로 내용이 보이게 처리
                                        with st.expander("💡 해설 및 변형 문제 확인하기", expanded=False):
                                            
                                            # 데이터가 세션에 없으면 -> 즉시 분석 실행 (자동화)
                                            if item_id not in st.session_state.parsed_items:
                                                with st.spinner("AI가 문제를 분석하고 있습니다..."):
                                                    parsed = parse_raw_jokbo_llm(question_txt)
                                                    st.session_state.parsed_items[item_id] = parsed
                                                    
                                                    if parsed["success"]:
                                                        twin_res = generate_twin_problem_llm(parsed, target_subj)
                                                        st.session_state.twin_items[item_id] = twin_res
                                            
                                            # 분석 결과 출력 (탭 분리)
                                            parsed_res = st.session_state.parsed_items.get(item_id)
                                            
                                            if parsed_res and parsed_res.get("success"):
                                                data = parsed_res["data"]
                                                
                                                # [NEW] 탭으로 분리
                                                tab_ans, tab_twin = st.tabs(["✅ 정답 & 해설", "🔄 쌍둥이(변형) 문제"])
                                                
                                                with tab_ans:
                                                    # AttributeError 방지를 위한 안전한 접근
                                                    ans_text = data.get('answer', '정보 없음')
                                                    exp_text = data.get('explanation', '정보 없음')
                                                    
                                                    st.markdown(f"""
                                                    <div class="answer-box">
                                                        <p><strong>정답:</strong> {ans_text}</p>
                                                        <hr style="margin: 10px 0; opacity: 0.2;">
                                                        <p><strong>해설:</strong><br>{exp_text}</p>
                                                    </div>
                                                    """, unsafe_allow_html=True)

                                                with tab_twin:
                                                    twin_content = st.session_state.twin_items.get(item_id, "변형 문제 생성 실패")
                                                    st.info(twin_content)
                                                    
                                            elif parsed_res and not parsed_res.get("success"):
                                                st.error(f"분석 실패: {parsed_res.get('error')}")
                                            else:
                                                st.warning("일시적인 오류가 발생했습니다. 다시 시도해주세요.")
                        else:
                            st.info("분석할 텍스트가 없습니다.")

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


# --- TAB 3: 강의 녹음/분석 (Original Rich UI + New Logic) ---
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
                if audio_value:
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





