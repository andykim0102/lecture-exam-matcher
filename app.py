# app.py (Refactored for User Experience & Performance)
import time
import re
import json
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import google.generativeai as genai

# ==========================================
# 0. Page Config & Design System
# ==========================================
st.set_page_config(
    page_title="Med-Study OS",
    layout="wide",
    page_icon="🩺",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, distraction-free studying
st.markdown("""
<style>
    /* Global Clean Look */
    .stApp { background-color: #f8f9fa; }
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; letter-spacing: -0.5px; }
    
    /* PDF Container styling */
    .pdf-container {
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }

    /* Question Card Styling */
    .q-card {
        background-color: white;
        border: 1px solid #edf2f7;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        transition: all 0.2s ease;
        border-left: 4px solid #007aff;
    }
    .q-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.05);
    }
    .q-meta {
        font-size: 0.8rem;
        color: #8e8e93;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .q-text {
        font-size: 1rem;
        font-weight: 500;
        line-height: 1.6;
        color: #1c1c1e;
        margin-bottom: 16px;
    }
    .q-badge {
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.75rem;
    }

    /* Answer/Explanation Box */
    .ans-box {
        background-color: #f1f8e9;
        border-radius: 8px;
        padding: 16px;
        margin-top: 12px;
        animation: fadeIn 0.3s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-5px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Button Styling Override */
    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
    div.stButton > button[kind="secondary"] {
        border: 1px solid #d1d1d6;
        background-color: white;
        color: #333;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
        border-bottom: 1px solid #e0e0e0;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        color: #888;
    }
    .stTabs [aria-selected="true"] {
        color: #007aff !important;
        border-bottom: 2px solid #007aff;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 1. State Management
# ==========================================
DEFAULT_STATE = {
    "logged_in": False,
    "db": [], 
    "bookmarks": [],
    "api_key": "",
    "lecture_doc": None,
    "lecture_filename": None,
    "current_page": 0,
    "selected_subject": None,
    "last_page_sig": None,
    "current_related_qs": [],
    "analyzed_data": {},  # { question_hash: {parsed: ..., twin: ...} }
    "chat_history": []
}

for k, v in DEFAULT_STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==========================================
# 2. Core Logic Functions
# ==========================================

def get_embedding(text):
    """Robust embedding retrieval with error handling."""
    if not st.session_state.api_key: return None
    try:
        genai.configure(api_key=st.session_state.api_key)
        # Try primary model first, fallback if needed
        model = "models/text-embedding-004"
        result = genai.embed_content(model=model, content=text[:9000])
        return result.get("embedding")
    except Exception:
        try:
            # Fallback to older model
            result = genai.embed_content(model="models/embedding-001", content=text[:9000])
            return result.get("embedding")
        except:
            return None

def find_relevant_questions(query_text, subject, threshold=0.65):
    """Finds questions from DB relevant to the current page text."""
    if not st.session_state.db or not query_text: return []
    
    # Filter DB by subject
    subject_db = [item for item in st.session_state.db if item.get("subject") == subject]
    if not subject_db: return []

    # Get query embedding
    q_emb = get_embedding(query_text)
    if not q_emb: return []

    # Calculate Similarities
    db_embs = [item["embedding"] for item in subject_db]
    sims = cosine_similarity([q_emb], db_embs)[0]

    # Filter & Sort
    results = []
    for idx, score in enumerate(sims):
        if score >= threshold:
            results.append({
                "score": score,
                "content": subject_db[idx]
            })
    
    # Sort by relevance
    return sorted(results, key=lambda x: x["score"], reverse=True)

def generate_ai_analysis(question_text):
    """Generates structure (JSON) and Twin Problem using LLM."""
    if not st.session_state.api_key: return None
    genai.configure(api_key=st.session_state.api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    prompt = f"""
    You are a medical tutor. Analyze this exam question.
    
    [Question Text]:
    {question_text}

    1. Extract the correct answer and a detailed explanation.
    2. Create a "Twin Problem" (similar concept, different scenario).
    
    Output ONLY valid JSON format:
    {{
        "answer": "String (e.g., 3)",
        "explanation": "String (Detailed logic)",
        "twin_problem": "String (Full question text)",
        "twin_answer": "String",
        "twin_explanation": "String"
    }}
    """
    try:
        res = model.generate_content(prompt)
        text = res.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except:
        return None

# ==========================================
# 3. UI Components
# ==========================================

def sidebar_ui():
    with st.sidebar:
        st.markdown("### 🩺 Med-Study OS")
        
        # Profile / Auth
        if st.session_state.logged_in:
            st.success("로그인됨: Admin")
            if st.button("로그아웃", use_container_width=True):
                st.session_state.logged_in = False
                st.rerun()
        
        st.divider()

        # Subject List
        st.markdown("**📚 내 과목 (My Subjects)**")
        subjects = sorted({x.get("subject", "기타") for x in st.session_state.db})
        if subjects:
            for s in subjects:
                if st.button(f"📘 {s}", key=f"nav_{s}", use_container_width=True):
                    st.session_state.selected_subject = s
                    st.rerun()
        else:
            st.info("등록된 과목이 없습니다.\n'데이터 관리' 탭에서 추가하세요.")

        st.divider()
        
        # Settings
        with st.expander("⚙️ 설정 (API Key)"):
            key_input = st.text_input("Gemini API Key", value=st.session_state.api_key, type="password")
            if key_input: st.session_state.api_key = key_input
            st.caption("Google AI Studio에서 키를 발급받으세요.")

def login_screen():
    c1, c2, c3 = st.columns([1,1,1])
    with c2:
        st.markdown("<div style='height:100px;'></div>", unsafe_allow_html=True)
        st.title("Med-Study OS")
        st.markdown("스마트한 의대생을 위한 학습 파트너")
        
        with st.form("login_form"):
            uid = st.text_input("ID")
            pwd = st.text_input("PW", type="password")
            submitted = st.form_submit_button("Start Learning", type="primary", use_container_width=True)
            
            if submitted:
                if pwd == "1234":  # Simple Demo Auth
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("비밀번호를 확인하세요. (Demo: 1234)")

def main_study_ui():
    # Top Navigation for Subject
    if not st.session_state.selected_subject:
        st.info("👈 사이드바에서 학습할 과목을 선택해주세요.")
        return

    st.markdown(f"## 📖 {st.session_state.selected_subject} 학습 모드")
    
    col_pdf, col_quiz = st.columns([1.1, 1])
    
    # --- LEFT: PDF Viewer ---
    with col_pdf:
        uploaded_pdf = st.file_uploader("강의록 PDF 열기", type="pdf", label_visibility="collapsed")
        
        if uploaded_pdf:
            # Load PDF Logic
            if st.session_state.lecture_filename != uploaded_pdf.name:
                st.session_state.lecture_doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
                st.session_state.lecture_filename = uploaded_pdf.name
                st.session_state.current_page = 0
            
            doc = st.session_state.lecture_doc
            
            # PDF Navigation
            c_prev, c_page, c_next = st.columns([1, 2, 1])
            with c_prev:
                if st.button("◀ 이전", use_container_width=True, disabled=(st.session_state.current_page <= 0)):
                    st.session_state.current_page -= 1
                    st.rerun()
            with c_page:
                st.markdown(f"<div style='text-align:center; font-weight:bold;'>Page {st.session_state.current_page + 1} / {len(doc)}</div>", unsafe_allow_html=True)
            with c_next:
                if st.button("다음 ▶", use_container_width=True, disabled=(st.session_state.current_page >= len(doc)-1)):
                    st.session_state.current_page += 1
                    st.rerun()

            # Render Page
            page = doc.load_page(st.session_state.current_page)
            pix = page.get_pixmap(dpi=150) # Standard DPI for speed
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            st.image(img, use_container_width=True, output_format="JPEG")
            
            # Extract Text for Matching
            current_text = page.get_text().strip()
        else:
            st.markdown("""
            <div style="padding:40px; text-align:center; border:2px dashed #ccc; border-radius:12px; color:#888;">
                <h3>📂 강의록 PDF를 업로드하세요</h3>
                <p>AI가 현재 페이지와 연관된 족보 문제를 찾아줍니다.</p>
            </div>
            """, unsafe_allow_html=True)
            current_text = ""

    # --- RIGHT: Context-Aware Problems ---
    with col_quiz:
        st.markdown("### 🎯 관련 기출 문제 (Check Point)")
        
        if not current_text:
            st.info("강의록을 열면 문제가 표시됩니다.")
        else:
            # 1. Update Matching (Only if page changed)
            page_sig = hash(current_text)
            if st.session_state.last_page_sig != page_sig:
                with st.spinner("🔍 관련 문제 분석 중..."):
                    st.session_state.current_related_qs = find_relevant_questions(
                        current_text, 
                        st.session_state.selected_subject
                    )
                    st.session_state.last_page_sig = page_sig

            # 2. Display Results
            questions = st.session_state.current_related_qs
            
            if not questions:
                st.markdown("""
                <div style="text-align:center; padding:30px; color:#888;">
                    <p>이 페이지와 직접적으로 연관된 문제가 발견되지 않았습니다.</p>
                    <small>전체 문제를 보려면 '데이터 관리' 탭을 확인하세요.</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.caption(f"총 {len(questions)}개의 관련 문제를 찾았습니다.")
                
                for idx, item in enumerate(questions):
                    q_content = item["content"]["text"]
                    score = item["score"]
                    q_id = f"q_{page_sig}_{idx}"
                    
                    # --- QUESTION CARD UI ---
                    st.markdown(f"""
                    <div class="q-card">
                        <div class="q-meta">
                            <span class="q-badge">유사도 {int(score*100)}%</span>
                            <span>{item['content']['source']} (P.{item['content']['page']})</span>
                        </div>
                        <div class="q-text">{q_content[:300]}...</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Interaction Buttons (Accordion style logic)
                    c_act1, c_act2, c_act3 = st.columns([1, 1, 2])
                    
                    # A. Bookmark Toggle
                    is_bookmarked = q_content in st.session_state.bookmarks
                    if c_act1.button("★ 저장" if not is_bookmarked else "★ 저장됨", key=f"bk_{q_id}"):
                        if is_bookmarked:
                            st.session_state.bookmarks.remove(q_content)
                        else:
                            st.session_state.bookmarks.append(q_content)
                        st.rerun()

                    # B. AI Analysis / View Answer
                    # Use session state to toggle visibility of answer to prevent reload reset
                    show_ans_key = f"show_ans_{q_id}"
                    if show_ans_key not in st.session_state: st.session_state[show_ans_key] = False

                    if c_act2.button("정답 확인", key=f"btn_ans_{q_id}"):
                        st.session_state[show_ans_key] = not st.session_state[show_ans_key]
                        # Trigger AI analysis if first time
                        if q_id not in st.session_state.analyzed_data:
                            with st.spinner("AI 튜터가 분석 중..."):
                                analysis = generate_ai_analysis(q_content)
                                if analysis:
                                    st.session_state.analyzed_data[q_id] = analysis
                                else:
                                    st.error("분석 실패")
                        st.rerun()

                    # Display Answer Section
                    if st.session_state[show_ans_key]:
                        data = st.session_state.analyzed_data.get(q_id)
                        if data:
                            st.markdown(f"""
                            <div class="ans-box">
                                <strong>✅ 정답: {data.get('answer')}</strong><br><br>
                                {data.get('explanation')}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            with st.expander("🧩 쌍둥이 문제 풀어보기 (변형 문제)"):
                                st.markdown(f"**Q. {data.get('twin_problem')}**")
                                if st.button("쌍둥이 문제 정답 보기", key=f"twin_btn_{q_id}"):
                                    st.info(f"정답: {data.get('twin_answer')}\n\n해설: {data.get('twin_explanation')}")
                        else:
                            st.warning("상세 분석 데이터를 불러오지 못했습니다. 원본 텍스트를 참고하세요.")
                            st.text_area("원본 텍스트", q_content, height=100)

                    st.markdown("---")


def management_ui():
    st.markdown("## 📂 데이터 및 족보 관리")
    
    t1, t2 = st.tabs(["족보 업로드", "북마크(오답노트)"])
    
    with t1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 1. 새로운 족보 추가")
            new_subj = st.text_input("과목명 (예: 해부학)", placeholder="과목명을 입력하세요")
            files = st.file_uploader("PDF 파일 선택", accept_multiple_files=True, type="pdf")
            
            if st.button("업로드 및 학습 시작", type="primary"):
                if not st.session_state.api_key:
                    st.error("설정에서 API Key를 먼저 입력하세요.")
                elif not files or not new_subj:
                    st.warning("과목명과 파일을 모두 입력해주세요.")
                else:
                    progress_bar = st.progress(0)
                    total_files = len(files)
                    
                    for idx, f in enumerate(files):
                        doc = fitz.open(stream=f.read(), filetype="pdf")
                        for p_num, page in enumerate(doc):
                            text = page.get_text().strip()
                            if len(text) > 50: # Ignore empty pages
                                emb = get_embedding(text)
                                if emb:
                                    st.session_state.db.append({
                                        "subject": new_subj,
                                        "source": f.name,
                                        "page": p_num + 1,
                                        "text": text,
                                        "embedding": emb
                                    })
                        progress_bar.progress((idx + 1) / total_files)
                    
                    st.success(f"학습 완료! 총 {len(st.session_state.db)} 페이지 저장됨.")
        
        with col2:
            st.markdown("#### 2. 데이터베이스 현황")
            st.metric("총 학습된 페이지 수", len(st.session_state.db))
            
            if st.session_state.db:
                df_data = []
                for item in st.session_state.db:
                    df_data.append({"과목": item['subject'], "출처": item['source']})
                st.dataframe(df_data, use_container_width=True, height=300)
                
                if st.button("DB 전체 초기화 (주의)", type="secondary"):
                    st.session_state.db = []
                    st.rerun()

    with t2:
        st.markdown("#### ⭐ 내가 저장한 문제들")
        if not st.session_state.bookmarks:
            st.info("아직 저장된 문제가 없습니다. 학습 중 '★ 저장' 버튼을 눌러보세요.")
        else:
            for i, bm in enumerate(st.session_state.bookmarks):
                with st.expander(f"북마크 #{i+1}"):
                    st.write(bm)
                    if st.button("삭제", key=f"del_bm_{i}"):
                        st.session_state.bookmarks.pop(i)
                        st.rerun()

# ==========================================
# 4. Main Execution
# ==========================================
def main():
    if not st.session_state.logged_in:
        login_screen()
    else:
        sidebar_ui()
        
        # Simple Tab Layout for Main Features
        menu = st.tabs(["📝 학습하기 (Study)", "⚙️ 데이터 관리 (Manage)"])
        
        with menu[0]:
            main_study_ui()
        with menu[1]:
            management_ui()

if __name__ == "__main__":
    main()
