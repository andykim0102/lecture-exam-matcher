# app.py (Full Features Restored + UX Enhanced)
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
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; letter-spacing: -0.5px; color: #1c1c1e; }
    
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

    /* Tabs & Buttons */
    div.stButton > button { border-radius: 8px; font-weight: 600; }
    .stTabs [data-baseweb="tab-list"] { background-color: transparent; border-bottom: 1px solid #e0e0e0; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; color: #888; }
    .stTabs [aria-selected="true"] { color: #007aff !important; border-bottom: 2px solid #007aff; }
    
    /* Chat Styling */
    .stChatMessage { background-color: white; border-radius: 12px; padding: 10px; border: 1px solid #eee; }
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
    "chat_history": [],
    "hot_pages": [],      # For Hot Page Analysis
    "hot_pages_analyzed": False,
    "transcribed_text": "" # For Audio Tab
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

def find_relevant_questions(query_text, subject, threshold=0.65, top_k=5):
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
    top_idxs = np.argsort(sims)[::-1][:top_k]
    
    for idx in top_idxs:
        score = sims[idx]
        if score >= threshold:
            results.append({
                "score": float(score),
                "content": subject_db[idx]
            })
    
    return results

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

def analyze_hot_pages(doc, subject):
    """Background analysis to find important pages in PDF."""
    hot_list = []
    subject_db = [item for item in st.session_state.db if item.get("subject") == subject]
    if not subject_db: return []
    
    db_embs = [item["embedding"] for item in subject_db]
    
    # Sample every page (or every 2nd page for speed if needed)
    for p_idx in range(len(doc)):
        try:
            page = doc.load_page(p_idx)
            txt = page.get_text().strip()
            if len(txt) > 50:
                emb = get_embedding(txt)
                if emb:
                    sims = cosine_similarity([emb], db_embs)[0]
                    max_score = max(sims)
                    if max_score >= 0.70: # Threshold for 'Hot'
                        hot_list.append({"page": p_idx, "score": max_score})
        except: pass
        
    return sorted(hot_list, key=lambda x: x["score"], reverse=True)[:15]

def transcribe_audio_gemini(audio_bytes):
    if not st.session_state.api_key: return None
    try:
        genai.configure(api_key=st.session_state.api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([
            "Transcribe this audio file into text accurately.",
            {"mime_type": "audio/wav", "data": audio_bytes}
        ])
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# ==========================================
# 3. UI Components
# ==========================================

def sidebar_ui():
    with st.sidebar:
        st.markdown("### 🩺 Med-Study OS")
        
        # Profile
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
            st.info("등록된 과목이 없습니다.")

        st.divider()
        # Settings
        with st.expander("⚙️ 설정 (API Key)"):
            key_input = st.text_input("Gemini API Key", value=st.session_state.api_key, type="password")
            if key_input: st.session_state.api_key = key_input

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
                if pwd == "1234":
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("비밀번호: 1234")

def main_study_ui():
    if not st.session_state.selected_subject:
        st.info("👈 사이드바에서 학습할 과목을 선택해주세요.")
        return

    st.markdown(f"## 📖 {st.session_state.selected_subject} 학습 모드")
    
    col_pdf, col_right = st.columns([1.1, 1])
    
    # --- LEFT: PDF Viewer ---
    with col_pdf:
        uploaded_pdf = st.file_uploader("강의록 PDF 열기", type="pdf", label_visibility="collapsed")
        
        if uploaded_pdf:
            # Load PDF & Analyze Hot Pages
            if st.session_state.lecture_filename != uploaded_pdf.name:
                st.session_state.lecture_doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
                st.session_state.lecture_filename = uploaded_pdf.name
                st.session_state.current_page = 0
                st.session_state.hot_pages_analyzed = False
                st.session_state.chat_history = []
            
            doc = st.session_state.lecture_doc
            
            # Hot Page Analysis Trigger
            if not st.session_state.hot_pages_analyzed:
                with st.spinner("🚀 강의록 전체 분석 중 (적중 예상 페이지 찾는 중)..."):
                    st.session_state.hot_pages = analyze_hot_pages(doc, st.session_state.selected_subject)
                    st.session_state.hot_pages_analyzed = True
                st.rerun()

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
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            st.image(img, use_container_width=True, output_format="JPEG")
            current_text = page.get_text().strip()
        else:
            st.markdown("""
            <div style="padding:40px; text-align:center; border:2px dashed #ccc; border-radius:12px; color:#888;">
                <h3>📂 강의록 PDF를 업로드하세요</h3>
            </div>
            """, unsafe_allow_html=True)
            current_text = ""

    # --- RIGHT: Multi-Function Panel ---
    with col_right:
        # Integrated Tabs for all functionalities
        r_tab1, r_tab2, r_tab3, r_tab4 = st.tabs(["🎯 관련 문제", "💬 AI 튜터", "🔥 핵심 페이지", "📂 전체 문제"])
        
        # Tab 1: Related Questions (Main Feature)
        with r_tab1:
            if not current_text:
                st.info("강의록을 열면 관련 문제가 표시됩니다.")
            else:
                # Matching Logic
                page_sig = hash(current_text)
                if st.session_state.last_page_sig != page_sig:
                    with st.spinner("🔍 분석 중..."):
                        st.session_state.current_related_qs = find_relevant_questions(
                            current_text, 
                            st.session_state.selected_subject
                        )
                        st.session_state.last_page_sig = page_sig

                questions = st.session_state.current_related_qs
                
                if not questions:
                    st.markdown("<div style='text-align:center; padding:30px; color:#888;'>관련 문제가 없습니다.</div>", unsafe_allow_html=True)
                else:
                    for idx, item in enumerate(questions):
                        q_content = item["content"]["text"]
                        score = item["score"]
                        q_id = f"q_{page_sig}_{idx}"
                        
                        st.markdown(f"""
                        <div class="q-card">
                            <div class="q-meta">
                                <span class="q-badge">유사도 {int(score*100)}%</span>
                                <span>{item['content']['source']} (P.{item['content']['page']})</span>
                            </div>
                            <div class="q-text">{q_content[:200]}...</div>
                        </div>
                        """, unsafe_allow_html=True)

                        c_act1, c_act2 = st.columns([1, 2])
                        show_ans_key = f"show_ans_{q_id}"
                        if show_ans_key not in st.session_state: st.session_state[show_ans_key] = False

                        if c_act1.button("★ 저장", key=f"bk_{q_id}"):
                            st.session_state.bookmarks.append(q_content)
                            st.toast("저장되었습니다!")

                        if c_act2.button("정답 및 해설 확인", key=f"btn_ans_{q_id}", type="primary"):
                            st.session_state[show_ans_key] = not st.session_state[show_ans_key]
                            if q_id not in st.session_state.analyzed_data:
                                with st.spinner("AI 분석 중..."):
                                    analysis = generate_ai_analysis(q_content)
                                    if analysis: st.session_state.analyzed_data[q_id] = analysis

                        if st.session_state[show_ans_key]:
                            data = st.session_state.analyzed_data.get(q_id)
                            if data:
                                st.markdown(f"""
                                <div class="ans-box">
                                    <strong>✅ 정답: {data.get('answer')}</strong><br><br>
                                    {data.get('explanation')}
                                </div>
                                """, unsafe_allow_html=True)
                                with st.expander("🧩 변형 문제 (Twin Problem)"):
                                    st.write(data.get('twin_problem'))
                                    if st.button("변형 문제 답 보기", key=f"twin_{q_id}"):
                                        st.write(f"정답: {data.get('twin_answer')}")

        # Tab 2: AI Tutor (Chat)
        with r_tab2:
            st.caption("현재 강의 내용에 대해 질문하세요.")
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]): st.write(msg["content"])
            
            if prompt := st.chat_input("질문 입력..."):
                if not st.session_state.api_key:
                    st.error("API Key 필요")
                else:
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    with st.chat_message("user"): st.write(prompt)
                    
                    with st.chat_message("assistant"):
                        with st.spinner("생각 중..."):
                            # Context building
                            context_str = f"Current Lecture Page:\n{current_text[:1000]}\n"
                            if st.session_state.current_related_qs:
                                context_str += "\nRelated Exams:\n" + "\n".join([x["content"]["text"][:300] for x in st.session_state.current_related_qs[:2]])
                            
                            full_prompt = f"Context:\n{context_str}\n\nUser Question: {prompt}\nAnswer as a medical tutor."
                            
                            genai.configure(api_key=st.session_state.api_key)
                            model = genai.GenerativeModel("gemini-1.5-flash")
                            res = model.generate_content(full_prompt)
                            st.write(res.text)
                            st.session_state.chat_history.append({"role": "assistant", "content": res.text})

        # Tab 3: Hot Pages
        with r_tab3:
            st.markdown("##### 🔥 출제 예상 페이지")
            if not st.session_state.hot_pages:
                st.caption("분석된 중요 페이지가 없거나 분석 중입니다.")
            else:
                st.caption(f"{len(st.session_state.hot_pages)}개의 중요 페이지 발견")
                cols = st.columns(3)
                for i, item in enumerate(st.session_state.hot_pages):
                    with cols[i % 3]:
                        if st.button(f"P.{item['page']+1}", key=f"hot_{i}", help=f"적중률 {int(item['score']*100)}%"):
                            st.session_state.current_page = item['page']
                            st.rerun()

        # Tab 4: View All
        with r_tab4:
            st.markdown("##### 📂 전체 문제 모아보기")
            subject_db = [item for item in st.session_state.db if item.get("subject") == st.session_state.selected_subject]
            if not subject_db:
                st.caption("데이터 없음")
            else:
                for i, item in enumerate(subject_db):
                    with st.expander(f"P.{item['page']} - {item['text'][:30]}..."):
                        st.write(item['text'])

def management_ui():
    st.markdown("## 📂 데이터 및 족보 관리")
    t1, t2 = st.tabs(["족보 업로드", "북마크(오답노트)"])
    with t1:
        new_subj = st.text_input("과목명", placeholder="해부학")
        files = st.file_uploader("PDF 파일 선택", accept_multiple_files=True, type="pdf")
        if st.button("학습 시작", type="primary"):
            if files and new_subj and st.session_state.api_key:
                bar = st.progress(0)
                for idx, f in enumerate(files):
                    doc = fitz.open(stream=f.read(), filetype="pdf")
                    for p_num, page in enumerate(doc):
                        txt = page.get_text().strip()
                        if len(txt) > 50:
                            emb = get_embedding(txt)
                            if emb:
                                st.session_state.db.append({
                                    "subject": new_subj, "source": f.name, "page": p_num+1, "text": txt, "embedding": emb
                                })
                    bar.progress((idx+1)/len(files))
                st.success("완료!")
    with t2:
        if st.session_state.bookmarks:
            for b in st.session_state.bookmarks:
                st.info(b)
                st.button("삭제", key=f"del_{hash(b)}")
        else:
            st.caption("저장된 문제 없음")

def record_ui():
    st.markdown("## 🎙️ 강의 녹음 및 분석")
    col1, col2 = st.columns(2)
    with col1:
        audio_val = st.audio_input("녹음 시작")
        if audio_val and st.button("분석 시작"):
            with st.spinner("텍스트 변환 및 족보 매칭 중..."):
                txt = transcribe_audio_gemini(audio_val.getvalue())
                st.session_state.transcribed_text = txt
    
    with col2:
        st.markdown("### 분석 결과")
        if st.session_state.transcribed_text:
            st.write(st.session_state.transcribed_text)
            st.divider()
            st.markdown("**💡 추천 학습 포인트**")
            rels = find_relevant_questions(st.session_state.transcribed_text, st.session_state.selected_subject or "전체")
            for r in rels[:3]:
                st.info(f"관련 족보: {r['content']['text'][:100]}...")

# ==========================================
# 4. Main Execution
# ==========================================
def main():
    if not st.session_state.logged_in:
        login_screen()
    else:
        sidebar_ui()
        menu = st.tabs(["📝 학습하기 (Study)", "⚙️ 데이터 관리", "🎙️ 강의 녹음"])
        
        with menu[0]:
            main_study_ui()
        with menu[1]:
            management_ui()
        with menu[2]:
            record_ui()

if __name__ == "__main__":
    main()
