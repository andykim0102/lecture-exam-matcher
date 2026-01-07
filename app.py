import streamlit as st
import pandas as pd
import base64
import os
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_mic_recorder import mic_recorder

# =========================
# 1. 초기 설정 및 세션 관리
# =========================
st.set_page_config(page_title="Med-Study AI Visualizer", layout="wide")

if 'pre_analysis' not in st.session_state: st.session_state.pre_analysis = []
if 'exam_db' not in st.session_state: st.session_state.exam_db = []
if 'pdf_bytes' not in st.session_state: st.session_state.pdf_bytes = None

def get_pdf_text(file):
    reader = PdfReader(file)
    return [page.extract_text() or "" for page in reader.pages]

def display_pdf(file_bytes, page_num):
    """PDF를 베이스64로 인코딩하여 브라우저에 표시"""
    base64_pdf = base64.b64encode(file_bytes).decode('utf-8')
    # PDF 페이지 이동은 URL 파라미터 #page=N으로 조절
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}#page={page_num}" width="100%" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# =========================
# 2. UI 레이아웃
# =========================
st.title("🩺 Med-Study OS: 시각적 단권화 뷰어")

tab1, tab2 = st.tabs(["📂 데이터 준비 및 분석", "📖 시각적 단권화 뷰어 (수업 전/중)"])

with tab1:
    st.header("1. 족보 및 강의록 등록")
    col_a, col_b = st.columns(2)
    
    with col_a:
        exam_files = st.file_uploader("족보 PDF 등록", type="pdf", accept_multiple_files=True)
        if st.button("족보 인덱싱"):
            all_exams = []
            for f in exam_files:
                pages = get_pdf_text(f)
                all_exams.extend([{"info": f.name, "text": t} for t in pages if t.strip()])
            st.session_state.exam_db = all_exams
            st.success("족보 등록 완료")

    with col_b:
        lec_file = st.file_uploader("오늘의 강의록 PDF", type="pdf")
        if lec_file:
            st.session_state.pdf_bytes = lec_file.read()
            if st.button("사전 족보 매칭 분석"):
                lec_pages = get_pdf_text(lec_file)
                vec = TfidfVectorizer(ngram_range=(1, 2))
                exam_texts = [e['text'] for e in st.session_state.exam_db]
                if exam_texts:
                    mat = vec.fit_transform(exam_texts)
                    results = []
                    for i, p_text in enumerate(lec_pages):
                        if not p_text.strip(): continue
                        qv = vec.transform([p_text])
                        sims = cosine_similarity(qv, mat).flatten()
                        if sims.max() > 0.2:
                            best_idx = sims.argmax()
                            results.append({
                                "page": i + 1,
                                "exam_info": st.session_state.exam_db[best_idx]['info'],
                                "exam_text": st.session_state.exam_db[best_idx]['text']
                            })
                    st.session_state.pre_analysis = results
                    st.success("분석 완료! 뷰어 탭으로 이동하세요.")

# =========================
# 3. 시각적 단권화 뷰어 (핵심)
# =========================
with tab2:
    if st.session_state.pdf_bytes is None:
        st.info("먼저 강의록을 업로드해주세요.")
    else:
        # 좌측 상단 컨트롤러
        st.subheader("🧐 강의록-족보 매칭 뷰어")
        
        col_pdf, col_match = st.columns([1.2, 0.8])
        
        # 매칭된 페이지 리스트 추출
        matched_pages = [res['page'] for res in st.session_state.pre_analysis]
        
        with col_pdf:
            st.markdown("### 📄 강의록 원본")
            page_to_show = st.select_slider("페이지 선택", options=range(1, 50), value=1)
            display_pdf(st.session_state.pdf_bytes, page_to_show)

        with col_match:
            st.markdown("### 🚨 매칭된 족보 지문")
            
            # 실시간 녹음 기능 추가 (수업 중 상황 가정)
            st.write("🎙️ **실시간 강의 분석**")
            mic_recorder(start_prompt="수업 중 매칭 시작", stop_prompt="중지", key='viewer_mic')
            
            st.divider()
            
            # 현재 페이지에 매칭된 정보가 있는지 확인
            page_matches = [res for res in st.session_state.pre_analysis if res['page'] == page_to_show]
            
            if page_matches:
                st.success(f"현재 {page_to_show}페이지와 매칭된 기출이 있습니다!")
                for match in page_matches:
                    with st.container(border=True):
                        st.error(f"📍 관련 기출: {match['exam_info']}")
                        st.write(f"**과거 지문:** {match['exam_text'][:400]}...")
                        if st.button("📌 이 내용 노트에 추가", key=f"save_{page_to_show}"):
                            st.toast("단권화 노트에 저장되었습니다.")
            else:
                st.info("이 페이지와 관련된 기출 정보가 없습니다.")
                st.caption("기출이 없는 페이지는 개념 위주로 가볍게 학습하세요.")
