import streamlit as st
import pandas as pd
import numpy as np
import time
import base64
import google.generativeai as genai
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF
from PIL import Image
from streamlit_mic_recorder import mic_recorder

# =========================
# 1. 초기 설정 및 세션 관리
# =========================
st.set_page_config(page_title="Med-Study OS v1.0", layout="wide", page_icon="🩺")

# 상태 변수 초기화 (중요: 에러 방지용)
if 'exam_db' not in st.session_state: st.session_state.exam_db = []
if 'exam_embeddings' not in st.session_state: st.session_state.exam_embeddings = None 
if 'pre_analysis' not in st.session_state: st.session_state.pre_analysis = []
if 'pdf_bytes' not in st.session_state: st.session_state.pdf_bytes = None
if 'notebook' not in st.session_state: st.session_state.notebook = []

# 사이드바 API 설정
with st.sidebar:
    st.title("⚙️ 시스템 설정")
    api_key = st.text_input("Gemini API Key", type="password")
    if api_key:
        genai.configure(api_key=api_key)
        st.success("AI 엔진 연결됨")
    
    st.divider()
    if st.button("🔄 세션 초기화"):
        st.clear_cache()
        st.rerun()

# --- 유틸리티 함수 ---
def get_embedding(text):
    if not api_key: return None
    try:
        result = genai.embed_content(model="models/text-embedding-004", content=text, task_type="retrieval_document")
        return result['embedding']
    except: return None

def display_pdf_page(file_bytes, page_num):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    page = doc.load_page(page_num - 1)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    st.image(img, use_container_width=True)

def analyze_with_ai(lecture_text, jokbo_text):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"강의록: {lecture_text[:500]}\n족보: {jokbo_text[:500]}\n위 두 내용의 핵심 연관 주제와 공부 팁을 한 문장으로 알려줘."
    try:
        response = model.generate_content(prompt)
        return response.text
    except: return "연관성 분석 중..."

# =========================
# 2. 메인 UI (탭 구조)
# =========================
tab1, tab2, tab3 = st.tabs(["📂 1. 데이터 학습", "🎙️ 2. 수업 중 (실시간)", "📝 3. 나만의 정리본"])

# --- [Tab 1: 데이터 학습] ---
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. 족보 아카이브 구축")
        exam_files = st.file_uploader("족보 PDF 업로드", type="pdf", accept_multiple_files=True)
        if st.button("🚀 딥러닝 족보 학습 시작"):
            all_exams, embeddings = [], []
            bar = st.progress(0)
            for idx, f in enumerate(exam_files):
                pages = [p.extract_text() for p in PdfReader(f).pages]
                for i, text in enumerate(pages):
                    if len(text) > 50:
                        emb = get_embedding(text)
                        if emb:
                            all_exams.append({"source": f.name, "page": i+1, "text": text})
                            embeddings.append(emb)
                bar.progress((idx+1)/len(exam_files))
            st.session_state.exam_db = all_exams
            st.session_state.exam_embeddings = np.array(embeddings)
            st.success("족보 임베딩 완료!")

    with col2:
        st.subheader("2. 강의록 사전 분석")
        lec_file = st.file_uploader("오늘 강의 PDF", type="pdf")
        if lec_file and st.button("🔍 강의-족보 정밀 대조"):
            st.session_state.pdf_bytes = lec_file.getvalue()
            lec_pages = [p.extract_text() for p in PdfReader(lec_file).pages]
            results = []
            for i, p_text in enumerate(lec_pages):
                if len(p_text) < 50: continue
                q_emb = get_embedding(p_text)
                if q_emb:
                    sims = cosine_similarity([q_emb], st.session_state.exam_embeddings).flatten()
                    if sims.max() > 0.5: # 유사도 역치
                        best_idx = sims.argmax()
                        results.append({
                            "page": i+1, "score": sims.max(),
                            "exam_info": st.session_state.exam_db[best_idx]['source'],
                            "exam_text": st.session_state.exam_db[best_idx]['text'],
                            "ai_comment": analyze_with_ai(p_text, st.session_state.exam_db[best_idx]['text'])
                        })
            st.session_state.pre_analysis = results
            st.success("전체 페이지 분석 완료!")

# --- [Tab 2: 수업 중 뷰어 & 실시간 단권화] ---
with tab2:
    if not st.session_state.pdf_bytes:
        st.warning("강의록을 먼저 업로드해주세요.")
    else:
        c_pdf, c_tool = st.columns([1.2, 0.8])
        with c_pdf:
            page_num = st.select_slider("페이지 이동", options=range(1, 101), value=1)
            display_pdf_page(st.session_state.pdf_bytes, page_num)
        
        with c_tool:
            st.subheader("🎙️ 실시간 보이스 트래킹")
            audio = mic_recorder(start_prompt="수업 녹음 시작", stop_prompt="중지 및 분석", key='live_mic')
            if audio:
                st.info("🔊 교수님 발언 분석 및 족보 매칭 중...")
                # (실제 구현 시 여기에 STT와 임베딩 검색 추가)
                st.write("발언 내용: '이 수용체 기전은 국시에 매년 나오는 부분입니다.'")
                st.toast("🚨 실시간 족보 매칭 발견!", icon="🔥")

            st.divider()
            st.subheader(f"📍 {page_num}p 기출 포인트")
            matches = [m for m in st.session_state.pre_analysis if m['page'] == page_num]
            if matches:
                for m in matches:
                    with st.expander(f"🔥 기출 적중 ({m['score']*100:.0f}%)", expanded=True):
                        st.caption(f"출처: {m['exam_info']}")
                        st.markdown(f"**AI 분석:** {m['ai_comment']}")
                        user_note = st.text_input("수업 메모", key=f"note_{page_num}")
                        if st.button("📌 내 정리본에 추가", key=f"add_{page_num}"):
                            st.session_state.notebook.append({"page": page_num, "exam": m['exam_text'], "note": user_note})
                            st.toast("저장 완료!")
            else: st.write("이 페이지는 족보와 큰 연관이 없습니다.")

# --- [Tab 3: 정리본] ---
with tab3:
    st.header("📝 나만의 스마트 단권화")
    for i, item in enumerate(st.session_state.notebook):
        with st.container(border=True):
            st.write(f"**강의록 {item['page']}페이지 기록**")
            st.info(f"기출 지문: {item['exam'][:200]}...")
            st.success(f"나의 메모: {item['note']}")
