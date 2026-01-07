import os
import re
import sqlite3
import joblib
import pandas as pd
import streamlit as st
import speech_recognition as sr
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# 1. 초기 설정 및 DB 연결
# =========================
st.set_page_config(page_title="Med-Study AI Assistant", layout="wide")
user_id = st.sidebar.text_input("User ID", "med_student_01")
DATA_DIR = f"data/{user_id}"
os.makedirs(DATA_DIR, exist_ok=True)

conn = sqlite3.connect(os.path.join(DATA_DIR, "med_study.db"))
conn.execute("CREATE TABLE IF NOT EXISTS exam_db (id INTEGER PRIMARY KEY, year TEXT, text TEXT)")
conn.commit()

# 세션 상태 관리 (데이터 흐름 유지)
if 'pre_analysis' not in st.session_state: st.session_state.pre_analysis = []
if 'is_listening' not in st.session_state: st.session_state.is_listening = False

# =========================
# 2. 핵심 로직 (PDF 분석 및 검색)
# =========================
def get_pdf_text(file):
    reader = PdfReader(file)
    return [page.extract_text() for page in reader.pages]

def build_exam_index():
    rows = conn.execute("SELECT id, text FROM exam_db").fetchall()
    if not rows: return None
    texts = [r[1] for r in rows]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix, [r[0] for r in rows]

# =========================
# 3. UI 구성 (수업 전 -> 수업 중)
# =========================
st.title("🩺 스마트 강의록-족보 매칭 비서")

tab1, tab2 = st.tabs(["📅 수업 전: 족보 매칭 및 사전 정리", "🎤 수업 중: 실시간 음성 어시스턴트"])

# --- [Step 1: 수업 전 사전 분석] ---
with tab1:
    st.header("오늘 수업할 파일을 올려주세요")
    exam_files = st.file_uploader("먼저 보관 중인 '족보' PDF들을 등록하세요 (최초 1회)", type="pdf", accept_multiple_files=True, key="exams")
    
    if st.button("족보 DB 업데이트"):
        for f in exam_files:
            texts = get_pdf_text(f)
            conn.executemany("INSERT INTO exam_db (year, text) VALUES (?, ?)", [("2024", t) for t in texts if t])
        conn.commit()
        st.success("족보 데이터베이스가 구축되었습니다.")

    st.divider()
    lecture_file = st.file_uploader("오늘 수업용 '강의록' PDF 업로드", type="pdf", key="lecture")
    
    if lecture_file and st.button("수업 전 자동 단권화 분석 시작"):
        with st.spinner("강의록의 각 페이지와 족보를 대조 중..."):
            lecture_pages = get_pdf_text(lecture_file)
            vec, mat, pids = build_exam_index()
            
            analysis_results = []
            for i, page_text in enumerate(lecture_pages):
                if not page_text: continue
                qv = vec.transform([page_text])
                sims = cosine_similarity(qv, mat).flatten()
                if sims.max() > 0.3: # 유사도 0.3 이상만 추출
                    best_idx = sims.argmax()
                    exam_row = conn.execute("SELECT text FROM exam_db WHERE id=?", (pids[best_idx],)).fetchone()
                    analysis_results.append({"page": i+1, "score": sims.max(), "exam_text": exam_row[0]})
            
            st.session_state.pre_analysis = analysis_results
            st.success(f"분석 완료! 총 {len(analysis_results)}개의 페이지가 족보와 매칭됩니다.")

    # 분석 결과 시각화
    if st.session_state.pre_analysis:
        st.subheader("📊 오늘 강의 기출 포인트 리포트")
        for res in st.session_state.pre_analysis:
            with st.expander(f"📄 강의록 {res['page']}페이지 (기출 유사도: {int(res['score']*100)}%)"):
                st.info(f"**관련 족보 지문:** {res['exam_text'][:200]}...")

# --- [Step 2: 수업 중 실시간 음성 매칭] ---
with tab2:
    st.header("교수님 설명 실시간 트래킹")
    st.write("교수님의 설명을 들으며 관련 족보를 실시간으로 화면에 띄웁니다.")
    
    col_ctrl, col_view = st.columns([1, 2])
    
    with col_ctrl:
        if st.button("🎤 수업 시작 (음성 인식)"):
            r = sr.Recognizer()
            with sr.Microphone() as source:
                st.write("교수님 음성 청취 중...")
                try:
                    audio = r.listen(source, timeout=5, phrase_time_limit=10)
                    text = r.recognize_google(audio, language='ko-KR')
                    st.session_state.live_text = text
                    st.success(f"인식된 내용: {text}")
                except:
                    st.error("음성이 들리지 않거나 인식에 실패했습니다.")

    with col_view:
        if 'live_text' in st.session_state:
            st.subheader("🚨 실시간 매칭 알림")
            # 사전 분석된 결과 중에서 실시간 음성 키워드와 매칭되는 페이지 탐색
            matched = [res for res in st.session_state.pre_analysis if any(word in res['exam_text'] for word in st.session_state.live_text.split())]
            
            if matched:
                for m in matched:
                    st.warning(f"**지금 설명하시는 내용이 강의록 {m['page']}p 족보와 관련이 있습니다!**")
                    st.write(f"기출 내용 재확인: {m['exam_text'][:150]}...")
            else:
                st.write("실시간 일치 문항 없음")
