import streamlit as st
import pandas as pd
import sqlite3
import os
import time
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_mic_recorder import mic_recorder

# =========================
# 1. 초기 설정 및 세션 관리
# =========================
st.set_page_config(page_title="Med-Study Live Demo", layout="wide")
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# 데모용 가상 데이터 및 세션 상태 초기화
if 'lecture_analysis' not in st.session_state: st.session_state.lecture_analysis = []
if 'live_logs' not in st.session_state: st.session_state.live_logs = []

def get_db_connection():
    conn = sqlite3.connect(os.path.join(DATA_DIR, "med_exam.db"))
    # 족보 테이블: 내용뿐만 아니라 연도, 번호, 출제유형(키워드/지문/함정) 저장
    conn.execute("""
        CREATE TABLE IF NOT EXISTS exams (
            id INTEGER PRIMARY KEY,
            year TEXT,
            question_num TEXT,
            content TEXT,
            pattern TEXT
        )
    """)
    return conn

# =========================
# 2. 핵심 로직 (매칭 엔진)
# =========================
def search_exam_live(query, conn):
    rows = conn.execute("SELECT year, question_num, content, pattern FROM exams").fetchall()
    if not rows or not query: return []
    
    contents = [r[2] for r in rows]
    vec = TfidfVectorizer(ngram_range=(1, 2))
    mat = vec.fit_transform(contents)
    
    qv = vec.transform([query])
    sims = cosine_similarity(qv, mat).flatten()
    
    results = []
    for i in sims.argsort()[::-1]:
        if sims[i] > 0.15: # 매칭 임계값
            results.append({
                "year": rows[i][0],
                "num": rows[i][1],
                "content": rows[i][2],
                "pattern": rows[i][3],
                "score": sims[i]
            })
    return results

# =========================
# 3. UI 화면 구성
# =========================
st.title("🩺 의대생 실시간 족보 매칭 시스템 (Demo)")

tab1, tab2 = st.tabs(["📂 족보/강의록 사전 세팅", "🎙️ 수업 시작 (실시간 녹음)"])

# --- [Tab 1: 사전 세팅] ---
with tab1:
    st.header("수업 전 데이터 로드")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. 족보 데이터 등록")
        # 데모 편의를 위해 직접 입력 기능 추가
        with st.expander("직접 족보 데이터 입력 (데모용)"):
            ey = st.text_input("출제 연도", "2023")
            en = st.text_input("문항 번호", "15번")
            ep = st.selectbox("출제 유형", ["개념 정의", "증상 매칭", "치료법(오답 유도)"])
            ec = st.text_area("족보 지문 내용", "심근경색의 급성기 치료에서 ST분절 상승 여부에 따른 약물 선택 기준")
            if st.button("족보 추가"):
                conn = get_db_connection()
                conn.execute("INSERT INTO exams (year, question_num, content, pattern) VALUES (?,?,?,?)", (ey, en, ec, ep))
                conn.commit()
                st.success("족보 등록 완료!")

    with col2:
        st.subheader("2. 오늘 강의록 업로드")
        lec_file = st.file_uploader("강의록 PDF 업로드", type="pdf")
        if lec_file and st.button("강의록-족보 사전 대조"):
            st.success("분석 완료! 오늘 수업 중 3번의 기출 적중이 예상됩니다.")

# --- [Tab 2: 실시간 수업 모드] ---
with tab2:
    st.header("🎧 실시간 강의 분석 중")
    
    # 레이아웃 배치
    col_mic, col_status = st.columns([1, 2])
    
    with col_mic:
        st.write("교수님 음성을 인식합니다.")
        # 실시간 녹음 컨트롤
        audio = mic_recorder(start_prompt="🔴 녹음 시작 (강의 청취)", stop_prompt="⏹️ 중지", key='recorder')
        
        if audio:
            # 데모 상황 가정을 위해 인식된 텍스트 시뮬레이션 (실제 구현 시 STT API 연결)
            # 여기서는 예시로 '심근경색' 관련 발언을 했다고 가정
            st.audio(audio['bytes'])
            st.info("음성 분석 중...")
            time.sleep(1)
            simulated_text = "자, 이번 페이지에서는 심근경색 환자가 왔을 때 급성기에 어떤 약물을 먼저 써야 하는지, 특히 ST분절 상승이 중요하다고 했죠?"
            st.session_state.live_logs.append(simulated_text)
    
    with col_status:
        st.subheader("🚨 실시간 족보 적중 알림")
        if st.session_state.live_logs:
            current_speech = st.session_state.live_logs[-1]
            st.chat_message("professor").write(current_speech)
            
            # 매칭 검색
            conn = get_db_connection()
            hits = search_exam_live(current_speech, conn)
            
            if hits:
                for hit in hits:
                    st.toast(f"족보 적중! {hit['year']}년 {hit['num']}", icon="🔥")
                    with st.warning():
                        st.markdown(f"### 🚩 기출 정보: {hit['year']}년 {hit['num']}")
                        st.write(f"**출제 방식:** {hit['pattern']}")
                        st.write(f"**과거 지문:** {hit['content']}")
                        st.markdown("---")
                        st.caption("💡 Tip: 교수님이 이 부분 설명할 때 족보와 같은 키워드를 사용하셨습니다.")
            else:
                st.write("관련 기출 정보가 없습니다.")

# --- 하단 로그 ---
if st.session_state.live_logs:
    with st.expander("전체 강의 기록 보기"):
        for log in st.session_state.live_logs:
            st.text(log)
