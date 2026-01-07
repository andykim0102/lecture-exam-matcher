import os
import re
import sqlite3
import joblib
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_mic_recorder import mic_recorder

# =========================
# 1. 초기 설정 및 DB 연결
# =========================
st.set_page_config(page_title="Med-Study AI Assistant", layout="wide")
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# 세션 상태 초기화 (NameError 방지)
if 'pre_analysis' not in st.session_state: st.session_state.pre_analysis = []
if 'bundle' not in st.session_state: st.session_state.bundle = None

def get_db_connection(user_id):
    conn = sqlite3.connect(os.path.join(DATA_DIR, f"{user_id}.db"))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS exams (
            id INTEGER PRIMARY KEY, year TEXT, num TEXT, content TEXT, pattern TEXT
        )
    """)
    return conn

def get_pdf_text(file):
    reader = PdfReader(file)
    return [page.extract_text() or "" for page in reader.pages]

# =========================
# 2. 메인 UI 화면
# =========================
st.title("🩺 스마트 강의록-족보 매칭 비서")
user_id = st.sidebar.text_input("사용자 ID", "demo_user")
conn = get_db_connection(user_id)

tab1, tab2, tab3 = st.tabs(["📅 수업 전: 자동 분석", "🎙️ 수업 중: 실시간 매칭", "🎯 수업 후: 복습"])

# --- [Tab 1: 수업 전 사전 분석] ---
with tab1:
    st.header("강의실 가기 전: 기출 포인트 미리보기")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. 족보 데이터 등록")
        exam_files = st.file_uploader("족보 PDF 등록", type="pdf", accept_multiple_files=True)
        if st.button("족보 인덱싱 시작"):
            for f in exam_files:
                texts = get_pdf_text(f)
                conn.executemany("INSERT INTO exams (year, num, content, pattern) VALUES (?,?,?,?)", 
                                 [("2024", "미지정", t, "일반") for t in texts if t.strip()])
            conn.commit()
            
            # 인덱스 구축
            rows = conn.execute("SELECT content FROM exams").fetchall()
            if rows:
                texts = [r[0] for r in rows]
                vec = TfidfVectorizer(ngram_range=(1, 2))
                mat = vec.fit_transform(texts)
                st.session_state.bundle = {"vectorizer": vec, "matrix": mat, "raw": rows}
                st.success("족보 데이터베이스 구축 완료!")

    with col2:
        st.subheader("2. 오늘 강의록 분석")
        lec_file = st.file_uploader("강의록 PDF", type="pdf")
        if lec_file and st.button("수업 전 자동 단권화"):
            if st.session_state.bundle:
                lec_pages = get_pdf_text(lec_file)
                bundle = st.session_state.bundle
                analysis = []
                for i, p_text in enumerate(lec_pages):
                    if not p_text.strip(): continue
                    qv = bundle["vectorizer"].transform([p_text])
                    sims = cosine_similarity(qv, bundle["matrix"]).flatten()
                    if sims.max() > 0.25:
                        best_idx = sims.argmax()
                        analysis.append({"page": i+1, "score": sims.max(), "content": bundle["raw"][best_idx][0]})
                st.session_state.pre_analysis = analysis
                st.success(f"분석 완료! {len(analysis)}개 페이지에서 족보 관련성 발견.")
            else:
                st.error("먼저 족보를 등록해 주세요.")

# --- [Tab 2: 수업 중 실시간 매칭] ---
with tab2:
    st.header("🎧 실시간 강의 트래킹")
    if not st.session_state.pre_analysis:
        st.warning("수업 전 분석을 먼저 완료해 주세요.")
    else:
        st.info("녹음 버튼을 누르면 교수님의 설명을 분석하여 관련 족보 정보를 즉시 띄웁니다.")
        
        # 실제 음성 녹음 도구
        audio = mic_recorder(start_prompt="🔴 교수님 설명 녹음 시작", stop_prompt="⏹️ 중지 및 분석", key='recorder')
        
        if audio:
            st.audio(audio['bytes'])
            # 데모용: 인식된 텍스트 시뮬레이션 (실제 구현 시 OpenAI Whisper 등 연동)
            simulated_speech = "이 수용체 기전은 작년 국시에도 나왔고 아주 중요합니다."
            st.subheader(f"인식된 강의 내용: {simulated_speech}")
            
            # 실시간 매칭 알림
            hits = [item for item in st.session_state.pre_analysis if any(word in item['content'] for word in simulated_speech.split()[:3])]
            if hits:
                for hit in hits:
                    st.toast(f"🔥 족보 적중! 강의록 {hit['page']}p", icon="🚨")
                    with st.warning():
                        st.markdown(f"### 🚨 실시간 족보 적중 (강의록 {hit['page']}페이지 관련)")
                        st.write(f"**과거 기출 내용:** {hit['content'][:200]}...")
            else:
                st.write("현재 발언과 일치하는 족보 정보가 없습니다.")

# --- [Tab 3: 수업 후 복습] ---
with tab3:
    st.header("오늘의 요약 및 단권화")
    if st.session_state.pre_analysis:
        df_results = pd.DataFrame(st.session_state.pre_analysis)
        st.dataframe(df_results)
        
        # Anki 카드 생성 기능
        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Anki 카드용 CSV 다운로드", csv, "anki_cards.csv", "text/csv")
    else:
        st.write("표시할 분석 데이터가 없습니다.")
