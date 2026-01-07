import os
import re
import sqlite3
import joblib
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 음성 인식 라이브러리는 설치 오류를 방지하기 위해 예외 처리를 추가합니다.
try:
    import speech_recognition as sr
except ImportError:
    sr = None

# =========================
# 1. 환경 설정 및 세션 초기화
# =========================
st.set_page_config(page_title="Med-Study AI Assistant", layout="wide")
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# 데이터 흐름을 유지하기 위한 세션 상태 정의
if 'pre_analysis' not in st.session_state: st.session_state.pre_analysis = []
if 'bundle' not in st.session_state: st.session_state.bundle = None

# =========================
# 2. 핵심 로직 함수
# =========================
def db_connect(user_id):
    path = os.path.join(DATA_DIR, f"{user_id}.db")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE IF NOT EXISTS exam_db (id INTEGER PRIMARY KEY, text TEXT)")
    return conn

def get_pdf_text(file):
    reader = PdfReader(file)
    return [page.extract_text() or "" for page in reader.pages]

def build_index(conn):
    rows = conn.execute("SELECT id, text FROM exam_db").fetchall()
    if not rows: return None
    texts = [r[1] for r in rows]
    vec = TfidfVectorizer(ngram_range=(1, 2))
    mat = vec.fit_transform(texts)
    return {"vectorizer": vec, "matrix": mat, "ids": [r[0] for r in rows]}

# =========================
# 3. 메인 UI 화면
# =========================
st.title("🩺 스마트 강의록-족보 매칭 비서")
user_id = st.sidebar.text_input("사용자 ID", "demo_user")
conn = db_connect(user_id)

tab1, tab2, tab3 = st.tabs(["📅 수업 전: 자동 정리", "🎤 수업 중: 실시간 매칭", "🎯 수업 후: 복습 리포트"])

# --- [Step 1: 수업 전 사전 분석] ---
with tab1:
    st.header("오늘 수업할 파일을 분석합니다")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1단계: 족보 라이브러리 구축")
        exam_files = st.file_uploader("족보 PDF 등록", type="pdf", accept_multiple_files=True)
        if st.button("족보 DB 저장 및 인덱싱"):
            for f in exam_files:
                texts = get_pdf_text(f)
                conn.executemany("INSERT INTO exam_db (text) VALUES (?)", [(t,) for t in texts if t.strip()])
            conn.commit()
            st.session_state.bundle = build_index(conn)
            st.success("족보 인덱싱이 완료되었습니다.")

    with col2:
        st.subheader("2단계: 오늘 강의록 분석")
        lec_file = st.file_uploader("수업용 강의록 PDF", type="pdf")
        if lec_file and st.button("수업 전 자동 단권화 실행"):
            if st.session_state.bundle:
                lec_pages = get_pdf_text(lec_file)
                bundle = st.session_state.bundle
                
                analysis = []
                for i, p_text in enumerate(lec_pages):
                    if not p_text.strip(): continue
                    qv = bundle["vectorizer"].transform([p_text])
                    sims = cosine_similarity(qv, bundle["matrix"]).flatten()
                    if sims.max() > 0.2: # 유사도 기준값
                        best_idx = sims.argmax()
                        exam_txt = conn.execute("SELECT text FROM exam_db WHERE id=?", (bundle["ids"][best_idx],)).fetchone()[0]
                        analysis.append({"page": i+1, "score": sims.max(), "exam_text": exam_txt})
                
                st.session_state.pre_analysis = analysis
                st.success(f"분석 완료! {len(analysis)}개 페이지에서 기출 흔적 발견.")
            else:
                st.error("먼저 족보 인덱싱을 진행해 주세요.")

# --- [Step 2: 수업 중 실시간 매칭] ---
with tab2:
    st.header("교수님 설명 실시간 트래킹")
    if not st.session_state.pre_analysis:
        st.warning("수업 전 분석을 먼저 완료해 주세요.")
    else:
        st.info("교수님의 설명을 듣고 오늘 배운 내용 중 족보 관련 내용을 즉시 띄웁니다.")
        
        # 음성 인식 UI
        if sr is None:
            st.error("SpeechRecognition 라이브러리가 설치되지 않았습니다.")
        else:
            if st.button("🎤 교수님 설명 듣기 (10초)"):
                r = sr.Recognizer()
                with sr.Microphone() as source:
                    with st.spinner("듣는 중..."):
                        try:
                            audio = r.listen(source, timeout=10)
                            text = r.recognize_google(audio, language='ko-KR')
                            st.subheader(f"인식된 내용: {text}")
                            
                            # 실시간 매칭 로직
                            for item in st.session_state.pre_analysis:
                                if any(word in item['exam_text'] for word in text.split()):
                                    st.warning(f"🚨 **지금 설명 중인 내용이 {item['page']}페이지 족보와 관련이 있습니다!**")
                                    st.write(f"기출 요약: {item['exam_text'][:200]}...")
                        except:
                            st.error("음성 인식에 실패했습니다. 마이크 설정을 확인하세요.")

# --- [Step 3: 수업 후 복습 리포트] ---
with tab3:
    st.header("오늘의 단권화 요약")
    if st.session_state.pre_analysis:
        df = pd.DataFrame(st.session_state.pre_analysis)
        st.table(df[['page', 'score']])
        
        # Anki용 데이터 추출
        anki_csv = df[['page', 'exam_text']].to_csv(index=False).encode('utf-8')
        st.download_button("📥 오늘 기출 기반 Anki 카드 다운로드", anki_csv, "anki_cards.csv", "text/csv")
    else:
        st.write("표시할 데이터가 없습니다.")
