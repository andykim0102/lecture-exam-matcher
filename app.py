import os
import re
import sqlite3
import time
import joblib
import pandas as pd
import streamlit as st
from dataclasses import dataclass
from typing import List, Tuple, Any
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# 1. Config & Directory
# =========================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

def safe_filename(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9가-힣._-]+", "_", s.strip())[:64] or "user"

def user_dir(user_id: str) -> str:
    d = os.path.join(DATA_DIR, safe_filename(user_id))
    os.makedirs(d, exist_ok=True)
    return d

# =========================
# 2. Database & Search Logic
# =========================
def db_connect(user_id: str):
    conn = sqlite3.connect(os.path.join(user_dir(user_id), "user.db"))
    conn.execute("CREATE TABLE IF NOT EXISTS pages (id INTEGER PRIMARY KEY, doc_name TEXT, page_num INTEGER, text TEXT)")
    return conn

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\x00", " ")).strip()

def extract_pdf_pages(pdf_bytes: bytes):
    reader = PdfReader(pdf_bytes)
    return [(i + 1, normalize(p.extract_text() or "")) for i, p in enumerate(reader.pages)]

@dataclass
class IndexBundle:
    vectorizer: TfidfVectorizer
    matrix: Any
    page_ids: List[int]

def build_index(conn):
    rows = conn.execute("SELECT id, text FROM pages").fetchall()
    if not rows: return None
    texts, pids = [r[1] for r in rows], [r[0] for r in rows]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6))
    return IndexBundle(vec, vec.fit_transform(texts), pids)

def search_exam(conn, bundle: IndexBundle, query: str):
    qv = bundle.vectorizer.transform([query])
    sims = cosine_similarity(qv, bundle.matrix).flatten()
    results = []
    for i in sims.argsort()[::-1][:5]:
        if sims[i] <= 0.1: continue
        row = conn.execute("SELECT doc_name, page_num, text FROM pages WHERE id=?", (bundle.page_ids[i],)).fetchone()
        results.append({"score": float(sims[i]), "doc": row[0], "page": row[1], "text": row[2]})
    return results

# =========================
# 3. Main UI Flow
# =========================
st.set_page_config(page_title="의대생 학습 OS", layout="wide")
st.title("🩺 Med-Study OS: 족보 매칭 & 암기 비서")

# 사이드바: 유저 관리
user_id = st.sidebar.text_input("사용자 ID", "medical_student_01")
conn = db_connect(user_id)
index_path = os.path.join(user_dir(user_id), "index.joblib")

# 세션 상태 초기화 (데이터 흐름 유지의 핵심)
if 'match_data' not in st.session_state:
    st.session_state.match_data = None

tab1, tab2, tab3 = st.tabs(["📤 족보/강의록 등록", "⚡ 수업 중 (Live)", "🎯 수업 후 (복습)"])

# --- Tab 1: 데이터 빌드업 ---
with tab1:
    st.header("학기 초: 족보 및 강의록 인덱싱")
    files = st.file_uploader("PDF 업로드", type="pdf", accept_multiple_files=True)
    if st.button("파일 데이터베이스 저장"):
        for f in files:
            pages = extract_pdf_pages(f.getvalue())
            conn.executemany("INSERT INTO pages (doc_name, page_num, text) VALUES (?, ?, ?)", 
                             [(f.name, p, t) for p, t in pages if t])
        conn.commit()
        st.success("데이터 저장 완료!")

    if st.button("AI 검색 엔진 최적화 (Index Build)"):
        bundle = build_index(conn)
        joblib.dump(bundle, index_path)
        st.success("인덱싱 완료! 이제 실시간 매칭이 가능합니다.")

# --- Tab 2: 수업 중 실시간 어시스턴트 ---
with tab2:
    st.header("실시간 강의 매칭 엔진")
    bundle = joblib.load(index_path) if os.path.exists(index_path) else None
    
    if not bundle:
        st.warning("먼저 Tab 1에서 인덱스를 구축해주세요.")
    else:
        live_note = st.text_area("✍️ 교수님 강조 사항 / 실시간 필기", placeholder="교수님이 언급하신 키워드를 적으세요...")
        
        if live_note:
            results = search_exam(conn, bundle, live_note)
            if results:
                st.session_state.match_data = results # 복습 탭으로 데이터 전달
                st.subheader("🚨 관련 기출 족보 탐지!")
                for r in results:
                    with st.expander(f"📍 {r['doc']} (p.{r['page']}) - 유사도 {int(r['score']*100)}%"):
                        st.write(r['text'])
                        st.progress(r['score'])
            else:
                st.info("현재 입력과 관련된 과거 기출이 없습니다.")

# --- Tab 3: 수업 후 인텔리전트 복습 ---
with tab3:
    st.header("복습 및 암기 최적화")
    
    if not st.session_state.match_data:
        st.info("수업 중 매칭된 데이터가 없습니다. 필기를 먼저 진행해주세요.")
    else:
        df = pd.DataFrame(st.session_state.match_data)
        
        sub_tab1, sub_tab2, sub_tab3 = st.tabs(["📝 단권화 리포트", "🧠 암기(Anki)", "🤖 AI 기억법"])
        
        with sub_tab1:
            st.subheader("오늘의 기출 우선순위")
            st.error(f"가장 중요한 키워드: {df.iloc[0]['doc']}의 개념")
            st.table(df[['doc', 'page', 'score']])
            
        with sub_tab2:
            st.subheader("Anki 카드 추출")
            anki_df = df[['doc', 'text']].rename(columns={'doc': 'Front', 'text': 'Back'})
            st.download_button("Anki용 CSV 받기", anki_df.to_csv(index=False).encode('utf-8'), "anki.csv")
            
        with sub_tab3:
            st.subheader("AI Mnemonics (기억법)")
            topic = st.selectbox("암기가 필요한 구간", df['text'].str[:50])
            if st.button("기억의 궁전 스토리 생성"):
                st.success("생성 완료!")
                st.write(f"👉 '{topic}...' 을(를) 외우기 위해 당신의 책상 위 오른쪽 모서리에 이 개념이 놓여있다고 상상하세요!")
