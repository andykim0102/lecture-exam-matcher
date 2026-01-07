import os
import re
import sqlite3
from dataclasses import dataclass
from typing import List, Tuple, Any

import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# =========================
# Config
# =========================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

def safe_filename(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^a-zA-Z0-9가-힣._-]+", "_", s)
    return s[:64] if s else "user"

def user_dir(user_id: str) -> str:
    d = os.path.join(DATA_DIR, safe_filename(user_id))
    os.makedirs(d, exist_ok=True)
    return d

def user_db_path(user_id: str) -> str:
    return os.path.join(user_dir(user_id), "user.db")

def user_index_path(user_id: str) -> str:
    return os.path.join(user_dir(user_id), "tfidf_index.joblib")

# =========================
# DB
# =========================
def db_connect(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS pages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_name TEXT,
        page_num INTEGER,
        text TEXT
    )
    """)
    conn.commit()
    return conn

def db_insert_pages(conn, doc_name: str, pages: List[Tuple[int, str]]):
    conn.executemany(
        "INSERT INTO pages (doc_name, page_num, text) VALUES (?, ?, ?)",
        [(doc_name, p, t) for p, t in pages if t.strip()]
    )
    conn.commit()

def db_fetch_all(conn):
    cur = conn.execute("SELECT id, doc_name, page_num, text FROM pages")
    return cur.fetchall()

# =========================
# PDF
# =========================
def extract_pdf_pages(pdf_bytes: bytes):
    reader = PdfReader(pdf_bytes)
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append((i + 1, normalize(text)))
    return pages

def normalize(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# =========================
# Index
# =========================
@dataclass
class IndexBundle:
    vectorizer: TfidfVectorizer
    matrix: Any
    page_ids: List[int]

def build_index(conn):
    rows = db_fetch_all(conn)
    texts = [r[3] for r in rows]
    page_ids = [r[0] for r in rows]

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 6),
        min_df=1,
        max_df=0.95
    )
    matrix = vectorizer.fit_transform(texts)
    return IndexBundle(vectorizer, matrix, page_ids)

def search(conn, bundle: IndexBundle, query: str, k: int = 5):
    qv = bundle.vectorizer.transform([query])
    sims = cosine_similarity(qv, bundle.matrix).flatten()
    idxs = sims.argsort()[::-1][:k]

    results = []
    for i in idxs:
        if sims[i] <= 0:
            continue
        pid = bundle.page_ids[i]
        row = conn.execute(
            "SELECT doc_name, page_num, text FROM pages WHERE id=?",
            (pid,)
        ).fetchone()
        results.append({
            "score": float(sims[i]),
            "doc": row[0],
            "page": row[1],
            "text": row[2][:300] + "..."
        })
    return results

# =========================
# UI
# =========================
st.set_page_config(page_title="Lecture–Exam Matcher", layout="wide")
st.title("📚 Lecture–Exam Matcher (Demo)")

user_id = st.sidebar.text_input("User ID", "demo_user")
conn = db_connect(user_db_path(user_id))

index_path = user_index_path(user_id)
bundle = joblib.load(index_path) if os.path.exists(index_path) else None

tab1, tab2 = st.tabs(["📤 Upload & Index", "🔍 Search"])

with tab1:
    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Save PDFs"):
        for f in files:
            pages = extract_pdf_pages(f.getvalue())
            db_insert_pages(conn, f.name, pages)
        st.success("PDFs saved")

    if st.button("Build Index"):
        bundle = build_index(conn)
        joblib.dump(bundle, index_path)
        st.success("Index built")

with tab2:
    query = st.text_input("Search keyword")
    if st.button("Search") and bundle:
        results = search(conn, bundle, query)
        for r in results:
            st.markdown(
                f"**{r['doc']} p.{r['page']}** (score {r['score']:.2f})"
            )
            st.write(r["text"])
# --- 기존 매칭 결과가 'df_results'라는 데이터프레임에 있다고 가정할 때 ---

st.divider() # 시각적 구분선
st.header("🎯 수업 후: 복습 및 단권화 지원")

# 1. 탭으로 기능 분류 (깔끔한 UI)
tab1, tab2, tab3 = st.tabs(["📄 단권화 노트", "🧠 암기 카드(Anki)", "🤖 AI 퀴즈"])

with tab1:
    st.subheader("족보 주석 포함 PDF 생성")
    st.write("강의록의 기출 구간에 족보 번호를 자동으로 입힌 PDF를 생성합니다.")
    if st.button("단권화 PDF 다운로드 (준비 중)"):
        # 여기에 PyMuPDF(fitz) 등을 활용한 PDF 편집 로직이 들어갑니다.
        st.info("현재 개발 중인 기능입니다. 기출 위치가 표시된 레이어를 생성합니다.")

with tab2:
    st.subheader("Anki 카드 세트 추출")
    st.write("오늘 매칭된 족보 문항을 기반으로 Anki(.csv) 파일을 만듭니다.")
    
    # 예시 데이터 생성 로직
    if not df_results.empty:
        anki_data = df_results[['lecture_keyword', 'exam_content']].rename(
            columns={'lecture_keyword': 'Front', 'exam_content': 'Back'}
        )
        csv = anki_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Anki용 CSV 다운로드",
            data=csv,
            file_name='medical_anki_cards.csv',
            mime='text/csv',
        )
    else:
        st.warning("매칭된 데이터가 없어 카드를 만들 수 없습니다.")

with tab3:
    st.subheader("AI 예측 변형 문제")
    # GPT API가 연결되어 있다면 매칭된 내용을 프롬프트로 전달
    if st.button("오늘의 핵심 퀴즈 생성"):
        with st.spinner('AI가 족보 패턴을 분석 중...'):
            # 가상의 결과 예시
            st.success("분석 완료!")
            st.markdown("""
            **Q. 다음 중 오늘 배운 'A 기전'의 족보 빈출 오답 유형은?**
            1. 증상과 약물을 반대로 매칭
            2. 발병 시기를 2주에서 4주로 변경
            3. 유전 형식을 우성에서 열성으로 변경
            
            *정답은 **3번**입니다. 작년 족보에서 이 부분이 함정으로 나왔습니다.*
            """)
