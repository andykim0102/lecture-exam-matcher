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
