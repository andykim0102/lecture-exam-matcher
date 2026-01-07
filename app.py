import os
import re
import sqlite3
from dataclasses import dataclass
from typing import List, Tuple

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

def user_dir(user_id: str) -> str:
    d = os.path.join(DATA_DIR, safe_filename(user_id))
    os.makedirs(d, exist_ok=True)
    return d

def user_db_path(user_id: str) -> str:
    return os.path.join(user_dir(user_id), "user.db")

def user_index_path(user_id: str) -> str:
    return os.path.join(user_dir(user_id), "tfidf_index.joblib")

def safe_filename(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^a-zA-Z0-9가-힣._-]+", "_", s)
    return s[:64] if s else "user"

# =========================
# DB
# =========================
def db_connect(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS pages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_name TEXT NOT NULL,
        page_num INTEGER NOT NULL,
        text TEXT NOT NULL
    );
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT
    );
    """)
    conn.commit()
    return conn

def db_insert_pages(conn, doc_name: str, pages: List[Tuple[int, str]]):
    conn.executemany(
        "INSERT INTO pages(doc_name, page_num, text) VALUES (?, ?, ?)",
        [(doc_name, pnum, txt) for pnum, txt in pages if txt and txt.strip()]
    )
    conn.commit()

def db_clear(conn):
    conn.execute("DELETE FROM pages;")
    conn.execute("DELETE FROM meta;")
    conn.commit()

def db_fetch_all_pages(conn) -> List[Tuple[int, str, int, str]]:
    # returns (id, doc_name, page_num, text)
    cur = conn.execute("SELECT id, doc_name, page_num, text FROM pages ORDER BY id ASC;")
    return cur.fetchall()

# =========================
# PDF Parsing
# =========================
def extract_pdf_pages(pdf_bytes: bytes) -> List[Tuple[int, str]]:
    reader = PdfReader(pdf_bytes)
    out = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        txt = normalize_text(txt)
        out.append((i + 1, txt))
    return out

def normalize_text(t: str) -> str:
    t = t.replace("\x00", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

# =========================
# Indexing & Search
# =========================
@dataclass
class IndexBundle:
    vectorizer: TfidfVectorizer
    matrix
    page_ids: List[int]  # maps row -> pages.id

def build_index(conn) -> IndexBundle:
    rows = db_fetch_all_pages(conn)
    if not rows:
        raise ValueError("No pages to index.")

    page_ids = [r[0] for r in rows]
    texts = [r[3] for r in rows]

    # 한국어는 형태소 분석 없이도 데모는 충분히 가능하게 char-ngrams로 잡습니다.
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 6),
        min_df=1,
        max_df=0.95
    )
    matrix = vectorizer.fit_transform(texts)
    return IndexBundle(vectorizer=vectorizer, matrix=matrix, page_ids=page_ids)

def search_index(conn, bundle: IndexBundle, query: str, top_k: int = 8):
    query = normalize_text(query)
    if not query:
        return []

    qv = bundle.vectorizer.transform([query])
    sims = cosine_similarity(qv, bundle.matrix).flatten()
    ranked = sims.argsort()[::-1][:top_k]

    # fetch rows for ranked page_ids
    results = []
    for idx in ranked:
        score = float(sims[idx])
        if score <= 0:
            continue
        page_id = bundle.page_ids[idx]
        cur = conn.execute("SELECT doc_name, page_num, text FROM pages WHERE id=?;", (page_id,))
        row = cur.fetchone()
        if row:
            doc_name, page_num, text = row
            results.append({
                "score": score,
                "doc_name": doc_name,
                "page_num": page_num,
                "snippet": make_snippet(text, query),
                "full_text": text
            })
    return results

def make_snippet(text: str, query: str, width: int = 240) -> str:
    if not text:
        return ""
    q = query.strip()
    pos = text.find(q)
    if pos == -1:
        # fallback: first chunk
        return (text[:width] + "…") if len(text) > width else text
    start = max(0, pos - width // 3)
    end = min(len(text), pos + width)
    snippet = text[start:end]
    if start > 0:
        snippet = "…" + snippet
    if end < len(text):
        snippet = snippet + "…"
    return snippet

# =========================
# Simple “Note Draft” (Post-class)
# =========================
def extract_key_sentences(text: str, max_sentences: int = 8) -> List[str]:
    # 아주 단순한 문장 분리 + 길이 기반 필터 (데모용)
    sents = re.split(r"(?<=[.!?。]|다\.)\s+", text.strip())
    sents = [s.strip() for s in sents if len(s.strip()) >= 20]
    return sents[:max_sentences]

def draft_one_page_note(lecture_text: str, matched_pages: List[dict]) -> str:
    out = []
    out.append("## 오늘 강의 핵심 요약(초안)")
    for s in extract_key_sentences(lecture_text, 6):
        out.append(f"- {s}")

    out.append("\n## 족보/기출 연결(초안)")
    if not matched_pages:
        out.append("- (매칭된 페이지 없음) 키워드를 더 구체화해보세요.")
    else:
        for r in matched_pages[:6]:
            out.append(f"- [{r['doc_name']} p.{r['page_num']}] (유사도 {r['score']:.3f}) {r['snippet']}")

    out.append("\n## 단권화 정리 템플릿(채워넣기)")
    out.append("- 정의/개념:")
    out.append("- 왜 중요한가(시험 포인트):")
    out.append("- 자주 나오는 문제 패턴:")
    out.append("- 실수 포인트/함정:")
    out.append("- 1줄 암기문장:")
    return "\n".join(out)

# =========================
# UI
# =========================
st.set_page_config(page_title="족보-강의 매칭 데모", layout="wide")

st.title("📚 족보/강의 매칭 데모 (세미 클로즈드 + 벌크 업로드 + 매칭)")

with st.sidebar:
    st.subheader("🔐 데모 로그인")
    user_id = st.text_input("닉네임(사용자 ID)", value=st.session_state.get("user_id", "taeyop"))
    user_id = safe_filename(user_id)
    st.session_state["user_id"] = user_id

    st.divider()
    st.caption("사용자별로 DB/인덱스가 분리됩니다.")
    st.write(f"현재 사용자: **{user_id}**")

# Init DB
conn = db_connect(user_db_path(user_id))

# Load index if exists
bundle = None
index_path = user_index_path(user_id)
if os.path.exists(index_path):
    try:
        bundle = joblib.load(index_path)
    except Exception:
        bundle = None

tab1, tab2, tab3, tab4 = st.tabs(["1) 업로드/인덱싱", "2) Pre-class(예습 추천)", "3) In-class(즉시 매칭)", "4) Post-class(단권화 초안)"])

# -------------------------
# 1) Upload / Index
# -------------------------
with tab1:
    st.subheader("1) 벌크 업로드 + 자동 인덱싱")
    colA, colB = st.columns([2, 1], gap="large")

    with colA:
        files = st.file_uploader(
            "PDF 여러 개를 한 번에 올려주세요 (족보/강의록 등)",
            type=["pdf"],
            accept_multiple_files=True
        )

        if st.button("📥 업로드 반영(텍스트 추출 → DB 저장)"):
            if not files:
                st.warning("업로드할 PDF를 선택해주세요.")
            else:
                total_pages = 0
                for f in files:
                    pages = extract_pdf_pages(f.getvalue())
                    db_insert_pages(conn, f.name, pages)
                    total_pages += len(pages)
                st.success(f"완료: {len(files)}개 PDF, 총 {total_pages}페이지를 DB에 저장했습니다.")

    with colB:
        st.write("**현재 데이터 상태**")
        rows = db_fetch_all_pages(conn)
        st.metric("저장된 페이지 수", len(rows))

        if st.button("🧠 인덱스 빌드/갱신 (검색 준비)"):
            rows = db_fetch_all_pages(conn)
            if not rows:
                st.warning("DB에 페이지가 없습니다. 먼저 PDF를 업로드하세요.")
            else:
                bundle = build_index(conn)
                joblib.dump(bundle, index_path)
                st.success("인덱스를 생성/갱신했습니다. 이제 검색/매칭이 됩니다.")

        if st.button("🗑️ 이 사용자 데이터 초기화(데모용)"):
            db_clear(conn)
            if os.path.exists(index_path):
                os.remove(index_path)
            st.success("DB/인덱스를 초기화했습니다.")

    st.divider()
    st.subheader("🔎 빠른 검색(전체)")
    query = st.text_input("검색어(키워드/문장)", placeholder="예: 산화환원, Simpson rule, 혈압 조절, ...")
    topk = st.slider("상위 결과 수", 3, 15, 8)

    if st.button("검색 실행"):
        if bundle is None:
            st.error("인덱스가 없습니다. 먼저 '인덱스 빌드/갱신'을 눌러주세요.")
        else:
            results = search_index(conn, bundle, query, top_k=topk)
            if not results:
                st.info("결과가 없어요. 키워드를 바꿔보세요.")
            else:
                for r in results:
                    with st.expander(f"({r['score']:.3f}) {r['doc_name']} / p.{r['page_num']}"):
                        st.write(r["snippet"])
                        st.caption("원문(일부)")
                        st.text(r["full_text"][:1200] + ("…" if len(r["full_text"]) > 1200 else ""))

# -------------------------
# 2) Pre-class
# -------------------------
with tab2:
    st.subheader("2) Pre-class: 오늘 강의 예습(기출 비중 높은 페이지 추천)")

    st.write("강의 주제/키워드를 넣으면, 업로드한 족보/강의록에서 **가장 관련 높은 페이지**를 추천합니다.")
    lecture_topic = st.text_area("오늘 들을 강의 키워드(여러 줄 가능)", height=120, placeholder="예: 심장 전기생리, 활동전위, refractory period ...")

    if st.button("🎯 예습 추천 생성"):
        if bundle is None:
            st.error("인덱스가 없습니다. 먼저 업로드/인덱싱 탭에서 인덱스를 만들어주세요.")
        else:
            results = search_index(conn, bundle, lecture_topic, top_k=10)
            if not results:
                st.info("추천이 안 나왔어요. 키워드를 더 구체화해보세요.")
            else:
                st.success("예습 추천 페이지 TOP 10")
                for r in results:
                    st.markdown(f"- **{r['doc_name']} p.{r['page_num']}** (유사도 {r['score']:.3f}) — {r['snippet']}")

# -------------------------
# 3) In-class
# -------------------------
with tab3:
    st.subheader('3) In-class: "교수님이 중요하다" 순간 즉시 매칭(데모)')
    st.write("실시간 음성 인식까지는 데모 2차에서 붙이고, 1차 데모에서는 **핵심 키워드 입력 → 즉시 우측 패널 매칭**으로 체감하게 만듭니다.")

    colL, colR = st.columns([1, 1], gap="large")
    with colL:
        live_keyword = st.text_input('교수님 멘트/키워드(예: "이거 시험에 나옴", "Starling curve")')
        if st.button("⚡ 즉시 매칭"):
            if bundle is None:
                st.error("인덱스가 없습니다.")
            else:
                st.session_state["live_results"] = search_index(conn, bundle, live_keyword, top_k=8)

    with colR:
        st.write("### 📌 관련 기출/족보(우측 패널)")
        results = st.session_state.get("live_results", [])
        if not results:
            st.caption("왼쪽에서 키워드를 넣고 '즉시 매칭'을 눌러보세요.")
        else:
            for r in results:
                st.markdown(f"**({r['score']:.3f}) {r['doc_name']} / p.{r['page_num']}**")
                st.write(r["snippet"])
                st.divider()

# -------------------------
# 4) Post-class
# -------------------------
with tab4:
    st.subheader("4) Post-class: 강의록 + 족보를 합친 '나만의 단권화 노트' 초안")
    st.write("강의 메모(또는 강의록 텍스트)를 붙여넣으면, 관련 페이지를 끌어와서 단권화 템플릿으로 초안을 뽑습니다. (데모 1차는 비LLM)")

    lecture_note = st.text_area("오늘 강의 메모/강의록 텍스트", height=180, placeholder="수업 직후 메모를 그대로 붙여넣기")
    match_hint = st.text_input("추가 매칭 힌트(선택)", placeholder="예: '시험', '기출', '정의', 특정 개념명…")

    if st.button("🧾 단권화 노트 초안 생성"):
        if bundle is None:
            st.error("인덱스가 없습니다.")
        else:
            q = lecture_note + "\n" + match_hint
            matched = search_index(conn, bundle, q, top_k=10)
            draft = draft_one_page_note(lecture_note, matched)
            st.text_area("생성된 초안", value=draft, height=360)

st.caption("데모 1차: 로컬에서 돌아가는 검색/매칭 체감용 MVP (사용자별 DB 분리 + 벌크 업로드 + 페이지 매칭).")
