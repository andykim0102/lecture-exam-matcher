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
import streamlit as st
import pandas as pd
# PDF 생성 및 이미지 처리를 위한 라이브러리 (필요 시 설치: pip install fpdf)
from fpdf import FPDF 

# --- [차별점 1] 족보 매칭 기반 우선순위 및 인사이트 ---
st.divider()
st.header("🚀 의대생 맞춤형 Post-Class 엔진")

# 가상의 매칭 데이터(df_results)가 있다고 가정
if not df_results.empty:
    # 기출 횟수에 따른 우선순위 계산 로직 추가
    df_results['priority_score'] = df_results['match_count'] * 10  # 예시 로직
    
    st.subheader("📍 오늘 강의의 핵심 '족보' 포인트")
    top_picks = df_results.nlargest(3, 'priority_score')
    for i, row in top_picks.iterrows():
        st.error(f"**중요!** '{row['lecture_keyword']}' 관련 내용은 최근 5년간 {row['match_count']}회 출제되었습니다.")

    # --- [차별점 2] 드래그 & 드롭 대용: AI 노트 정리 (아이디어 3번 반영) ---
    st.subheader("📝 AI 스마트 노트 생성")
    with st.expander("강의록과 족보를 합친 '단권화 초안' 보기"):
        st.write("AI가 매칭된 데이터를 바탕으로 요약 노트를 생성했습니다.")
        summary_text = ""
        for i, row in df_results.iterrows():
            summary_text += f"- **{row['lecture_keyword']}**: {row['exam_content']} (기출: {row['year']}년)\n"
        st.info(summary_text)
        
        # 노트 저장 기능
        st.download_button("나만의 요약 노트(.txt) 저장", summary_text)

    # --- [차별점 3] 기억법 서비스: 암기 스토리텔링 (아이디어 4번 반영) ---
    st.subheader("🧠 암기 최적화: 기억의 궁전 & Mnemonics")
    selected_topic = st.selectbox("어떤 개념이 안 외워지나요?", df_results['lecture_keyword'].unique())
    
    if st.button(f"'{selected_topic}' 암기법 생성"):
        with st.spinner('암기 스토리를 만드는 중...'):
            # 실제 서비스 시 GPT API 연동 구간
            st.success("생성 완료! 아래 시나리오로 외워보세요.")
            st.markdown(f"""
            > **Mnemonic Scenario:** > "{selected_topic}"을 외우기 위해 **[기억의 궁전]** 거실에 있는 소파를 떠올려보세요. 
            > 소파 위에 {df_results[df_results['lecture_keyword']==selected_topic]['exam_content'].values[0]}가 
            > 거대하게 놓여있다고 상상하며 연결하는 겁니다!
            """)

    # --- [차별점 4] Anki 연동 (실행 성능 향상) ---
    st.subheader("📥 외부 앱 연동")
    col1, col2 = st.columns(2)
    with col1:
        # CSV 포맷으로 Anki 카드 생성
        anki_csv = df_results[['lecture_keyword', 'exam_content']].to_csv(index=False).encode('utf-8')
        st.download_button("Anki 카드 세트(.csv) 다운로드", anki_csv, "anki_cards.csv", "text/csv")
    with col2:
        if st.button("iPad 굿노트용 PDF 내보내기"):
            st.write("매칭 주석이 포함된 PDF를 생성 중입니다...")

else:
    st.warning("먼저 강의록과 족보 파일을 업로드하여 매칭을 진행해 주세요.")

import streamlit as st
import time

# --- [차별점: 수업 중 실시간 어시스턴트] ---
st.divider()
st.header("⚡ 실시간 수업 모드 (In-class Live)")

# 수업 중 모드 활성화 스위치
live_mode = st.toggle("실시간 수업 어시스턴트 시작")

if live_mode:
    st.info("🎤 교수님의 설명을 분석하여 관련 족보를 실시간으로 탐색합니다.")
    
    # 레이아웃 분할: 왼쪽(실시간 필기/STT), 오른쪽(실시간 족보 알림)
    col_live, col_match = st.columns([1, 1])
    
    with col_live:
        st.subheader("📝 실시간 강의 요약")
        # 실제 구현 시 음성 인식(STT) 라이브러리 연동 구간
        user_note = st.text_area("교수님 강조 사항이나 필기를 입력하세요 (또는 음성 인식 중...)", 
                                 placeholder="예: '이 수용체 기전은 작년 국시에도 나왔고...'")
        
    with col_match:
        st.subheader("🚨 실시간 족보 매칭")
        if user_note:
            # 실시간 입력 내용과 기존 업로드된 df_results(족보) 매칭 시뮬레이션
            with st.spinner('관련 기출 확인 중...'):
                time.sleep(0.5) # 분석 처리 속도 시뮬레이션
                
                # 입력된 텍스트에 족보 키워드가 포함되었는지 간단 체크
                matched_found = False
                for i, row in df_results.iterrows():
                    if row['lecture_keyword'] in user_note:
                        st.warning(f"**기출 일치!** [{row['year']}년] {row['exam_content']}")
                        st.caption(f"우선순위: {'🔥'* (int(row['match_count']))}")
                        matched_found = True
                
                if not matched_found:
                    st.write("아직 일치하는 과거 기출 문항이 없습니다.")

    # --- [차별점: 실시간 드래그 & 드롭 대안] ---
    st.subheader("📸 실시간 화면 캡처 및 태깅")
    if st.button("현재 슬라이드 족보 태그와 함께 저장"):
        st.success("현재 강의록 페이지가 2023년 기출 정보와 매칭되어 '단권화 후보'로 등록되었습니다.")

else:
    st.write("수업 시작 시 위 토글을 켜주세요.")
