import time
import streamlit as st
import google.generativeai as genai
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 0. 페이지 설정
# ==========================================
st.set_page_config(page_title="Med-Study OS", layout="wide", page_icon="🩺")
st.caption("📌 사용 흐름: 족보 학습 → 강의 페이지 분석 → 족보 기반 시험 포인트 확인")

# ==========================================
# 1. 세션 상태
# ==========================================
if "db" not in st.session_state:
    st.session_state.db = []

if "lecture_doc" not in st.session_state:
    st.session_state.lecture_doc = None

if "lecture_filename" not in st.session_state:
    st.session_state.lecture_filename = None

if "current_page" not in st.session_state:
    st.session_state.current_page = 0

if "text_models" not in st.session_state:
    st.session_state.text_models = []

if "best_text_model" not in st.session_state:
    st.session_state.best_text_model = None

if "api_key_ok" not in st.session_state:
    st.session_state.api_key_ok = False


# ==========================================
# 2. 핵심 설정값
# ==========================================
# 족보 근거 판단 임계값 (데모용 추천 0.70~0.75)
JOKBO_THRESHOLD = 0.72


# ==========================================
# 3. PDF / 임베딩 / 검색
# ==========================================
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text() or ""
        if text.strip():
            pages.append({"page": i + 1, "text": text, "source": file.name})
    return pages


def get_embedding(text: str):
    text = (text or "").strip()
    if not text:
        return []

    text = text[:12000]  # 안정성 컷

    try:
        return genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document",
        )["embedding"]
    except Exception:
        try:
            return genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document",
            )["embedding"]
        except Exception:
            return []


def find_relevant_jokbo(query_text: str, db: list[dict], top_k: int = 3):
    if not db:
        return []

    query_emb = get_embedding(query_text)
    if not query_emb:
        return []

    valid_items = [item for item in db if item.get("embedding")]
    if not valid_items:
        return []

    db_embs = [item["embedding"] for item in valid_items]
    sims = cosine_similarity([query_emb], db_embs)[0]
    top_idxs = np.argsort(sims)[::-1][:top_k]

    return [{"score": float(sims[i]), "content": valid_items[i]} for i in top_idxs]


def has_jokbo_evidence(related: list[dict]) -> bool:
    """족보 근거가 충분한지 판단"""
    if not related:
        return False
    return related[0]["score"] >= JOKBO_THRESHOLD


# ==========================================
# 4. 모델 자동 선택
# ==========================================
@st.cache_data(show_spinner=False)
def list_text_models(api_key: str):
    genai.configure(api_key=api_key)
    models = genai.list_models()
    return [
        m.name for m in models
        if "generateContent" in (m.supported_generation_methods or [])
    ]


def pick_best_text_model(model_names: list[str]):
    flash = [m for m in model_names if "flash" in m.lower()]
    return flash[0] if flash else (model_names[0] if model_names else None)


def generate_with_fallback(prompt: str, model_names: list[str]):
    last_err = None
    for name in model_names:
        try:
            model = genai.GenerativeModel(name)
            res = model.generate_content(prompt)
            return res.text, name
        except Exception as e:
            last_err = e
    raise last_err


# ==========================================
# 5. 프롬프트 (족보 근거 기반 ONLY)
# ==========================================
def build_jokbo_based_prompt(subject: str, lecture_text: str, jokbo_ctx: str):
    """
    ⚠️ 족보 근거가 있을 때만 호출됨
    """
    return f"""
너는 의대 시험 대비 조교다.
아래 '족보 발췌'에 근거해서만 답변하라.
추측, 확장, 새로운 해석은 금지한다.

[1️⃣ 족보에서 나온 형태]
- 족보에서 이 개념이 어떤 형태로 나왔는지 요약
- 정의 / 기전 / 비교 / 문제 유형 중 무엇이었는지 명시

[2️⃣ 강의 내용과의 연결]
- 강의 내용이 족보 내용과 어떻게 연결되는지 설명
- 족보 p.번호를 반드시 언급

[3️⃣ 시험 변형 가능성]
- 족보 문제를 어떻게 변형해서 낼 수 있는지 2가지

[4️⃣ 예상 문제]
- 족보 스타일을 유지한 객관식 1문항
- 정답 + 해설 포함

과목: {subject}

[강의 내용]
{lecture_text}

[족보 발췌]
{jokbo_ctx}
""".strip()
# ==========================================
# 6. 사이드바 (상태)
# ==========================================
with st.sidebar:
    st.title("🩺 Med-Study 상태")

    api_key = st.text_input("Gemini API Key", type="password")

    if api_key:
        try:
            genai.configure(api_key=api_key)
            models = list_text_models(api_key)
            if not models:
                st.session_state.api_key_ok = False
                st.error("generateContent 가능한 모델이 없습니다.")
            else:
                st.session_state.api_key_ok = True
                st.session_state.text_models = models
                st.session_state.best_text_model = pick_best_text_model(models)
                st.success("AI 연결 완료")
                st.caption(f"사용 모델(자동): {st.session_state.best_text_model}")
        except Exception as e:
            st.session_state.api_key_ok = False
            st.error(f"모델 조회 실패: {e}")

    st.divider()
    st.caption(f"📊 학습된 족보 페이지: **{len(st.session_state.db)}**")

    if st.button("족보 DB 초기화"):
        st.session_state.db = []
        st.rerun()


# ==========================================
# 7. 메인 UI
# ==========================================
tab1, tab2, tab3 = st.tabs(["📂 족보 학습", "📖 강의 공부", "⌨️ 실시간 텍스트 분석"])


# ==================================================
# TAB 1 — 족보 학습
# ==================================================
with tab1:
    st.header("📂 족보 학습")
    st.info("족보를 학습해 강의 내용과 시험 출제 근거를 연결합니다.")

    files = st.file_uploader("족보 PDF 업로드", type="pdf", accept_multiple_files=True)
    max_pages = st.number_input("파일당 최대 학습 페이지(데모용)", 1, 200, 30)

    if st.button("📚 시험 대비 DB 구축"):
        if not api_key or not st.session_state.api_key_ok:
            st.error("유효한 API Key를 먼저 설정하세요.")
            st.stop()
        if not files:
            st.warning("족보 PDF를 업로드하세요.")
            st.stop()

        bar = st.progress(0)
        status = st.empty()
        new_db = []

        for i, f in enumerate(files):
            pages = extract_text_from_pdf(f)[: int(max_pages)]
            for j, p in enumerate(pages):
                status.text(f"🧠 DB 구축: {f.name} ({j+1}/{len(pages)})")
                emb = get_embedding(p["text"])
                if emb:
                    p["embedding"] = emb
                    new_db.append(p)
                time.sleep(0.8)  # 429 완화
            bar.progress((i + 1) / len(files))

        st.session_state.db.extend(new_db)
        status.text("✅ 학습 완료")
        st.success(f"{len(new_db)} 페이지 학습 완료")


# ==================================================
# TAB 2 — 강의 공부 (족보 근거 → AI)
# ==================================================
with tab2:
    st.header("📖 강의 공부")
    st.info("강의 페이지의 내용이 족보에서 어떻게 나왔는지만 확인합니다.")

    # 과목 선택 + 기타 입력
    c1, c2 = st.columns([1, 2])
    with c1:
        subject_choice = st.selectbox("과목", ["해부학", "생리학", "약리학", "기타"], index=1)
    with c2:
        custom_subject = st.text_input("기타 과목명", disabled=(subject_choice != "기타"))

    subject_final = resolve_subject(subject_choice, custom_subject)
    st.caption(f"현재 과목: **{subject_final}**")

    lec_file = st.file_uploader("강의록 PDF", type="pdf", key="lec")

    if lec_file:
        if (
            st.session_state.lecture_doc is None
            or st.session_state.lecture_filename != lec_file.name
        ):
            st.session_state.lecture_doc = fitz.open(stream=lec_file.read(), filetype="pdf")
            st.session_state.lecture_filename = lec_file.name
            st.session_state.current_page = 0

        doc = st.session_state.lecture_doc
        col_view, col_ai = st.columns([6, 4])

        with col_view:
            b1, b2, b3 = st.columns([1, 2, 1])
            if b1.button("◀"):
                if st.session_state.current_page > 0:
                    st.session_state.current_page -= 1
            b2.markdown(
                f"<center>{st.session_state.current_page+1}/{len(doc)}</center>",
                unsafe_allow_html=True
            )
            if b3.button("▶"):
                if st.session_state.current_page < len(doc) - 1:
                    st.session_state.current_page += 1

            page = doc.load_page(st.session_state.current_page)
            pix = page.get_pixmap(dpi=150)
            st.image(Image.frombytes("RGB", [pix.width, pix.height], pix.samples),
                     use_container_width=True)
            page_text = (page.get_text() or "").strip()

        with col_ai:
            st.subheader("🔎 족보 근거")
            if st.button("이 페이지 확인"):
                if not st.session_state.db:
                    st.error("족보 DB가 없습니다.")
                    st.stop()
                if not page_text:
                    st.warning("텍스트가 없는 페이지입니다.")
                    st.stop()

                related = find_relevant_jokbo(page_text, st.session_state.db, top_k=3)

                if not has_jokbo_evidence(related):
                    st.warning("📌 관련 족보 근거를 찾지 못했습니다. (AI 분석 생략)")
                    st.stop()

                # 1) 족보 근거 먼저 표시
                for i, r in enumerate(related):
                    with st.expander(f"족보 근거 #{i+1} (유사도 {r['score']:.3f})"):
                        st.write(f"페이지 {r['content']['page']}")
                        st.write(r["content"]["text"])

                # 2) 근거가 있을 때만 AI 호출
                jokbo_ctx = "\n".join(
                    f"- (p{r['content']['page']}) {r['content']['text'][:300]}"
                    for r in related
                )

                prompt = build_jokbo_based_prompt(
                    subject=subject_final,
                    lecture_text=page_text,
                    jokbo_ctx=jokbo_ctx
                )

                models = st.session_state.text_models or []
                fallback = models + ["models/gemini-1.5-flash-latest"]

                with st.spinner("족보 기반 분석 중..."):
                    result, used = generate_with_fallback(prompt, fallback)
                    st.caption(f"사용 모델: {used}")
                    st.markdown(result)


# ==================================================
# TAB 3 — 실시간 텍스트 분석 (족보 근거 있을 때만)
# ==================================================
with tab3:
    st.header("⌨️ 실시간 텍스트 분석")
    st.info("족보에 근거가 있을 때만 분석합니다.")

    c1, c2 = st.columns([1, 2])
    with c1:
        subject_choice_live = st.selectbox("과목", ["해부학", "생리학", "약리학", "기타"], index=1)
    with c2:
        custom_subject_live = st.text_input("기타 과목명", disabled=(subject_choice_live != "기타"))

    subject_final_live = resolve_subject(subject_choice_live, custom_subject_live)
    st.caption(f"현재 과목: **{subject_final_live}**")

    user_text = st.text_area(
        "강의 중 중요한 문장을 그대로 입력",
        height=140,
        placeholder="예) 이 단계는 시험에 자주 나오는 포인트다."
    )

    if st.button("족보 연결 확인"):
        if not st.session_state.db:
            st.error("족보 DB가 없습니다.")
            st.stop()

        query = (user_text or "").strip()
        if not query:
            st.error("텍스트를 입력하세요.")
            st.stop()

        related = find_relevant_jokbo(query, st.session_state.db, top_k=3)

        if not has_jokbo_evidence(related):
            st.warning("📌 관련 족보 근거를 찾지 못했습니다. (AI 분석 생략)")
            st.stop()

        for i, r in enumerate(related):
            with st.expander(f"족보 근거 #{i+1} (유사도 {r['score']:.3f})"):
                st.write(f"페이지 {r['content']['page']}")
                st.write(r["content"]["text"])

        jokbo_ctx = "\n".join(
            f"- (p{r['content']['page']}) {r['content']['text'][:300]}"
            for r in related
        )

        prompt = build_jokbo_based_prompt(
            subject=subject_final_live,
            lecture_text=query,
            jokbo_ctx=jokbo_ctx
        )

        models = st.session_state.text_models or []
        fallback = models + ["models/gemini-1.5-flash-latest"]

        with st.spinner("족보 기반 분석 중..."):
            result, used = generate_with_fallback(prompt, fallback)
            st.caption(f"사용 모델: {used}")
            st.markdown(result)
