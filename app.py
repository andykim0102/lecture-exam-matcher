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
st.set_page_config(
    page_title="Med-Study OS",
    layout="wide",
    page_icon="🩺"
)

st.caption("📌 사용 흐름: 족보 학습 → 강의 페이지 분석 → 실시간 시험 포인트 정리")

# ==========================================
# 1. 세션 상태 초기화
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
# 2. 핵심 로직 함수
# ==========================================
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text() or ""
        if text.strip():
            pages.append({
                "page": i + 1,
                "text": text,
                "source": file.name
            })
    return pages


def get_embedding(text):
    text = (text or "").strip()[:12000]
    if not text:
        return []

    try:
        return genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )["embedding"]
    except Exception:
        try:
            return genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )["embedding"]
        except Exception:
            return []


def find_relevant_jokbo(query, db, top_k=3):
    if not db:
        return []

    query_emb = get_embedding(query)
    if not query_emb:
        return []

    valid = [d for d in db if d.get("embedding")]
    if not valid:
        return []

    embs = [d["embedding"] for d in valid]
    sims = cosine_similarity([query_emb], embs)[0]
    idxs = np.argsort(sims)[::-1][:top_k]

    return [
        {"score": float(sims[i]), "content": valid[i]}
        for i in idxs
    ]


@st.cache_data(show_spinner=False)
def list_text_models(api_key):
    genai.configure(api_key=api_key)
    models = genai.list_models()
    return [
        m.name for m in models
        if "generateContent" in (m.supported_generation_methods or [])
    ]


def pick_best_text_model(names):
    flash = [n for n in names if "flash" in n.lower()]
    return flash[0] if flash else (names[0] if names else None)


def generate_with_fallback(prompt, model_names):
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
# 3. 사이드바 (시스템 상태)
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
                st.error("사용 가능한 모델 없음")
            else:
                st.session_state.api_key_ok = True
                st.session_state.text_models = models
                st.session_state.best_text_model = pick_best_text_model(models)
                st.success("AI 연결 완료")
        except Exception as e:
            st.session_state.api_key_ok = False
            st.error(f"연결 실패: {e}")

    st.divider()
    st.caption(
        f"""
📊 시스템 현황  
- 족보 페이지 수: {len(st.session_state.db)}  
- 사용 모델: {st.session_state.best_text_model or "미선택"}
"""
    )

    if st.button("족보 DB 초기화"):
        st.session_state.db = []
        st.rerun()


# ==========================================
# 4. 메인 UI
# ==========================================
tab1, tab2, tab3 = st.tabs([
    "📂 족보 학습",
    "📖 강의 공부",
    "⌨️ 실시간 텍스트 분석"
])


# ==================================================
# TAB 1 — 족보 학습
# ==================================================
with tab1:
    st.header("📂 족보 학습")
    st.info("과거 시험 족보를 학습시켜, 강의 내용과 자동 연결합니다.")

    files = st.file_uploader(
        "족보 PDF 업로드",
        type="pdf",
        accept_multiple_files=True
    )

    max_pages = st.number_input(
        "파일당 최대 학습 페이지 (데모용)",
        min_value=1,
        max_value=200,
        value=30
    )

    if st.button("📚 시험 대비 DB 구축 시작"):
        if not api_key:
            st.error("API Key를 입력하세요.")
            st.stop()

        progress = st.progress(0)
        status = st.empty()
        new_db = []

        for i, f in enumerate(files or []):
            pages = extract_text_from_pdf(f)[:max_pages]

            for j, p in enumerate(pages):
                status.text(
                    f"🧠 시험 대비 DB 구축 중: {f.name} "
                    f"({j+1}/{len(pages)} 페이지)"
                )
                emb = get_embedding(p["text"])
                if emb:
                    p["embedding"] = emb
                    new_db.append(p)
                time.sleep(0.8)

            progress.progress((i + 1) / len(files))

        st.session_state.db.extend(new_db)
        st.success(f"✅ 총 {len(new_db)} 페이지 학습 완료")

        st.info(
            "다음 단계 👉 **강의 공부 탭**에서 강의 페이지를 열고 분석하세요."
        )
# ==================================================
# TAB 2 — 강의 공부 (페이지 분석)
# ==================================================
with tab2:
    st.header("📖 강의 공부")
    st.info("강의 페이지를 한 장씩 보면서, 시험과의 연결 포인트를 즉시 분석합니다.")

    lec_file = st.file_uploader("강의록 PDF 업로드", type="pdf", key="lecture")

    if lec_file:
        # 새 파일이면 다시 로드
        if (
            st.session_state.lecture_doc is None
            or st.session_state.lecture_filename != lec_file.name
        ):
            st.session_state.lecture_doc = fitz.open(
                stream=lec_file.read(),
                filetype="pdf"
            )
            st.session_state.lecture_filename = lec_file.name
            st.session_state.current_page = 0

        doc = st.session_state.lecture_doc
        col_view, col_ai = st.columns([6, 4])

        # ---------- 왼쪽: PDF 뷰어 ----------
        with col_view:
            nav1, nav2, nav3 = st.columns([1, 2, 1])

            if nav1.button("◀ 이전"):
                if st.session_state.current_page > 0:
                    st.session_state.current_page -= 1

            nav2.markdown(
                f"<center>{st.session_state.current_page + 1} / {len(doc)}</center>",
                unsafe_allow_html=True
            )

            if nav3.button("다음 ▶"):
                if st.session_state.current_page < len(doc) - 1:
                    st.session_state.current_page += 1

            page = doc.load_page(st.session_state.current_page)
            pix = page.get_pixmap(dpi=150)
            st.image(
                Image.frombytes("RGB", [pix.width, pix.height], pix.samples),
                use_container_width=True
            )

            page_text = (page.get_text() or "").strip()

        # ---------- 오른쪽: AI 분석 ----------
        with col_ai:
            st.subheader("🧠 시험 대비 AI 분석")

            if st.button("⚡ 이 페이지 분석"):
                if not api_key or not st.session_state.api_key_ok:
                    st.error("API Key 상태를 확인하세요.")
                    st.stop()

                if not st.session_state.db:
                    st.error("족보 DB가 없습니다. 먼저 족보를 학습하세요.")
                    st.stop()

                if not page_text:
                    st.warning("이 페이지에는 텍스트가 없습니다 (스캔 이미지 가능).")
                    st.stop()

                with st.spinner("시험 포인트 분석 중..."):
                    related = find_relevant_jokbo(
                        page_text,
                        st.session_state.db,
                        top_k=3
                    )

                    jokbo_ctx = "\n".join([
                        f"- (p{r['content']['page']}) {r['content']['text'][:200]}"
                        for r in related
                    ])

                    prompt = f"""
너는 의대 시험 대비 조교다.
아래 형식을 반드시 지켜서 답변해.

[1️⃣ 핵심 개념]
- bullet 5개

[2️⃣ 족보 연결]
- 족보 페이지 번호를 반드시 언급
- '기출 변형 / 반복 개념 / 새 강조점'으로 구분

[3️⃣ 예상 문제]
- 객관식 2문항
- 단답형 1문항
- 각 문제의 정답과 해설 포함

---
[강의 페이지 텍스트]
{page_text}

[관련 족보 발췌]
{jokbo_ctx if jokbo_ctx else "(관련 족보 없음)"}
""".strip()

                    model_list = st.session_state.text_models or []
                    fallback = model_list + [
                        "models/gemini-1.5-flash-latest",
                        "models/gemini-1.5-pro-latest"
                    ]

                    try:
                        result, used = generate_with_fallback(prompt, fallback)
                        st.caption(f"사용 모델: {used}")

                        st.markdown("### 🔍 분석 결과")
                        st.markdown(result)

                    except Exception as e:
                        st.error(f"분석 실패: {e}")


# ==================================================
# TAB 3 — 실시간 텍스트 분석 (강의 중 메모용)
# ==================================================
with tab3:
    st.header("⌨️ 실시간 텍스트 분석")
    st.info(
        "강의 중 교수님이 강조한 문장을 바로 입력하면, "
        "시험 출제 가능성을 즉시 분석합니다."
    )

    if not api_key or not st.session_state.api_key_ok:
        st.warning("사이드바에서 API Key를 먼저 설정하세요.")
        st.stop()

    if not st.session_state.db:
        st.warning("족보 DB가 없습니다. 먼저 족보를 학습하세요.")
        st.stop()

    user_input = st.text_area(
        "🚨 교수님이 '중요하다 / 시험에 낼 수 있다'고 말한 내용을 그대로 입력하세요",
        height=160,
        placeholder="예) 이 기전은 교과서에는 없지만 임상적으로 중요하다..."
    )

    if st.button("📊 시험 출제 가능성 분석"):
        query = user_input.strip()
        if not query:
            st.error("분석할 텍스트를 입력하세요.")
            st.stop()

        with st.spinner("족보 연결 중..."):
            related = find_relevant_jokbo(query, st.session_state.db, top_k=3)

        st.subheader("🔎 족보와의 연결")
        context_str = ""

        if not related:
            st.write("→ 기존 족보에는 없는 새로운 강조점일 가능성이 큽니다.")
        else:
            for i, r in enumerate(related):
                with st.expander(
                    f"관련 족보 #{i+1} (유사도 {r['score']:.3f})"
                ):
                    st.write(f"페이지 {r['content']['page']}")
                    st.write(r["content"]["text"])
                context_str += (
                    f"- (페이지 {r['content']['page']}) "
                    f"{r['content']['text']}\n"
                )

        st.divider()
        st.subheader("🩺 Med-Study 시험 인사이트")

        final_prompt = f"""
상황: 의대 강의 중 실시간 시험 대비 정리.

교수님 발언:
{query}

관련 족보:
{context_str if context_str else "(관련 족보 없음)"}

미션:
1. 이 발언이 시험에 나올 가능성을 ★☆☆☆☆~★★★★★로 평가.
2. 그 이유를 족보 관점에서 설명.
3. 예상 문제 2개 + 정답/해설.
4. 바로 외울 수 있는 '암기 포인트' 5줄.
""".strip()

        model_list = st.session_state.text_models or []
        fallback = model_list + [
            "models/gemini-1.5-flash-latest",
            "models/gemini-1.5-pro-latest"
        ]

        try:
            result, used = generate_with_fallback(final_prompt, fallback)
            st.caption(f"사용 모델: {used}")
            st.markdown(result)
        except Exception as e:
            st.error(f"분석 실패: {e}")
