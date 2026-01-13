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
st.caption("📌 사용 흐름: 족보 학습 → 강의 페이지 분석 → 실시간 텍스트 분석")

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
# 1.5. 족보 근거 판단 임계값
# ==========================================
# 데모 추천: 0.70~0.75 사이. 너무 높으면 "없음"이 자주 뜸.
JOKBO_THRESHOLD = 0.72

def has_jokbo_evidence(related: list[dict]) -> bool:
    """관련 족보가 '있다'고 판단할 최소 조건"""
    return bool(related) and related[0]["score"] >= JOKBO_THRESHOLD


# ==========================================
# 2. PDF/임베딩/검색 함수
# ==========================================
def extract_text_from_pdf(file):
    """PDF를 텍스트로 변환 (fitz 사용)"""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text() or ""
        if text.strip():
            pages.append({"page": i + 1, "text": text, "source": file.name})
    return pages


def get_embedding(text: str):
    """임베딩 생성 (가능하면 text-embedding-004, 아니면 embedding-001)"""
    text = (text or "").strip()
    if not text:
        return []

    # 데모 안정성: 너무 긴 텍스트 컷
    text = text[:12000]

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
    """유사도 검색"""
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


# ==========================================
# 3. 모델 자동 선택 + fallback
# ==========================================
@st.cache_data(show_spinner=False)
def list_text_models(api_key: str):
    genai.configure(api_key=api_key)
    models = genai.list_models()
    out = []
    for m in models:
        methods = getattr(m, "supported_generation_methods", []) or []
        if "generateContent" in methods:
            out.append(m.name)  # 보통 "models/..." 형태
    return out


def pick_best_text_model(model_names: list[str]):
    if not model_names:
        return None
    flash = [m for m in model_names if "flash" in m.lower()]
    return flash[0] if flash else model_names[0]


def generate_with_fallback(prompt: str, model_names: list[str]):
    last_err = None
    for name in model_names:
        if not name:
            continue
        try:
            model = genai.GenerativeModel(name)
            res = model.generate_content(prompt)
            text = getattr(res, "text", None)
            if text:
                return text, name
            return str(res), name
        except Exception as e:
            last_err = e
            continue
    raise last_err


# ==========================================
# 4. 과목 선택(기타 포함) + 프롬프트 템플릿
# ==========================================
def resolve_subject(selected: str, custom: str) -> str:
    """UI에서 선택된 과목을 최종 과목명으로 확정"""
    selected = (selected or "").strip()
    if selected != "기타":
        return selected
    custom = (custom or "").strip()
    return custom if custom else "기타(미입력)"


def get_subject_templates():
    """해부/생리/약리 전용 템플릿 + 기타(범용)"""
    return {
        "해부학": {
            "focus": [
                "구조 이름(한글+영어), 위치 관계(인접 구조/층), 지배 신경/혈관",
                "손상/병변 시 임상 징후(근력저하/감각저하/반사 등)",
                "그림/표지(landmark), 통과 구조, 구멍/관(Foramen/Canal) 출제 포인트",
            ],
            "question_style": "구조-기능-임상 연결, 위치/지배/통과 구조를 헷갈리게 내는 객관식/단답형",
        },
        "생리학": {
            "focus": [
                "기전 흐름(A→B→C), 조절(피드백), 항상성 의미",
                "변수 변화 방향(증가/감소), 그래프/표로 나올 포인트",
                "대표 예외/헷갈리는 케이스(수용체, 호르몬, 교감/부교감 등)",
            ],
            "question_style": "기전 순서·변수 변화·그래프 해석·실험 상황 추론",
        },
        "약리학": {
            "focus": [
                "작용기전(MOA) → 효과 → 부작용 → 금기/주의",
                "같은 계열/유사 기전 약물 비교(차이점) + 대표 약물명",
                "임상 시나리오 기반(부작용 회피/대체약 선택/상호작용)",
            ],
            "question_style": "기전-부작용 매칭, 계열 비교, 임상 케이스에서 약 선택 문제",
        },
        "__GENERIC__": {
            "focus": [
                "핵심 개념을 시험 키워드 중심으로 요약",
                "자주 출제되는 헷갈 포인트(개념 구분/정의/비교)",
                "객관식/단답형으로 나올 만한 ‘정확한 표현’ 강조",
            ],
            "question_style": "정의·비교·기전/흐름·예외/함정 포인트 중심의 전형적 의대 시험 스타일",
        },
    }


def build_exam_prompt(subject: str, lecture_text: str, jokbo_ctx: str, mode: str):
    """
    ⚠️ 이 함수는 '족보 근거가 있을 때만' 호출되도록 TAB2/TAB3에서 막아둠.
    mode:
      - "page": 강의 페이지 분석
      - "live": 실시간 텍스트 분석
    """
    templates = get_subject_templates()
    t = templates.get(subject, templates["__GENERIC__"])

    # 기타 과목이면 "범용 의대 과목" 프레이밍 추가
    if subject not in ["해부학", "생리학", "약리학"]:
        subject_note = (
            f"과목명: {subject}\n"
            "너는 이 과목의 일반적인 의대 시험 출제 관점(정의/비교/기전/함정 포인트)을 적용해 분석하라.\n"
            "과목이 정확히 무엇이든, '의대 시험 대비'라는 목적을 최우선으로 하라."
        )
    else:
        subject_note = f"과목명: {subject}"

    base_rules = f"""
너는 의대 시험 출제자이자 채점자(그리고 조교)다.
아래 형식을 반드시 지켜라.

중요 원칙:
- 아래 [관련 족보 발췌]에 근거해서만 답하라.
- 새로운 해석/추측은 하지 마라.

[1️⃣ 족보에서 나온 형태]
- 정의/기전/비교/문제유형 중 어떤 형태였는지 명시
- 핵심 키워드 5개

[2️⃣ 강의 내용과의 연결]
- 강의 내용이 족보 어디(p.번호)에 해당하는지 설명
- 족보 발췌에서 근거 문장/키워드 2개 이상 짧게 인용

[3️⃣ 시험 변형 가능성]
- 족보 문제를 어떻게 변형할지 2가지

[4️⃣ 예상 문제]
- 객관식 1문항(족보 스타일 유지)
- 정답 + 해설(오답 이유 포함)

{subject_note}

과목 관점(필수 포함):
- {t["focus"][0]}
- {t["focus"][1]}
- {t["focus"][2]}
""".strip()

    if mode == "page":
        body = f"""
---
[강의 페이지 텍스트]
{lecture_text}

[관련 족보 발췌]
{jokbo_ctx}
""".strip()
    else:
        body = f"""
---
[실시간 입력 텍스트]
{lecture_text}

[관련 족보 발췌]
{jokbo_ctx}
""".strip()

    return base_rules + "\n" + body


# ==========================================
# 5. 응답 파서 + 섹션 렌더러 (서비스 UI화)
# ==========================================
def parse_ai_sections(text: str) -> dict:
    keys = ["족보에서 나온 형태", "강의 내용과의 연결", "시험 변형 가능성", "예상 문제"]
    sections = {k: "" for k in keys}
    current = None

    for raw in (text or "").splitlines():
        line = raw.strip()

        if "족보에서" in line and ("형태" in line or "나온" in line):
            current = "족보에서 나온 형태"
            continue
        if "강의" in line and ("연결" in line or "해당" in line):
            current = "강의 내용과의 연결"
            continue
        if "변형" in line:
            current = "시험 변형 가능성"
            continue
        if "예상" in line and "문제" in line:
            current = "예상 문제"
            continue

        if current:
            sections[current] += raw + "\n"

    return sections


def render_sections(sections: dict):
    st.markdown("### 🧾 족보 기반 분석 결과")

    st.subheader("1) 족보에서 나온 형태")
    st.markdown(sections.get("족보에서 나온 형태", "").strip() or "_(없음)_")

    st.subheader("2) 강의 내용과의 연결")
    st.markdown(sections.get("강의 내용과의 연결", "").strip() or "_(없음)_")

    st.subheader("3) 시험 변형 가능성")
    st.markdown(sections.get("시험 변형 가능성", "").strip() or "_(없음)_")

    st.subheader("4) 예상 문제")
    st.markdown(sections.get("예상 문제", "").strip() or "_(없음)_")


# ==========================================
# 6. 사이드바 (상태 / 설정)
# ==========================================
with st.sidebar:
    st.title("🩺 Med-Study 상태")

    api_key = st.text_input("Gemini API Key", type="password")

    if api_key:
        try:
            genai.configure(api_key=api_key)
            available_models = list_text_models(api_key)

            if not available_models:
                st.session_state.api_key_ok = False
                st.error("generateContent 가능한 모델이 없습니다.")
            else:
                st.session_state.api_key_ok = True
                st.session_state.text_models = available_models
                st.session_state.best_text_model = pick_best_text_model(available_models)
                st.success("AI 연결 완료")
                st.caption(f"사용 모델(자동): {st.session_state.best_text_model}")

        except Exception as e:
            st.session_state.api_key_ok = False
            st.error(f"모델 목록 조회 실패: {e}")

    st.divider()
    st.caption(f"📊 학습된 족보 페이지 수: **{len(st.session_state.db)}**")

    if st.button("족보 DB 초기화"):
        st.session_state.db = []
        st.rerun()


# ==========================================
# 7. 메인 UI
# ==========================================
tab1, tab2, tab3 = st.tabs(
    ["📂 족보 학습", "📖 강의 공부", "⌨️ 실시간 텍스트 분석"]
)

# ==================================================
# TAB 1 — 족보 학습
# ==================================================
with tab1:
    st.header("📂 족보 학습")
    st.info("과거 시험 족보를 학습시켜, 강의 내용과 시험 출제 포인트를 연결합니다.")

    files = st.file_uploader(
        "족보 PDF 업로드", type="pdf", accept_multiple_files=True
    )

    col_a, col_b = st.columns([1, 2])
    with col_a:
        max_pages = st.number_input(
            "파일당 최대 학습 페이지(데모용)",
            min_value=1, max_value=200, value=30, step=1
        )
    with col_b:
        st.caption("데모 안정성을 위해 파일당 학습 페이지를 제한하는 것을 권장합니다.")

    if st.button("📚 시험 대비 DB 구축 시작"):
        if not api_key or not st.session_state.api_key_ok:
            st.error("사이드바에서 유효한 API Key를 먼저 설정하세요.")
            st.stop()
        if not files:
            st.warning("족보 PDF를 업로드하세요.")
            st.stop()

        bar = st.progress(0)
        status = st.empty()
        new_db = []
        total_files = len(files)

        for i, f in enumerate(files):
            status.text(f"📖 파일 처리 중: {f.name}")
            pages = extract_text_from_pdf(f)[: int(max_pages)]

            for j, p in enumerate(pages):
                status.text(
                    f"🧠 DB 구축 중: {f.name} ({j+1}/{len(pages)} 페이지)"
                )
                emb = get_embedding(p["text"])
                if emb:
                    p["embedding"] = emb
                    new_db.append(p)
                time.sleep(0.8)  # 429 완화

            bar.progress((i + 1) / total_files)

        st.session_state.db.extend(new_db)
        status.text("✅ 학습 완료")
        st.success(f"총 {len(new_db)} 페이지 학습 완료")
        st.info("👉 다음: **강의 공부** 탭에서 강의 PDF를 열고 분석하세요.")


# ==================================================
# TAB 2 — 강의 공부 (족보 근거 → AI)
# ==================================================
with tab2:
    st.header("📖 강의 공부")
    st.info("강의 페이지 내용이 족보에서 어떻게 나왔는지만 확인합니다.")

    # 과목 선택 + 기타 입력
    c1, c2 = st.columns([1, 2])
    with c1:
        subject_choice = st.selectbox(
            "과목", ["해부학", "생리학", "약리학", "기타"], index=1
        )
    with c2:
        custom_subject = st.text_input(
            "기타 과목명", disabled=(subject_choice != "기타")
        )

    subject_final = resolve_subject(subject_choice, custom_subject)
    st.caption(f"현재 과목: **{subject_final}**")

    lec_file = st.file_uploader("강의록 PDF", type="pdf", key="lec")

    if lec_file:
        if (
            st.session_state.lecture_doc is None
            or st.session_state.lecture_filename != lec_file.name
        ):
            st.session_state.lecture_doc = fitz.open(
                stream=lec_file.read(), filetype="pdf"
            )
            st.session_state.lecture_filename = lec_file.name
            st.session_state.current_page = 0

        doc = st.session_state.lecture_doc
        col_view, col_ai = st.columns([6, 4])

        # ---------- PDF Viewer ----------
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
            st.image(
                Image.frombytes("RGB", [pix.width, pix.height], pix.samples),
                use_container_width=True
            )

            page_text = (page.get_text() or "").strip()
            if not page_text:
                st.warning("이 페이지에는 텍스트가 없습니다.")

        # ---------- AI Analyzer ----------
        with col_ai:
            st.subheader("🔎 족보 근거")

            if st.button("이 페이지 분석"):
                if not st.session_state.db:
                    st.error("족보 DB가 없습니다.")
                    st.stop()
                if not page_text:
                    st.error("분석할 텍스트가 없습니다.")
                    st.stop()

                related = find_relevant_jokbo(
                    page_text, st.session_state.db, top_k=3
                )

                # ✅ 족보 근거 없으면 AI 호출 스킵
                if not has_jokbo_evidence(related):
                    st.warning("📌 관련 족보 근거를 찾지 못했습니다. (AI 분석 생략)")
                    st.stop()

                # 1) 족보 근거 먼저 표시
                for i, r in enumerate(related):
                    with st.expander(
                        f"족보 근거 #{i+1} (유사도 {r['score']:.3f})"
                    ):
                        st.write(f"페이지 {r['content']['page']}")
                        st.write(r["content"]["text"])

                jokbo_ctx = "\n".join(
                    f"- (p{r['content']['page']}) {r['content']['text'][:300]}"
                    for r in related
                )

                prompt = build_exam_prompt(
                    subject=subject_final,
                    lecture_text=page_text,
                    jokbo_ctx=jokbo_ctx,
                    mode="page"
                )

                models = st.session_state.text_models or []
                fallback = models + ["models/gemini-1.5-flash-latest"]

                with st.spinner("족보 기반 분석 중..."):
                    result, used = generate_with_fallback(prompt, fallback)
                    st.caption(f"사용 모델: {used}")
                    sections = parse_ai_sections(result)
                    render_sections(sections)


# ==================================================
# TAB 3 — 실시간 텍스트 분석 (족보 근거 있을 때만)
# ==================================================
with tab3:
    st.header("⌨️ 실시간 텍스트 분석")
    st.info("족보에 근거가 있을 때만 시험 포인트로 변환합니다.")

    c1, c2 = st.columns([1, 2])
    with c1:
        subject_choice_live = st.selectbox(
            "과목", ["해부학", "생리학", "약리학", "기타"], index=1
        )
    with c2:
        custom_subject_live = st.text_input(
            "기타 과목명", disabled=(subject_choice_live != "기타")
        )

    subject_final_live = resolve_subject(
        subject_choice_live, custom_subject_live
    )
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

        related = find_relevant_jokbo(
            query, st.session_state.db, top_k=3
        )

        # ✅ 족보 근거 없으면 AI 호출 스킵
        if not has_jokbo_evidence(related):
            st.warning("📌 관련 족보 근거를 찾지 못했습니다. (AI 분석 생략)")
            st.stop()

        for i, r in enumerate(related):
            with st.expander(
                f"족보 근거 #{i+1} (유사도 {r['score']:.3f})"
            ):
                st.write(f"페이지 {r['content']['page']}")
                st.write(r["content"]["text"])

        jokbo_ctx = "\n".join(
            f"- (p{r['content']['page']}) {r['content']['text'][:300]}"
            for r in related
        )

        prompt = build_exam_prompt(
            subject=subject_final_live,
            lecture_text=query,
            jokbo_ctx=jokbo_ctx,
            mode="live"
        )

        models = st.session_state.text_models or []
        fallback = models + ["models/gemini-1.5-flash-latest"]

        with st.spinner("족보 기반 분석 중..."):
            result, used = generate_with_fallback(prompt, fallback)
            st.caption(f"사용 모델: {used}")
            sections = parse_ai_sections(result)
            render_sections(sections)
