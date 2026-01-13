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
아래 형식을 반드시 지켜라. 형식이 무너지면 답변 품질이 떨어진다.

[1️⃣ 핵심 개념]
- bullet 5개 (의대식 키워드, 불필요한 미사여구 금지)
- 가능하면 (한글 / 영어) 병기

[2️⃣ 족보 연결]
- 아래 중 하나로 분류하고 근거를 '구체적으로' 써라:
  (A) 족보 반복  (B) 족보 변형  (C) 족보에 없던 새로운 강조점
- 가능하면 "족보 p.번호"를 직접 언급

[3️⃣ 시험에 나오는 방식]
- {t["question_style"]}
- 학생들이 자주 헷갈리는 포인트 2개를 반드시 포함

[4️⃣ 예상 문제]
- 객관식 2문항 + 단답형 1문항
- 각 문항: 정답 + 해설(왜 다른 선택지는 틀렸는지도 포함)

[5️⃣ 암기 포인트]
- 시험 직전 외울 수 있는 5줄 요약(간결/정확)
- 모호한 표현 금지(“중요하다” 대신 “~와 구분”처럼)

{subject_note}

과목 관점(필수 포함 요소):
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
{jokbo_ctx if jokbo_ctx.strip() else "(관련 족보 없음)"}
""".strip()
    else:
        body = f"""
---
[교수님 발언/실시간 메모]
{lecture_text}

[관련 족보 발췌]
{jokbo_ctx if jokbo_ctx.strip() else "(관련 족보 없음)"}
""".strip()

    return base_rules + "\n" + body


# ==========================================
# 5. 응답 파서 + 섹션 렌더러 (서비스 UI화)
# ==========================================
def parse_ai_sections(text: str) -> dict:
    keys = ["핵심 개념", "족보 연결", "시험에 나오는 방식", "예상 문제", "암기 포인트"]
    sections = {k: "" for k in keys}
    current = None

    for raw in (text or "").splitlines():
        line = raw.strip()

        # 느슨한 헤더 감지
        if "핵심" in line and "개념" in line:
            current = "핵심 개념"
            continue
        if "족보" in line and ("연결" in line or "관련" in line):
            current = "족보 연결"
            continue
        if "시험" in line and ("방식" in line or "출제" in line):
            current = "시험에 나오는 방식"
            continue
        if "예상" in line and "문제" in line:
            current = "예상 문제"
            continue
        if "암기" in line and ("포인트" in line or "요약" in line):
            current = "암기 포인트"
            continue

        if current:
            sections[current] += raw + "\n"

    return sections


def render_sections(sections: dict, show_title: bool = True):
    if show_title:
        st.markdown("### 🔍 분석 결과")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🔑 핵심 개념")
        st.markdown(sections.get("핵심 개념", "").strip() or "_(없음)_")
    with c2:
        st.subheader("🧩 족보 연결")
        st.markdown(sections.get("족보 연결", "").strip() or "_(없음)_")

    st.subheader("📌 시험에 나오는 방식")
    st.markdown(sections.get("시험에 나오는 방식", "").strip() or "_(없음)_")

    st.subheader("📝 예상 문제")
    st.markdown(sections.get("예상 문제", "").strip() or "_(없음)_")

    st.subheader("🧠 암기 포인트")
    st.markdown(sections.get("암기 포인트", "").strip() or "_(없음)_")


# ==========================================
# 6. 사이드바 (상태)
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
                st.caption(f"✅ 텍스트 모델: {st.session_state.best_text_model}")
        except Exception as e:
            st.session_state.api_key_ok = False
            st.error(f"모델 목록 조회 실패: {e}")

    st.divider()
    st.caption(
        f"""
📊 시스템 현황  
- 학습된 족보 페이지: **{len(st.session_state.db)}**
"""
    )
    if st.session_state.best_text_model:
        st.caption(f"- 사용 모델(자동): **{st.session_state.best_text_model}**")

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
    st.info("과거 시험 족보를 학습시켜, 강의 내용과 시험 출제 포인트를 자동 연결합니다.")

    files = st.file_uploader("족보 PDF 업로드", type="pdf", accept_multiple_files=True)

    col_a, col_b = st.columns([1, 2])
    with col_a:
        max_pages = st.number_input("파일당 최대 학습 페이지(데모용)", min_value=1, max_value=200, value=30, step=1)
    with col_b:
        st.caption("데모 안정성을 위해 파일당 학습 페이지를 제한하는 걸 추천해.")

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
            status.text(f"📖 파일 읽는 중: {f.name}")
            pages = extract_text_from_pdf(f)[: int(max_pages)]

            for j, p in enumerate(pages):
                status.text(f"🧠 시험 대비 DB 구축 중: {f.name} ({j+1}/{len(pages)} 페이지)")
                emb = get_embedding(p["text"])
                if emb:
                    p["embedding"] = emb
                    new_db.append(p)
                time.sleep(0.8)  # 429 완화

            bar.progress((i + 1) / total_files)

        st.session_state.db.extend(new_db)
        status.text("✅ 학습 완료!")
        st.success(f"총 {len(new_db)} 페이지 학습 완료")
        st.info("다음 단계 👉 **강의 공부** 탭에서 강의 PDF를 열고 '이 페이지 분석'을 눌러보세요.")
# ==================================================
# TAB 2 — 강의 공부 (페이지 뷰어 + 분석)
# ==================================================
with tab2:
    st.header("📖 강의 공부")
    st.info("강의 페이지를 한 장씩 보면서, 족보와 연결해 ‘시험에 어떻게 나올지’까지 바로 뽑아줍니다.")

    # 과목 선택 + 기타 입력
    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        subject_choice = st.selectbox("과목 선택", ["해부학", "생리학", "약리학", "기타"], index=1, key="subject_tab2")
    with col_s2:
        custom_subject = ""
        if subject_choice == "기타":
            custom_subject = st.text_input("기타 과목명 입력 (예: 병리학, 생화학, 면역학)", key="custom_subject_tab2")

    subject_final = resolve_subject(subject_choice, custom_subject)
    st.caption(f"✅ 현재 과목: **{subject_final}**")

    lec_file = st.file_uploader("강의록 PDF 업로드", type="pdf", key="lecture_pdf")

    if lec_file:
        # 새 파일이면 문서 새로 열기
        if (
            st.session_state.lecture_doc is None
            or st.session_state.lecture_filename != lec_file.name
        ):
            st.session_state.lecture_doc = fitz.open(stream=lec_file.read(), filetype="pdf")
            st.session_state.lecture_filename = lec_file.name
            st.session_state.current_page = 0

        doc = st.session_state.lecture_doc
        col_view, col_ai = st.columns([6, 4])

        # ---------- PDF Viewer ----------
        with col_view:
            nav1, nav2, nav3 = st.columns([1, 2, 1])

            if nav1.button("◀ 이전", key="prev_page"):
                if st.session_state.current_page > 0:
                    st.session_state.current_page -= 1

            nav2.markdown(
                f"<center>{st.session_state.current_page + 1} / {len(doc)}</center>",
                unsafe_allow_html=True,
            )

            if nav3.button("다음 ▶", key="next_page"):
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
                st.warning("이 페이지에는 텍스트가 없습니다. (스캔 이미지 PDF일 수 있음)")

        # ---------- AI Analyzer ----------
        with col_ai:
            st.subheader("🧠 시험 대비 AI 분석")

            # 작은 상태 안내
            if not st.session_state.db:
                st.warning("먼저 '족보 학습' 탭에서 족보를 학습하세요.")
            if not api_key or not st.session_state.api_key_ok:
                st.warning("사이드바에서 API Key 상태를 확인하세요.")

            if st.button("⚡ 이 페이지 분석", key="analyze_page"):
                if not api_key or not st.session_state.api_key_ok:
                    st.error("유효한 API Key가 필요합니다.")
                    st.stop()
                if not st.session_state.db:
                    st.error("족보 DB가 없습니다. 먼저 족보를 학습하세요.")
                    st.stop()
                if not page_text:
                    st.error("이 페이지에는 텍스트가 없어 분석할 수 없습니다.")
                    st.stop()

                with st.spinner("족보 연결 + 시험 포인트 분석 중..."):
                    # 1) 관련 족보 검색
                    related = find_relevant_jokbo(page_text, st.session_state.db, top_k=3)
                    jokbo_ctx = "\n".join([
                        f"- (p{r['content']['page']}) {r['content']['text'][:220]}"
                        for r in related
                    ])

                    # 2) 과목/기타 반영 프롬프트 생성
                    prompt = build_exam_prompt(
                        subject=subject_final,
                        lecture_text=page_text,
                        jokbo_ctx=jokbo_ctx,
                        mode="page"
                    )

                    # 3) 모델 후보 구성
                    model_list = st.session_state.text_models or []
                    fallback_candidates = model_list + [
                        "models/gemini-1.5-flash-latest",
                        "models/gemini-1.5-pro-latest"
                    ]

                    # 4) 생성 + 파싱 + 렌더링
                    try:
                        result_text, used = generate_with_fallback(prompt, fallback_candidates)
                        st.caption(f"사용 모델: {used}")
                        sections = parse_ai_sections(result_text)
                        render_sections(sections)
                    except Exception as e:
                        msg = str(e)
                        if "429" in msg:
                            st.error("⚠️ 사용량(429) 제한입니다. 잠시 후 다시 시도하세요.")
                        else:
                            st.error(f"분석 실패: {e}")


# ==================================================
# TAB 3 — 실시간 텍스트 분석 (교수님 발언 메모 → 시험 포인트 변환)
# ==================================================
with tab3:
    st.header("⌨️ 실시간 텍스트 분석")
    st.info("강의 중 교수님이 강조한 문장을 입력하면, 족보와 연결해 시험 출제 포인트로 변환합니다.")

    # 과목 선택 + 기타 입력
    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        subject_choice_live = st.selectbox("과목 선택", ["해부학", "생리학", "약리학", "기타"], index=1, key="subject_tab3")
    with col_t2:
        custom_subject_live = ""
        if subject_choice_live == "기타":
            custom_subject_live = st.text_input("기타 과목명 입력 (예: 병리학, 생화학, 면역학)", key="custom_subject_tab3")

    subject_final_live = resolve_subject(subject_choice_live, custom_subject_live)
    st.caption(f"✅ 현재 과목: **{subject_final_live}**")

    if not api_key or not st.session_state.api_key_ok:
        st.warning("사이드바에서 API Key를 먼저 설정하세요.")
    if not st.session_state.db:
        st.warning("족보 DB가 없습니다. 먼저 족보를 학습하세요.")

    user_input = st.text_area(
        "🚨 교수님이 '중요하다/시험에 낼 수 있다'고 말한 내용을 그대로 입력하세요",
        height=160,
        placeholder="예) 이 단계는 rate-limiting step이라 시험에 자주 나온다…"
    )

    if st.button("📊 족보 매칭 & 시험 인사이트 생성", key="live_analyze"):
        if not api_key or not st.session_state.api_key_ok:
            st.error("유효한 API Key가 필요합니다.")
            st.stop()
        if not st.session_state.db:
            st.error("족보 DB가 없습니다. 먼저 족보를 학습하세요.")
            st.stop()

        query = (user_input or "").strip()
        if not query:
            st.error("분석할 텍스트를 입력하세요.")
            st.stop()

        with st.spinner("족보 연결 중..."):
            related = find_relevant_jokbo(query, st.session_state.db, top_k=3)

        st.subheader("🔎 족보와의 연결")
        context_str = ""
        if not related:
            st.write("→ 관련 족보를 찾지 못했습니다. (새로운 강조점일 수 있음)")
        else:
            for i, r in enumerate(related):
                with st.expander(f"관련 족보 #{i+1} (유사도 {r['score']:.3f})"):
                    st.write(f"페이지 {r['content']['page']}")
                    st.write(r["content"]["text"])
                context_str += f"- (p{r['content']['page']}) {r['content']['text']}\n"

        st.divider()

        # 과목/기타 반영 프롬프트 생성
        final_prompt = build_exam_prompt(
            subject=subject_final_live,
            lecture_text=query,
            jokbo_ctx=context_str,
            mode="live"
        )

        model_list = st.session_state.text_models or []
        fallback_candidates = model_list + [
            "models/gemini-1.5-flash-latest",
            "models/gemini-1.5-pro-latest"
        ]

        st.subheader("🩺 Med-Study 시험 인사이트")
        with st.spinner("AI가 시험 포인트로 변환 중..."):
            try:
                result_text, used = generate_with_fallback(final_prompt, fallback_candidates)
                st.caption(f"사용 모델: {used}")
                sections = parse_ai_sections(result_text)
                render_sections(sections)
            except Exception as e:
                msg = str(e)
                if "429" in msg:
                    st.error("⚠️ 사용량(429) 제한입니다. 잠시 후 다시 시도하세요.")
                else:
                    st.error(f"분석 실패: {e}")
