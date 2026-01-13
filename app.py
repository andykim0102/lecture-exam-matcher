import os
import time
import tempfile

import streamlit as st
import google.generativeai as genai
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 0. 페이지 설정
# ==========================================
st.set_page_config(page_title="Med-Study OS Final", layout="wide", page_icon="🩺")

# (선택) 디버깅용: SDK 버전 표시
st.write(f"현재 google-generativeai 버전: {getattr(genai, '__version__', 'unknown')}")

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
# 2. 핵심 함수
# ==========================================
def extract_text_from_pdf(file) -> list[dict]:
    """PDF를 텍스트로 변환 (fitz 사용)"""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    pages_content = []
    for page_num, page in enumerate(doc):
        text = page.get_text() or ""
        if text.strip():
            pages_content.append(
                {"page": page_num + 1, "text": text, "source": file.name}
            )
    return pages_content


def get_embedding(text: str):
    """임베딩 생성 (가능하면 text-embedding-004, 아니면 embedding-001)"""
    text = (text or "").strip()
    if not text:
        return []

    # 데모 안정성: 과도한 길이 컷
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


@st.cache_data(show_spinner=False)
def list_text_models(api_key: str):
    """현재 키에서 generateContent 가능한 모델 목록을 가져옴"""
    genai.configure(api_key=api_key)
    models = genai.list_models()

    out = []
    for m in models:
        methods = getattr(m, "supported_generation_methods", []) or []
        if "generateContent" in methods:
            out.append(m.name)  # 보통 "models/..." 형태
    return out


def pick_best_text_model(model_names: list[str]):
    """flash 계열 우선 선택"""
    if not model_names:
        return None
    flash = [m for m in model_names if "flash" in m.lower()]
    return flash[0] if flash else model_names[0]


def generate_with_fallback(prompt: str, model_names: list[str]):
    """모델 후보를 순서대로 시도해서 성공하면 반환"""
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
# 3. 사이드바
# ==========================================
with st.sidebar:
    st.title("⚙️ 설정")

    api_key = st.text_input("Gemini API Key", type="password")

    if api_key:
        try:
            genai.configure(api_key=api_key)
            available_models = list_text_models(api_key)

            if not available_models:
                st.session_state.api_key_ok = False
                st.error("generateContent 가능한 모델이 없습니다. (키/프로젝트 권한 문제 가능)")
            else:
                st.session_state.api_key_ok = True
                st.session_state.text_models = available_models
                st.session_state.best_text_model = pick_best_text_model(available_models)
                st.caption(f"✅ 텍스트 모델 자동 선택: {st.session_state.best_text_model}")

        except Exception as e:
            st.session_state.api_key_ok = False
            st.error(f"모델 목록 조회 실패: {e}")

    st.divider()
    st.write(f"📚 학습된 족보: {len(st.session_state.db)} 페이지")

    if st.button("초기화"):
        st.session_state.db = []
        st.rerun()


# ==========================================
# 4. 메인 UI
# ==========================================
tab1, tab2, tab3 = st.tabs(["📂 족보 학습", "📖 강의 공부", "⌨️ 실시간 텍스트 분석"])


# --------------------------
# TAB 1: 족보 학습
# --------------------------
with tab1:
    st.header("1. 족보 업로드")
    files = st.file_uploader("족보 PDF", accept_multiple_files=True, type="pdf")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        max_pages_per_file = st.number_input(
            "파일당 최대 학습 페이지(데모용)",
            min_value=1,
            max_value=200,
            value=30,
            step=1,
        )
    with col_b:
        st.caption("데모 안정성을 위해 파일당 학습 페이지 수를 제한하는 걸 추천해.")

    if st.button("학습 시작 🚀") and files:
        if not api_key:
            st.error("API Key를 입력하세요.")
            st.stop()

        bar = st.progress(0)
        status = st.empty()
        new_db = []
        total_files = len(files)

        for i, f in enumerate(files):
            status.text(f"📖 파일 읽는 중: {f.name}...")
            pages = extract_text_from_pdf(f)

            # 데모용 페이지 제한
            pages = pages[: int(max_pages_per_file)]

            for j, p in enumerate(pages):
                status.text(f"🧠 임베딩 중: {f.name} ({j+1}/{len(pages)} 페이지)...")
                emb = get_embedding(p["text"])
                if emb:
                    p["embedding"] = emb
                    new_db.append(p)

                # 속도 제한 완화(429 방지)
                time.sleep(0.8)

            bar.progress((i + 1) / total_files)

        st.session_state.db.extend(new_db)
        status.text("✅ 학습 완료!")
        st.success(f"{len(new_db)} 페이지 학습 완료!")


# --------------------------
# TAB 2: 강의 뷰어 & AI
# --------------------------
with tab2:
    st.header("2. 강의 뷰어 & AI")
    lec_file = st.file_uploader("강의록 PDF", type="pdf", key="lec")

    if lec_file:
        # 새 파일이면 문서 새로 열기
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

        with col_view:
            c1, c2, c3 = st.columns([1, 2, 1])

            if c1.button("◀"):
                if st.session_state.current_page > 0:
                    st.session_state.current_page -= 1

            c2.markdown(
                f"<center>{st.session_state.current_page + 1} / {len(doc)}</center>",
                unsafe_allow_html=True,
            )

            if c3.button("▶"):
                if st.session_state.current_page < len(doc) - 1:
                    st.session_state.current_page += 1

            page = doc.load_page(st.session_state.current_page)
            pix = page.get_pixmap(dpi=150)
            st.image(
                Image.frombytes("RGB", [pix.width, pix.height], pix.samples),
                use_container_width=True,
            )
            curr_text = (page.get_text() or "").strip()

        with col_ai:
            st.subheader("AI 분석")
            if st.button("분석하기 ⚡", key="analyze_page"):
                if not api_key or not st.session_state.api_key_ok:
                    st.error("API Key가 유효하지 않습니다(사이드바 확인).")
                    st.stop()

                if not st.session_state.db:
                    st.error("족보 데이터가 없습니다. 먼저 '족보 학습' 탭에서 학습하세요.")
                    st.stop()

                if not curr_text:
                    st.warning("텍스트가 없는 페이지입니다(스캔본 이미지일 수 있음).")
                    st.stop()

                with st.spinner("AI가 분석 중입니다..."):
                    try:
                        related = find_relevant_jokbo(curr_text, st.session_state.db, top_k=3)
                        ctx_str = "\n".join(
                            [f"- (p{item['content']['page']}) {item['content']['text'][:200]}" for item in related]
                        )

                        prompt = f"""
너는 의대 시험 대비 조교야.

[강의 페이지 텍스트]
{curr_text}

[관련 족보 발췌]
{ctx_str if ctx_str.strip() else "(관련 족보를 찾지 못함)"}

미션:
1) 이 페이지 핵심 개념 5개를 뽑아줘.
2) 족보와의 연결점을 '구체적으로' 말해줘(가능하면 페이지 번호 언급).
3) 예상 문제 3개(객관식 2 + 단답형 1) 만들어줘.
4) 각 문제의 정답/해설까지 써줘.
""".strip()

                        model_list = st.session_state.text_models or []
                        fallback_candidates = model_list + [
                            "models/gemini-1.5-flash-latest",
                            "models/gemini-1.5-pro-latest",
                        ]

                        text, used = generate_with_fallback(prompt, fallback_candidates)
                        st.caption(f"사용 모델: {used}")
                        st.markdown(text)

                    except Exception as e:
                        msg = str(e)
                        if "429" in msg:
                            st.error("⚠️ 사용량이 많습니다. 잠시 후 다시 시도해주세요.")
                        else:
                            st.error(f"에러 발생: {e}")


# --------------------------
# TAB 3: 실시간 텍스트 분석 (마이크 제거)
# --------------------------
with tab3:
    st.header("3. 실시간 텍스트 분석 (안정 버전)")
    st.info("강의 중 중요하다고 느낀 교수님 말을 그대로 입력하면, 족보와 연결해 분석합니다.")

    if not api_key or not st.session_state.api_key_ok:
        st.warning("먼저 사이드바에서 API Key를 입력/확인하세요.")
        st.stop()

    if not st.session_state.db:
        st.warning("먼저 '족보 학습' 탭에서 족보를 학습시켜주세요.")
        st.stop()

    user_text = st.text_area(
        "교수님 말씀 / 중요한 설명을 그대로 입력하세요",
        height=160,
        placeholder="예) 이 부분은 교과서에는 없지만 시험에 나올 수 있다...",
    )

    if st.button("족보 매칭 & 인사이트 생성", key="live_analyze"):
        query = (user_text or "").strip()
        if not query:
            st.error("분석할 텍스트를 입력하세요.")
            st.stop()

        with st.spinner("족보 뒤지는 중..."):
            related = find_relevant_jokbo(query, st.session_state.db, top_k=3)

        st.subheader("🔎 관련 족보")
        context_str = ""
        if not related:
            st.write("관련된 족보를 찾지 못했습니다. (새로운 강조점일 수 있음)")
        else:
            for idx, item in enumerate(related):
                with st.expander(f"관련 족보 #{idx+1} (유사도 {item['score']:.3f})"):
                    st.write(f"페이지 {item['content']['page']}")
                    st.write(item["content"]["text"])
                context_str += f"- (페이지 {item['content']['page']}) {item['content']['text']}\n"

        st.divider()
        st.subheader("🩺 Med-Study AI 분석")

        final_prompt = f"""
상황: 의대 강의 중 실시간 시험 대비 정리.

교수님 말씀:
{query}

관련 족보 발췌:
{context_str if context_str else "(관련 족보 없음)"}

미션:
1. 교수님 말씀이 족보의 어떤 부분과 연결되는지 분석.
2. 시험에 나올 가능성이 높은 포인트를 명확히 지적.
3. 예상 문제 3개 + 정답/해설.
4. 한눈에 외울 수 있는 암기 포인트 5줄.
""".strip()

        model_list = st.session_state.text_models or []
        fallback_candidates = model_list + [
            "models/gemini-1.5-flash-latest",
            "models/gemini-1.5-pro-latest",
        ]

        with st.spinner("AI 분석 중..."):
            try:
                text, used = generate_with_fallback(final_prompt, fallback_candidates)
                st.caption(f"사용 모델: {used}")
                st.markdown(text)
            except Exception as e:
                msg = str(e)
                if "429" in msg:
                    st.error("⚠️ 사용량이 많습니다. 잠시 후 다시 시도해주세요.")
                else:
                    st.error(f"에러 발생: {e}")
