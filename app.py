# app.py
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
st.caption("📌 흐름: (1) 족보 업로드→DB 구축  (2) 강의본 업로드→페이지 넘기며 옆에서 족보 근거 확인")

# ==========================================
# 1. 세션 상태 초기화
# ==========================================
if "db" not in st.session_state:
    st.session_state.db = []  # [{"page": int, "text": str, "source": str, "embedding": list[float]}]

if "api_key" not in st.session_state:
    st.session_state.api_key = None

if "api_key_ok" not in st.session_state:
    st.session_state.api_key_ok = False

if "text_models" not in st.session_state:
    st.session_state.text_models = []

if "best_text_model" not in st.session_state:
    st.session_state.best_text_model = None

if "lecture_doc" not in st.session_state:
    st.session_state.lecture_doc = None

if "lecture_filename" not in st.session_state:
    st.session_state.lecture_filename = None

if "current_page" not in st.session_state:
    st.session_state.current_page = 0

if "last_page_sig" not in st.session_state:
    st.session_state.last_page_sig = None  # 페이지 텍스트 hash로 중복 검색 방지

# ==========================================
# 2. 설정값
# ==========================================
JOKBO_THRESHOLD = 0.72  # 추천 0.70~0.75


def has_jokbo_evidence(related: list[dict]) -> bool:
    return bool(related) and related[0]["score"] >= JOKBO_THRESHOLD


# ==========================================
# 3. 유틸
# ==========================================
def ensure_configured():
    if st.session_state.get("api_key"):
        genai.configure(api_key=st.session_state["api_key"])


def extract_text_from_pdf(uploaded_file):
    """PDF를 텍스트로 변환 (fitz 사용)"""
    data = uploaded_file.getvalue()  # ✅ UploadedFile read() 재사용 이슈 방지
    doc = fitz.open(stream=data, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text() or ""
        if text.strip():
            pages.append({"page": i + 1, "text": text, "source": uploaded_file.name})
    return pages


def get_embedding(text: str):
    """임베딩 생성 (가능하면 text-embedding-004, 아니면 embedding-001)"""
    text = (text or "").strip()
    if not text:
        return []

    text = text[:12000]  # 데모 안정성
    ensure_configured()

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


def find_relevant_jokbo(query_text: str, db: list[dict], top_k: int = 5):
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
# 4. (옵션) AI 생성 – 필요할 때만
# ==========================================
@st.cache_data(show_spinner=False)
def list_text_models(api_key: str):
    genai.configure(api_key=api_key)
    models = genai.list_models()
    out = []
    for m in models:
        methods = getattr(m, "supported_generation_methods", []) or []
        if "generateContent" in methods:
            out.append(m.name)
    return out


def pick_best_text_model(model_names: list[str]):
    if not model_names:
        return None
    flash = [m for m in model_names if "flash" in m.lower()]
    return flash[0] if flash else model_names[0]


def generate_with_fallback(prompt: str, model_names: list[str]):
    ensure_configured()
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
    raise last_err


def build_simple_ai_prompt(lecture_text: str, jokbo_ctx: str):
    """수업 중 빠르게 '족보가 어떻게 나왔는지'만 간단 요약 (옵션)"""
    return f"""
너는 의대 시험 조교다.
아래 [관련 족보 발췌]를 근거로만, 강의 내용이 시험에서 어떤 식으로 나왔는지 짧게 정리하라.
추측 금지. 족보에 없는 내용 생성 금지.

형식:
- 한줄 요약(족보에서 어떤 포인트로 나왔는지)
- 키워드 5개
- 족보 문장 근거 2개 (짧게 인용)

[강의 페이지 텍스트]
{lecture_text}

[관련 족보 발췌]
{jokbo_ctx}
""".strip()


# ==========================================
# 5. UI 컴포넌트
# ==========================================
def render_jokbo_cards(related: list[dict]):
    st.markdown("### 📌 이 페이지와 유사한 족보 근거")
    if not related:
        st.info("족보 DB가 비어있거나, 텍스트 임베딩 생성에 실패했습니다.")
        return

    if not has_jokbo_evidence(related):
        st.warning("관련 족보 근거를 찾지 못했습니다.")
        return

    for i, r in enumerate(related):
        c = r["content"]
        score = r["score"]
        title = f"#{i+1}  유사도 {score:.3f} · {c.get('source','(unknown)')} · p{c.get('page','?')}"
        with st.container(border=True):
            st.markdown(f"**{title}**")
            snippet = (c.get("text") or "").strip().replace("\n", " ")
            st.write(snippet[:600] + ("…" if len(snippet) > 600 else ""))


# ==========================================
# 6. 사이드바
# ==========================================
with st.sidebar:
    st.title("🩺 Med-Study")

    api_key = st.text_input("Gemini API Key", type="password", key="api_key_input")

    if api_key:
        try:
            st.session_state.api_key = api_key
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
                st.caption(f"텍스트 모델(자동): {st.session_state.best_text_model}")
        except Exception as e:
            st.session_state.api_key_ok = False
            st.error(f"모델 목록 조회 실패: {e}")

    st.divider()
    st.caption(f"📚 학습된 족보 페이지 수: **{len(st.session_state.db)}**")

    colx, coly = st.columns(2)
    with colx:
        if st.button("족보 DB 초기화", key="reset_db_btn"):
            st.session_state.db = []
            st.session_state.last_page_sig = None
            st.rerun()
    with coly:
        st.caption(f"임계값: **{JOKBO_THRESHOLD:.2f}**")

# ==========================================
# 7. 메인 탭
# ==========================================
tab1, tab2 = st.tabs(["📂 1) 족보 업로드/학습", "📖 2) 강의본 보며 족보 확인"])

# ==================================================
# TAB 1 — 족보 업로드/학습
# ==================================================
with tab1:
    st.header("📂 1) 족보 업로드/학습")
    st.info("족보 PDF를 여러 개 올려서 페이지 단위로 임베딩 DB를 만들어둡니다.")

    files = st.file_uploader(
        "족보 PDF 업로드",
        type="pdf",
        accept_multiple_files=True,
        key="jokbo_pdf_uploader",
    )

    col_a, col_b = st.columns([1, 2])
    with col_a:
        max_pages = st.number_input(
            "파일당 최대 학습 페이지(데모용)",
            min_value=1,
            max_value=400,
            value=60,
            step=1,
            key="max_pages_input",
        )
    with col_b:
        st.caption("너무 많이 학습하면 임베딩 호출이 많아져 느려질 수 있어요. (데모는 30~80 추천)")

    if st.button("📚 족보 DB 구축 시작", key="build_db_btn"):
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
            if not pages:
                status.text(f"⚠️ 텍스트 추출 실패/빈 PDF: {f.name} (스캔본이면 OCR 필요)")
                bar.progress((i + 1) / total_files)
                continue

            for j, p in enumerate(pages):
                status.text(f"🧠 DB 구축: {f.name} ({j+1}/{len(pages)}p)")
                emb = get_embedding(p["text"])
                if emb:
                    p["embedding"] = emb
                    new_db.append(p)
                time.sleep(0.7)  # 429 완화(족보 구축 때만)

            bar.progress((i + 1) / total_files)

        st.session_state.db.extend(new_db)
        status.text("✅ 학습 완료")
        st.success(f"총 {len(new_db)} 페이지(텍스트 있는 페이지만) 학습 완료")
        st.info("👉 다음 탭에서 강의본을 올리고, 페이지 넘기면서 오른쪽에서 바로 족보 근거를 확인하세요.")


# ==================================================
# TAB 2 — 강의본 보며 족보 확인 (핵심)
# ==================================================
with tab2:
    st.header("📖 2) 강의본 보며 족보 확인")
    st.info("강의본을 페이지 넘기면, 오른쪽에 '족보가 어떻게 나왔는지' 근거가 자동으로 뜹니다.")

    if not st.session_state.db:
        st.warning("먼저 1번 탭에서 **족보 DB를 구축**하세요.")

    lec_file = st.file_uploader("강의본 PDF 업로드", type="pdf", key="lec_pdf_uploader")

    # 옵션: AI로 요약까지(느릴 수 있음)
    ai_toggle = st.toggle("옵션: AI로 '족보 포인트' 짧게 요약(느릴 수 있음)", value=False, key="ai_toggle")

    if lec_file:
        if st.session_state.lecture_doc is None or st.session_state.lecture_filename != lec_file.name:
            data = lec_file.getvalue()
            st.session_state.lecture_doc = fitz.open(stream=data, filetype="pdf")
            st.session_state.lecture_filename = lec_file.name
            st.session_state.current_page = 0
            st.session_state.last_page_sig = None  # 새 파일이면 캐시 리셋

        doc = st.session_state.lecture_doc
        col_view, col_right = st.columns([6, 4])

        # ---------- LEFT: PDF Viewer ----------
        with col_view:
            nav1, nav2, nav3 = st.columns([1, 2, 1])

            if nav1.button("◀", key="prev_page_btn"):
                if st.session_state.current_page > 0:
                    st.session_state.current_page -= 1

            nav2.markdown(
                f"<center><b>{st.session_state.current_page+1} / {len(doc)}</b></center>",
                unsafe_allow_html=True,
            )

            if nav3.button("▶", key="next_page_btn"):
                if st.session_state.current_page < len(doc) - 1:
                    st.session_state.current_page += 1

            page = doc.load_page(st.session_state.current_page)
            pix = page.get_pixmap(dpi=150)
            st.image(
                Image.frombytes("RGB", [pix.width, pix.height], pix.samples),
                use_container_width=True,
            )

            page_text = (page.get_text() or "").strip()
            if not page_text:
                st.warning("이 페이지에는 텍스트가 없습니다. (스캔 PDF면 OCR이 필요할 수 있어요)")

        # ---------- RIGHT: Jokbo matches (AUTO) ----------
        with col_right:
            if not st.session_state.db:
                st.info("족보 DB가 없어서 비교할 수 없습니다.")
                st.stop()

            # ✅ 자동 검색: 페이지 텍스트가 바뀌었을 때만
            sig = hash(page_text) if page_text else None

            if page_text and sig != st.session_state.last_page_sig:
                st.session_state.last_page_sig = sig
                related = find_relevant_jokbo(page_text, st.session_state.db, top_k=5)
                st.session_state["last_related"] = related
            else:
                related = st.session_state.get("last_related", [])

            render_jokbo_cards(related)

            # (옵션) AI 요약
            if ai_toggle and has_jokbo_evidence(related):
                if not api_key or not st.session_state.api_key_ok:
                    st.warning("AI 요약을 쓰려면 사이드바에 API Key를 넣어야 합니다.")
                else:
                    jokbo_ctx = "\n".join(
                        f"- ({r['content']['source']} p{r['content']['page']}) {r['content']['text'][:300]}"
                        for r in related[:3]
                    )
                    prompt = build_simple_ai_prompt(page_text, jokbo_ctx)
                    models = st.session_state.text_models or []
                    with st.spinner("AI 요약 중..."):
                        result, used = generate_with_fallback(prompt, models)
                        st.caption(f"사용 모델: {used}")
                        st.markdown("### 🧠 AI 요약(옵션)")
                        st.write(result)
    else:
        st.caption("강의본 PDF를 올리면, 왼쪽은 강의본/오른쪽은 족보 근거가 자동으로 표시됩니다.")
