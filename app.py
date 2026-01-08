import streamlit as st
import google.generativeai as genai
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

# ==========================================
# 1. 설정 및 초기화
# ==========================================
st.set_page_config(page_title="Med-Study OS Fixed", layout="wide", page_icon="🩺")

# 세션 상태 초기화 (새로고침 해도 데이터 유지)
if 'db' not in st.session_state: 
    st.session_state.db = []
if 'lecture_doc' not in st.session_state: 
    st.session_state.lecture_doc = None
if 'current_page' not in st.session_state: 
    st.session_state.current_page = 0
    # ==========================================
# 2. 핵심 함수 (Logic)
# ==========================================

def get_best_model():
    """사용 가능한 Gemini 모델 자동 탐색"""
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # 1순위: Flash (빠름), 2순위: Pro (성능)
        for m in models:
            if 'flash' in m.lower(): return m
        for m in models:
            if 'pro' in m.lower(): return m
        return models[0] if models else None
    except Exception:
        return None

def extract_text_from_pdf(file):
    """PDF를 텍스트로 변환"""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    pages_content = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages_content.append({"page": page_num + 1, "text": text, "source": file.name})
    return pages_content

def get_embedding(text):
    """임베딩 (Embedding-004 우선 사용)"""
    try:
        return genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )['embedding']
    except Exception:
        try:
            # 실패 시 구형 모델 시도
            return genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )['embedding']
        except:
            return []

def find_relevant_jokbo(query_text, db, top_k=3):
    """유사도 검색"""
    if not db: return []
    query_emb = get_embedding(query_text)
    if not query_emb: return []
    
    db_embs = [item['embedding'] for item in db]
    sims = cosine_similarity([query_emb], db_embs)[0]
    top_idxs = np.argsort(sims)[::-1][:top_k]
    
    return [{"score": sims[i], "content": db[i]} for i in top_idxs]
    # ==========================================
# 3. 사이드바 & 메인 UI
# ==========================================
with st.sidebar:
    st.title("⚙️ 설정")
    api_key = st.text_input("Gemini API Key", type="password")
    if api_key:
        genai.configure(api_key=api_key)
        model_name = get_best_model()
        if model_name:
            st.success(f"연결됨: {model_name.split('/')[-1]}")
        else:
            st.error("API Key 확인 필요")
            
    st.divider()
    st.write(f"📚 학습된 족보: {len(st.session_state.db)} 페이지")
    if st.button("초기화"):
        st.session_state.db = []
        st.rerun()

tab1, tab2 = st.tabs(["📂 족보 학습", "📖 강의 공부"])

# --- TAB 1: 족보 학습 ---
with tab1:
    st.header("1. 족보 업로드")
    files = st.file_uploader("족보 PDF", accept_multiple_files=True, type="pdf")
    
    if st.button("학습 시작 🚀") and files:
        if not api_key:
            st.error("API Key를 입력하세요.")
        else:
            bar = st.progress(0)
            new_db = []
            for i, f in enumerate(files):
                pages = extract_text_from_pdf(f)
                for p in pages:
                    emb = get_embedding(p['text'])
                    if emb:
                        p['embedding'] = emb
                        new_db.append(p)
                bar.progress((i + 1) / len(files))
            
            st.session_state.db.extend(new_db)
            st.success(f"{len(new_db)} 페이지 학습 완료!")

# --- TAB 2: 강의 분석 ---
with tab2:
    st.header("2. 강의 뷰어 & AI")
    lec_file = st.file_uploader("강의록 PDF", type="pdf", key="lec")
    
    if lec_file:
        if st.session_state.lecture_doc is None or st.session_state.lecture_doc.name != lec_file.name:
            st.session_state.lecture_doc = fitz.open(stream=lec_file.read(), filetype="pdf")
            st.session_state.current_page = 0
            
        doc = st.session_state.lecture_doc
        col_view, col_ai = st.columns([6, 4])
        
        with col_view:
            c1, c2, c3 = st.columns([1, 2, 1])
            if c1.button("◀"): 
                if st.session_state.current_page > 0: st.session_state.current_page -= 1
            c2.markdown(f"<center>{st.session_state.current_page + 1} / {len(doc)}</center>", unsafe_allow_html=True)
            if c3.button("▶"): 
                if st.session_state.current_page < len(doc) - 1: st.session_state.current_page += 1
            
            page = doc.load_page(st.session_state.current_page)
            pix = page.get_pixmap(dpi=150)
            st.image(Image.frombytes("RGB", [pix.width, pix.height], pix.samples), use_container_width=True)
            curr_text = page.get_text()

        with col_ai:
            if st.button("분석하기 ⚡"):
                if not api_key or not st.session_state.db:
                    st.error("API Key 또는 족보 데이터가 없습니다.")
                else:
                    with st.spinner("분석 중..."):
                        try:
                            related = find_relevant_jokbo(curr_text, st.session_state.db)
                            
                            # 검색된 족보 텍스트 정리
                            ctx_list = []
                            for item in related:
                                info = f"- {item['content']['source']} ({item['score']:.2f}): {item['content']['text'][:100]}..."
                                ctx_list.append(info)
                            ctx_str = "\n".join(ctx_list)
                            
                            # 프롬프트 구성 (들여쓰기 오류 방지를 위해 단순 문자열 사용)
                            prompt_text = "당신은 의대생 튜터입니다.\n"
                            prompt_text += f"[현재 강의]: {curr_text}\n"
                            prompt_text += f"[관련 족보]: {ctx_str}\n\n"
                            prompt_text += "요청:\n1. 강의와 족보의 연관성 요약\n2. 핵심 키워드 3개\n3. 예상 객관식 문제 1개"

                            model = genai.GenerativeModel(get_best_model())
                            res = model.generate_content(prompt_text)
                            st.markdown(res.text)
                        except Exception as e:
                            st.error(f"Error: {e}")

