import streamlit as st
import google.generativeai as genai
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. 설정 및 초기화
# ==========================================
st.set_page_config(page_title="Med-Study OS Fixed", layout="wide", page_icon="🧬")

# 세션 상태 초기화 (새로고침 해도 데이터 유지)
if 'db' not in st.session_state: st.session_state.db = []  # 족보 데이터 저장소
if 'lecture_doc' not in st.session_state: st.session_state.lecture_doc = None
if 'current_page' not in st.session_state: st.session_state.current_page = 0

# ==========================================
# 2. 핵심 함수 (Logic)
# ==========================================

def get_best_model():
    """
    현재 API Key로 사용 가능한 모델 중 가장 적합한 모델을 자동으로 찾습니다.
    (NotFound Error 방지용)
    """
    try:
        # 생성(generateContent)이 가능한 모델 목록 조회
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # 1순위: Flash (빠름), 2순위: Pro (똑똑함), 3순위: 아무거나
        for m in models:
            if 'flash' in m.lower(): return m
        for m in models:
            if 'pro' in m.lower(): return m
        
        return models[0] if models else None
    except Exception as e:
        st.error(f"모델 목록 조회 실패: {e}")
        return None

def extract_text_from_pdf(file):
    """PDF를 페이지별 텍스트로 변환"""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    pages_content = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():  # 빈 페이지 제외
            pages_content.append({
                "page": page_num + 1,
                "text": text,
                "source": file.name
            })
    return pages_content

def get_embedding(text):
    """텍스트를 벡터로 변환 (최신 모델 사용)"""
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception:
        # 구형 모델 폴백(Fallback)
        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            st.error(f"임베딩 오류: {e}")
            return []

def find_relevant_jokbo(query_text, db, top_k=3):
    """현재 강의 내용과 가장 유사한 족보 내용 검색"""
    if not db: return []
    
    query_embedding = get_embedding(query_text)
    if not query_embedding: return []

    db_embeddings = [item['embedding'] for item in db]
    
    # 코사인 유사도 계산
    similarities = cosine_similarity([query_embedding], db_embeddings)[0]
    
    # 상위 K개 추출
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            "score": similarities[idx],
            "content": db[idx]
        })
    return results

# ==========================================
# 3. 사이드바 (API 설정)
# ==========================================
with st.sidebar:
    st.title("⚙️ 시스템 설정")
    api_key = st.text_input("Gemini API Key", type="password")
    
    if api_key:
        genai.configure(api_key=api_key)
        
        # 연결 테스트 및 모델 확인
        try:
            model_name = get_best_model()
            if model_name:
                st.success(f"✅ 연결 성공! \n사용 모델: {model_name.split('/')[-1]}")
            else:
                st.error("❌ 사용 가능한 모델이 없습니다.")
        except:
            st.error("⚠️ API Key가 올바르지 않습니다.")
    
    st.divider()
    st.write(f"📊 학습된 족보 데이터: {len(st.session_state.db)} 페이지")
    if st.button("데이터 초기화"):
        st.session_state.db = []
        st.experimental_rerun()

# ==========================================
# 4. 메인 UI
# ==========================================
tab1, tab2 = st.tabs(["📂 1. 족보 학습 (Knowledge Base)", "📖 2. 강의 학습 (Study Mode)"])

# --- TAB 1: 족보 데이터 구축 ---
with tab1:
    st.header("1. 족보(기출문제) 업로드")
    st.caption("AI가 참고할 '지식 베이스'를 만드는 단계입니다.")
    
    uploaded_jokbo = st.file_uploader("족보 PDF 파일들을 드래그하세요", accept_multiple_files=True, type="pdf")
    
    if st.button("족보 학습 시작 🚀"):
        if not api_key:
            st.warning("왼쪽 사이드바에서 API Key를 먼저 입력해주세요.")
        elif not uploaded_jokbo:
            st.warning("파일을 업로드해주세요.")
        else:
            progress_bar = st.progress(0)
            status = st.empty()
            
            new_db = []
            total_files = len(uploaded_jokbo)
            
            for idx, file in enumerate(uploaded_jokbo):
                status.text(f"📖 Reading: {file.name}...")
                pages = extract_text_from_pdf(file)
                
                status.text(f"🧠 Embedding: {file.name} ({len(pages)} pages)...")
                for p in pages:
                    emb = get_embedding(p['text'])
                    if emb:
                        p['embedding'] = emb
                        new_db.append(p)
                
                progress_bar.progress((idx + 1) / total_files)
            
            st.session_state.db.extend(new_db)
            status.text("✅ 학습 완료!")
            st.success(f"총 {len(new_db)} 페이지가 지식 베이스에 추가되었습니다.")

# --- TAB 2: 강의 뷰어 및 분석 ---
with tab2:
    st.header("2. 강의록 뷰어 & AI 튜터")
    
    lecture_file = st.file_uploader("오늘 공부할 강의록 PDF", type="pdf", key="lecture")
    
    if lecture_file:
        # 파일 로드 (세션 최적화)
        if st.session_state.lecture_doc is None or st.session_state.lecture_doc.name != lecture_file.name:
            st.session_state.lecture_doc = fitz.open(stream=lecture_file.read(), filetype="pdf")
            st.session_state.current_page = 0
            
        doc = st.session_state.lecture_doc
        total_pages = len(doc)
        
        # 화면 분할 (좌: 뷰어, 우: AI)
        col_view, col_ai = st.columns([6, 4])
        
        with col_view:
            st.markdown("#### 📄 PDF Viewer")
            # 페이지 컨트롤
            c1, c2, c3 = st.columns([1, 2, 1])
            if c1.button("◀ 이전"):
                if st.session_state.current_page > 0: st.session_state.current_page -= 1
            c2.markdown(f"<div style='text-align: center;'>Page {st.session_state.current_page + 1} / {total_pages}</div>", unsafe_allow_html=True)
            if c3.button("다음 ▶"):
                if st.session_state.current_page < total_pages - 1: st.session_state.current_page += 1
            
            # PDF 렌더링
            page = doc.load_page(st.session_state.current_page)
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            st.image(img, use_container_width=True)
            
            current_text = page.get_text()

        with col_ai:
            st.markdown("#### 🤖 AI Analysis")
            
            if st.button("이 페이지 분석하기 ⚡", type="primary"):
                if not api_key:
                    st.error("API Key가 필요합니다.")
                elif len(st.session_state.db) == 0:
                    st.error("경고: 학습된 족보 데이터가 없습니다. (Tab 1에서 학습 필요)")
                elif not current_text.strip():
                    st.warning("텍스트를 인식할 수 없는 페이지입니다. (이미지 위주)")
                else:
                    with st.spinner("족보와 연결고리를 찾는 중..."):
                        try:
                            # 1. 모델 자동 선택
                            target_model = get_best_model()
                            if not target_model:
                                raise Exception("사용 가능한 모델 없음")

                            # 2. 관련 족보 검색
                            related = find_relevant_jokbo(current_text, st.session_state.db)
                            
                            # 3. 프롬프트 생성
                            context_str = ""
                            for item in related:
                                context_str += f"- [출처: {item['content']['source']} p.{item['content']['page']}] (유사도: {item['score']:.2f})\n...{item['content']['text'][:150]}...\n\n"
                            
                            prompt = f"""
                            당신은 의대생 튜터입니다. 아래 정보를 바탕으로 분석하세요.
                            
                            [현재 강의 내용]:
                            {current_text}
                            
                            [관련된 족보(기출)]:
                            {context_str}
                            
                            [요청사항]:
                            1. **기출 연계성**: 이 강의 내용이 과거 족보와 어떻게 연결되는지 한 문장으로 요약하세요.
                            2. **핵심 포인트**: 시험에 나올만한 키워드 3개를 뽑아주세요.
                            3. **예상 문제**: 이를 바탕으로 짧은 객관식 문제를
