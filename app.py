import streamlit as st
import google.generativeai as genai
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
from streamlit_mic_recorder import mic_recorder

# ==========================================
# 1. 설정 및 초기화
# ==========================================
st.set_page_config(page_title="Med-Study OS Final", layout="wide", page_icon="🩺")

if 'db' not in st.session_state: st.session_state.db = []
if 'lecture_doc' not in st.session_state: st.session_state.lecture_doc = None
if 'current_page' not in st.session_state: st.session_state.current_page = 0

# ==========================================
# 2. 핵심 함수 (Logic)
# ==========================================
def extract_text_from_pdf(file):
    """PDF를 텍스트로 변환 (fitz 사용)"""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    pages_content = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages_content.append({"page": page_num + 1, "text": text, "source": file.name})
    return pages_content

def get_embedding(text):
    """임베딩 (Embedding-004 사용)"""
    try:
        return genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )['embedding']
    except Exception:
        try:
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
# 3. 사이드바
# ==========================================
with st.sidebar:
    st.title("⚙️ 설정")
    api_key = st.text_input("Gemini API Key", type="password")
    if api_key:
        genai.configure(api_key=api_key)
        st.success("API Key 입력됨")
            
    st.divider()
    st.write(f"📚 학습된 족보: {len(st.session_state.db)} 페이지")
    if st.button("초기화"):
        st.session_state.db = []
        st.rerun()

# ==========================================
# 4. 메인 UI
# ==========================================
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
            status = st.empty()
            new_db = []
            total_files = len(files)
            
            for i, f in enumerate(files):
                status.text(f"📖 파일 읽는 중: {f.name}...")
                pages = extract_text_from_pdf(f)
                
                for j, p in enumerate(pages):
                    status.text(f"🧠 학습 중: {f.name} ({j+1}/{len(pages)} 페이지)...")
                    emb = get_embedding(p['text'])
                    if emb:
                        p['embedding'] = emb
                        new_db.append(p)
                    # [중요] 속도 제한 방지 대기
                    time.sleep(1.0) 
                
                bar.progress((i + 1) / total_files)
            
            st.session_state.db.extend(new_db)
            status.text("✅ 학습 완료!")
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
                    if not curr_text.strip():
                        st.warning("텍스트가 없는 페이지입니다.")
                    else:
                        with st.spinner("AI가 분석 중입니다..."):
                            try:
                                # 1. 관련 족보 찾기
                                related = find_relevant_jokbo(curr_text, st.session_state.db)
                                ctx_str = "\n".join([f"- {i['content']['text'][:100]}" for i in related])
                                
                                prompt = f"강의: {curr_text}\n족보: {ctx_str}\n\n연관성, 키워드, 문제 생성해줘."

                                # [핵심] 무료 한도가 넉넉한 1.5-flash 모델 강제 사용
                                model = genai.GenerativeModel("gemini-1.5-flash")
                                
                                response = model.generate_content(prompt)
                                st.markdown(response.text)
                                    
                            except Exception as e:
                                if "429" in str(e):
                                    st.error("⚠️ 사용량이 많습니다. 30초 뒤에 다시 시도해주세요.")
                                else:
                                    st.error(f"에러 발생: {e}")

# [업데이트] 오디오 처리를 위한 함수 추가
def process_audio_and_find_jokbo(audio_bytes, db):
    """오디오를 텍스트로 변환하고 관련 족보를 찾음"""
    if not db: return "학습된 족보가 없습니다.", []
    
    # 1. Gemini에게 오디오를 주고 텍스트 변환 요청 (STT)
    # 1.5-flash는 멀티모달이라 오디오 직접 입력 가능
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    # 오디오 바이트 처리
    prompt = "이 오디오 파일의 내용을 한국어로 정확하게 받아써줘(Transcribe)."
    
    try:
        # 오디오 데이터는 바이트 그대로 넘기기보다, 임시 파일 처리하거나 
        # API 구조에 맞게 Part 객체로 넘겨야 하지만, 
        # 간편하게는 텍스트 프롬프트로 처리하기보다 generate_content에 
        # mime_type을 지정한 blob 데이터를 넘기는 방식이 좋습니다.
        
        response = model.generate_content([
            prompt,
            {"mime_type": "audio/wav", "data": audio_bytes}
        ])
        transcribed_text = response.text
    except Exception as e:
        return f"오디오 처리 실패: {e}", []

    # 2. 변환된 텍스트로 족보 검색
    related_jokbo = find_relevant_jokbo(transcribed_text, db)
    
    return transcribed_text, related_jokbo

# ... (기존 get_embedding, find_relevant_jokbo 함수 동일) ...

# ==========================================
# 4. 메인 UI
# ==========================================
# 탭 구조 변경: 오디오 기능 탭 추가
tab1, tab2, tab3 = st.tabs(["📂 족보 학습", "📖 강의 공부", "🎙️ 실시간 강의 분석"])

# ... (tab1, tab2 코드는 기존과 동일 유지) ...

# --- TAB 3: 실시간 강의 분석 (신규 기능) ---
with tab3:
    st.header("3. 실시간 강의 듣기 & 족보 매칭")
    st.info("강의를 듣다가 '이거 나올 것 같은데?' 싶을 때 녹음하세요.")

    # 1. 녹음기 위젯
    # start_prompt: 녹음 시작 버튼 텍스트, stop_prompt: 정지 버튼 텍스트
    audio = mic_recorder(
        start_prompt="🔴 녹음 시작 (교수님 말씀)",
        stop_prompt="⏹️ 분석 시작",
        key='recorder',
        format="wav" # wav 포맷 권장
    )

    if audio:
        st.divider()
        st.subheader("🔊 분석 결과")
        
        if not api_key:
            st.error("설정 탭에서 API Key를 먼저 입력해주세요.")
        elif not st.session_state.db:
            st.warning("먼저 '족보 학습' 탭에서 족보를 학습시켜주세요.")
        else:
            with st.spinner("교수님 말씀 받아쓰기 & 족보 뒤지는 중..."):
                # 오디오 바이트 가져오기
                audio_bytes = audio['bytes']
                
                # 로직 수행
                transcript, related = process_audio_and_find_jokbo(audio_bytes, st.session_state.db)
                
                # 결과 출력 1: 스크립트
                st.markdown(f"**🗣️ 교수님 말씀 (STT):**")
                st.write(f"> {transcript}")
                
                # 결과 출력 2: 매칭된 족보
                st.markdown(f"**📄 관련 족보 내용:**")
                context_str = ""
                for idx, item in enumerate(related):
                    with st.expander(f"관련 족보 #{idx+1} (유사도: {item['score']:.4f})"):
                        st.write(f"페이지: {item['content']['page']}")
                        st.write(item['content']['text'])
                        context_str += f"- (페이지 {item['content']['page']}) {item['content']['text']}\n"

                # 결과 출력 3: 최종 인사이트 (AI 분석)
                st.divider()
                st.markdown("### 🩺 Med-Study AI의 통찰")
                
                if context_str:
                    final_prompt = f"""
                    상황: 의대 강의 중입니다.
                    교수님 말씀: {transcript}
                    
                    관련된 과거 족보 내용:
                    {context_str}
                    
                    미션:
                    1. 교수님의 말씀이 족보의 어떤 부분과 연결되는지 분석해.
                    2. "이 내용은 족보 O페이지의 내용 변형입니다" 또는 "족보에는 없던 새로운 강조점입니다" 처럼 구체적으로 지적해.
                    3. 시험에 어떻게 나올지 예측해줘.
                    """
                    
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    res = model.generate_content(final_prompt)
                    st.markdown(res.text)
                else:
                    st.write("관련된 족보 내용을 찾지 못했습니다. 새로운 내용일 수 있습니다!")
