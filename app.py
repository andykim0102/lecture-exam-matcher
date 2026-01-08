import streamlit as st
import pandas as pd
import google.generativeai as genai
from openai import OpenAI
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_pdf_viewer import pdf_viewer

# 1. 페이지 설정 및 뷰어 잘림 방지 스타일 적용
st.set_page_config(page_title="Med-Study AI Alpha", layout="wide")

st.markdown("""
    <style>
    .stMainBlockContainer { padding-top: 2rem; }
    iframe { min-height: 850px !important; border-radius: 12px; }
    .stAlert { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# 세션 상태 초기화
for key in ['notebook', 'pre_analysis', 'pdf_bytes', 'exam_db']:
    if key not in st.session_state:
        st.session_state[key] = [] if key != 'pdf_bytes' else None

# 2. AI 요약 엔진 (image_3188e7 에러 방지용 복구 로직 포함)
def get_ai_summary(text, api_key, provider="Gemini"):
    if not api_key: 
        return "🔑 사이드바에 API 키를 입력하면 AI 요약이 활성화됩니다."
    
    prompt = f"다음 의대 기출 지문을 핵심 기전 위주로 3줄 요약하세요:\n\n{text}"
    
    try:
        if provider == "Gemini":
            genai.configure(api_key=api_key)
            # [해결] 404 에러 방지를 위한 모델명 순회 (image_3188e7 대응)
            for model_name in ['gemini-1.5-flash', 'gemini-1.5-flash-latest', 'gemini-pro']:
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(prompt)
                    return response.text
                except:
                    continue
            return "❌ 모델 접근 권한이 없습니다. 새 프로젝트에서 API 키를 생성하세요."
            
        elif provider == "ChatGPT":
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini", 
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
            
    except Exception as e:
        # [해결] 429 에러 발생 시 안내 (image_3b72ac 대응)
        if "429" in str(e):
            return "⚠️ 할당량 초과(429). 1분 뒤 시도하거나 다른 API 키를 사용하세요."
        return f"🚨 요약 실패: {str(e)}"

# 3. 사이드바 API 설정
with st.sidebar:
    st.header("⚙️ AI 모델 설정")
    ai_provider = st.selectbox("LLM 엔진 선택", ["Gemini", "ChatGPT"])
    api_key = st.text_input(f"{ai_provider} API Key", type="password")
    st.caption("Gemini 404 에러 시 'Create API key in new project'로 발급받으세요.")

st.title("🩺 의대생 전용 스마트 학습 OS")
tab1, tab2, tab3 = st.tabs(["📅 사전 분석", "🎙️ 실시간 매칭", "🎯 나만의 정리본"])

# --- [Tab 1: 족보 DB화 및 분석] ---
with tab1:
    col_ex, col_lec = st.columns(2)
    with col_ex:
        st.subheader("📚 족보 등록")
        exam_files = st.file_uploader("족보 PDF 업로드", type="pdf", accept_multiple_files=True)
        if st.button("족보 인덱싱 시작"):
            db = []
            for f in exam_files:
                reader = PdfReader(f)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text: db.append({"source": f"{f.name} (p.{i+1})", "content": text.strip()})
            st.session_state.exam_db = db
            st.success(f"{len(db)}개의 유닛이 등록되었습니다.")

    with col_lec:
        st.subheader("📖 강의록 분석")
        lec_file = st.file_uploader("강의록 PDF", type="pdf")
        if lec_file:
            st.session_state.pdf_bytes = lec_file.getvalue()
            if st.button("기출 매칭 가동"):
                if not st.session_state.exam_db:
                    st.error("먼저 족보를 등록해주세요.")
                else:
                    reader = PdfReader(lec_file)
                    lec_texts = [p.extract_text() for p in reader.pages]
                    vec = TfidfVectorizer(ngram_range=(1, 2))
                    exam_matrix = vec.fit_transform([e['content'] for e in st.session_state.exam_db])
                    results = []
                    for i, p_text in enumerate(lec_texts):
                        if not p_text: continue
                        qv = vec.transform([p_text])
                        sims = cosine_similarity(qv, exam_matrix).flatten()
                        if sims.max() > 0.18:
                            idx = sims.argmax()
                            results.append({
                                "page": i+1, "score": sims.max(),
                                "info": st.session_state.exam_db[idx]['source'],
                                "content": st.session_state.exam_db[idx]['content']
                            })
                    st.session_state.pre_analysis = results
                    st.success("분석 완료!")
    # --- [Tab 2: 실시간 매칭 인터페이스] ---
with tab2:
    if st.session_state.pdf_bytes:
        c1, c2 = st.columns([1.1, 0.9])
        
        with c1:
            st.subheader("📄 강의록 뷰어")
            page_num = st.number_input("현재 페이지", min_value=1, value=1)
            # [해결] PDF 뷰어 높이 최적화 (image_3109da 대응)
            pdf_viewer(st.session_state.pdf_bytes, pages_to_render=[page_num], width=800, height=850)
        
        with c2:
            st.subheader("⚡ 실시간 족보 & AI 요약")
            page_hits = [h for h in st.session_state.pre_analysis if h['page'] == page_num]
            
            if page_hits:
                for h in page_hits:
                    with st.container(border=True):
                        st.error(f"🔥 기출 적중 ({int(h['score']*100)}% 일치)")
                        st.markdown(f"**📍 관련 족보:** {h['info']}")
                        
                        summary_key = f"sum_cache_{page_num}"
                        if st.button("🪄 AI 요약 요청", key=f"btn_{page_num}"):
                            with st.spinner("AI 분석 중..."):
                                st.session_state[summary_key] = get_ai_summary(h['content'], api_key, ai_provider)
                        
                        if summary_key in st.session_state:
                            st.info(st.session_state[summary_key])
                        else:
                            st.caption("버튼을 누르면 핵심 요약이 생성됩니다.")
                        
                        with st.expander("📄 원문 전체 확인"):
                            st.write(h['content'])
                        
                        user_note = st.text_area("💡 메모", key=f"note_{page_num}")
                        if st.button("📌 정리본 저장", key=f"save_{page_num}"):
                            st.session_state.notebook.append({
                                "page": page_num, "info": h['info'],
                                "summary": st.session_state.get(summary_key, "요약 없음"), "note": user_note
                            })
                            st.toast("저장 완료!")
            else:
                st.info("이 페이지는 관련 족보가 없습니다.")
    else:
        st.warning("1단계에서 강의록을 먼저 업로드하세요.")

# --- [Tab 3: 나만의 정리본] ---
with tab3:
    st.header("📝 스마트 단권화 정리본")
    if not st.session_state.notebook:
        st.info("저장된 항목이 없습니다.")
    else:
        for i, item in enumerate(reversed(st.session_state.notebook)):
            with st.expander(f"📔 [강의록 {item['page']}p] {item['info']}", expanded=True):
                st.markdown(f"**🤖 AI 요약:** {item['summary']}")
                st.success(f"**✏️ 메모:** {item['note']}")
                if st.button("🗑️ 삭제", key=f"del_{i}"):
                    st.session_state.notebook.pop(len(st.session_state.notebook)-1-i)
                    st.rerun()
