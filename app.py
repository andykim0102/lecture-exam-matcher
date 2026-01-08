import streamlit as st
import pandas as pd
import re
import google.generativeai as genai
from openai import OpenAI
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_mic_recorder import mic_recorder
from streamlit_pdf_viewer import pdf_viewer

# =========================
# 1. 초기 설정 및 AI 연결 함수
# =========================
st.set_page_config(page_title="Med-Study AI", layout="wide")

# 세션 초기화
for key in ['notebook', 'pre_analysis', 'pdf_bytes', 'exam_db']:
    if key not in st.session_state: st.session_state[key] = [] if key != 'pdf_bytes' else None

def get_ai_summary(text, api_key, provider="Gemini"):
    """LLM을 사용하여 방대한 족보 지문을 핵심 3줄로 요약"""
    if not api_key: return text[:300] + "..." # 키가 없으면 그냥 자르기
    
    prompt = f"의대생의 족보 공부를 돕기 위해 다음 지문을 핵심 내용만 3줄 이내로 요약해줘:\n\n{text}"
    try:
        if provider == "Gemini":
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            return response.text
        elif provider == "ChatGPT":
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o", messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
    except Exception as e:
        return f"요약 실패: {str(e)}"

# =========================
# 2. 메인 UI (사이드바에 API 설정 추가)
# =========================
with st.sidebar:
    st.header("⚙️ AI 설정")
    ai_provider = st.selectbox("LLM 선택", ["Gemini", "ChatGPT"])
    api_key = st.text_input(f"{ai_provider} API Key", type="password")
    st.info("API 키를 넣으면 족보 원문을 AI가 요약해줍니다.")

st.title("🩺 의대생 전용 스마트 학습 OS")

tab1, tab2, tab3 = st.tabs(["📅 수업 전: 분석", "🎙️ 수업 중: 실시간 매칭", "🎯 수업 후: 정리본"])

# --- [Tab 1: 수업 전 분석] (기존 로직 유지) ---
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📚 족보 등록")
        exam_files = st.file_uploader("족보 PDF 업로드", type="pdf", accept_multiple_files=True)
        if st.button("족보 인덱싱"):
            db = []
            for f in exam_files:
                reader = PdfReader(f)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text: db.append({"source": f"{f.name} (p.{i+1})", "content": text.strip()})
            st.session_state.exam_db = db
            st.success("족보 DB 구축 완료")

    with col2:
        st.subheader("📖 강의록 분석")
        lec_file = st.file_uploader("강의록 PDF", type="pdf")
        if lec_file:
            st.session_state.pdf_bytes = lec_file.getvalue()
            if st.button("사전 분석 실행"):
                lec_texts = [p.extract_text() for p in PdfReader(lec_file).pages]
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

# --- [Tab 2: 수업 중 뷰어 & AI 요약] ---
with tab2:
    if st.session_state.pdf_bytes:
        c1, c2 = st.columns([1.1, 0.9])
        
        with c1:
            st.subheader("📄 강의록 실시간 뷰어")
            page_num = st.number_input("페이지", min_value=1, value=1)
            # [해결 1] 컨테이너 높이를 고정하여 하단 잘림 방지
            with st.container(height=850, border=False):
                pdf_viewer(st.session_state.pdf_bytes, pages_to_render=[page_num], width=800)
        
        with c2:
            st.subheader("⚡ 실시간 족보/AI")
            page_hits = [h for h in st.session_state.pre_analysis if h['page'] == page_num]
            
            if page_hits:
                for h in page_hits:
                    with st.container(border=True):
                        st.error(f"🔥 기출 적중 ({int(h['score']*100)}% 일치)")
                        st.markdown(f"**📍 출처:** {h['info']}")
                        
                        # [해결 2, 3] AI 요약 적용
                        st.markdown("**🤖 AI 족보 요약**")
                        with st.spinner("AI가 요약 중..."):
                            summary = get_ai_summary(h['content'], api_key, ai_provider)
                            st.info(summary)
                        
                        with st.expander("📄 원문 전체 보기"):
                            st.write(h['content'])
                        
                        user_note = st.text_area("메모 입력", key=f"note_{page_num}")
                        if st.button("📌 저장", key=f"btn_{page_num}"):
                            st.session_state.notebook.append({
                                "page": page_num, "info": h['info'], 
                                "summary": summary, "note": user_note
                            })
                            st.toast("저장 완료!")
            else:
                st.info("기출 포인트가 없습니다.")
