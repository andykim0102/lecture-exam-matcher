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

# 1. 초기 설정
st.set_page_config(page_title="Med-Study AI", layout="wide")

# [해결 1] 뷰어 잘림 방지를 위한 강제 스타일 설정
st.markdown("""
    <style>
    .stMainBlockContainer { padding-top: 2rem; }
    iframe { min-height: 850px !important; }
    </style>
    """, unsafe_allow_html=True)

for key in ['notebook', 'pre_analysis', 'pdf_bytes', 'exam_db']:
    if key not in st.session_state: st.session_state[key] = [] if key != 'pdf_bytes' else None

# [해결 2, 3] AI 요약 함수 (가독성 개선)
def get_ai_summary(text, api_key, provider="Gemini"):
    if not api_key: return "⚠️ API 키를 입력하면 AI 요약이 제공됩니다."
    
    prompt = f"다음은 의대 기출문제 지문입니다. 핵심 내용만 3줄 이내로 요약해줘:\n\n{text}"
    try:
        if provider == "Gemini":
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')
            return model.generate_content(prompt).text
        elif provider == "ChatGPT":
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o", messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
    except Exception as e:
        return f"요약 실패: {str(e)}"

# 사이드바 API 설정
with st.sidebar:
    st.header("⚙️ AI 설정")
    ai_provider = st.selectbox("LLM 선택", ["Gemini", "ChatGPT"])
    api_key = st.text_input(f"{ai_provider} API Key", type="password")

st.title("🩺 의대생 전용 스마트 학습 OS")
tab1, tab2, tab3 = st.tabs(["📅 사전 분석", "🎙️ 실시간 매칭", "🎯 나만의 정리본"])

# --- [Tab 1: 분석] (로직 동일) ---
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📚 족보 등록")
        exam_files = st.file_uploader("족보 PDF 업로드", type="pdf", accept_multiple_files=True)
        if st.button("족보 DB화"):
            db = []
            for f in exam_files:
                for i, page in enumerate(PdfReader(f).pages):
                    text = page.extract_text()
                    if text: db.append({"source": f"{f.name} (p.{i+1})", "content": text.strip()})
            st.session_state.exam_db = db
            st.success("완료")
    with col2:
        st.subheader("📖 강의록 분석")
        lec_file = st.file_uploader("강의록 PDF", type="pdf")
        if lec_file:
            st.session_state.pdf_bytes = lec_file.getvalue()
            if st.button("매칭 시작"):
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
                        results.append({"page": i+1, "score": sims.max(), "info": st.session_state.exam_db[idx]['source'], "content": st.session_state.exam_db[idx]['content']})
                st.session_state.pre_analysis = results
                st.success("완료!")

# --- [Tab 2: 실시간 & 뷰어] ---
with tab2:
    if st.session_state.pdf_bytes:
        c1, c2 = st.columns([1.1, 0.9])
        with c1:
            st.subheader("📄 강의록 뷰어")
            page_num = st.number_input("페이지", min_value=1, value=1)
            # [해결 1] 높이 고정 및 스크롤 영역 확보
            pdf_viewer(st.session_state.pdf_bytes, pages_to_render=[page_num], width=800, height=900)
        
        with c2:
            st.subheader("⚡ 실시간 족보/AI")
            page_hits = [h for h in st.session_state.pre_analysis if h['page'] == page_num]
            if page_hits:
                for h in page_hits:
                    with st.container(border=True):
                        st.error(f"🔥 기출 적중 ({int(h['score']*100)}% 일치)")
                        st.markdown(f"**📍 출처:** {h['info']}")
                        
                        # [해결 2] AI 요약본 우선 노출 (가독성 최우선)
                        st.markdown("**🤖 AI 핵심 요약**")
                        summary = get_ai_summary(h['content'], api_key, ai_provider)
                        st.info(summary)
                        
                        # 원문은 접어두기
                        with st.expander("📄 원문 전체 보기"):
                            st.write(h['content'])
                        
                        note = st.text_area("수업 중 메모", key=f"note_{page_num}")
                        if st.button("📌 내 정리본에 추가", key=f"btn_{page_num}"):
                            st.session_state.notebook.append({"page": page_num, "info": h['info'], "summary": summary, "note": note})
                            st.toast("저장되었습니다!")
            else:
                st.info("기출 내역이 없습니다.")

# --- [Tab 3: 정리본] ---
with tab3:
    st.header("📝 나만의 스마트 정리본")
    for i, item in enumerate(st.session_state.notebook):
        with st.expander(f"📔 [강의록 {item['page']}p] {item['info']}", expanded=True):
            st.markdown(f"**🤖 AI 요약:** {item['summary']}")
            st.success(f"**✏️ 나의 메모:** {item['note']}")
            if st.button("삭제", key=f"del_{i}"):
                st.session_state.notebook.pop(i)
                st.rerun()
