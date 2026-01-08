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

# 세션 상태 초기화
for key in ['notebook', 'pre_analysis', 'pdf_bytes', 'exam_db']:
    if key not in st.session_state: 
        st.session_state[key] = [] if key != 'pdf_bytes' else None

def get_ai_summary(text, api_key, provider="Gemini"):
    if not api_key: 
        return "🔑 사이드바에 API 키를 입력하면 AI 요약이 활성화됩니다."
    
    prompt = f"다음 의대 기출 지문을 핵심 위주로 3줄 요약해줘:\n\n{text}"
    
    try:
        if provider == "Gemini":
            genai.configure(api_key=api_key)
            # [수정] 모델명을 gemini-pro에서 gemini-1.5-flash로 변경 (가장 안정적)
            model = genai.GenerativeModel('gemini-1.5-flash') 
            response = model.generate_content(prompt)
            return response.text
        elif provider == "ChatGPT":
            client = OpenAI(api_key=api_key)
            # [수정] GPT 모델도 최신 gpt-4o-mini로 변경 (비용 절감 및 속도 향상)
            response = client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
    except Exception as e:
        if "429" in str(e):
            return "⚠️ API 할당량이 초과되었습니다. 잠시 후 다시 시도하거나 계정 설정을 확인해주세요."
        return f"요약 실패: {str(e)}"

# 사이드바 API 설정
with st.sidebar:
    st.header("⚙️ AI 설정")
    ai_provider = st.selectbox("LLM 선택", ["Gemini", "ChatGPT"])
    api_key = st.text_input(f"{ai_provider} API Key", type="password")

st.title("🩺 의대생 전용 스마트 학습 OS")
tab1, tab2, tab3 = st.tabs(["📅 사전 분석", "🎙️ 실시간 매칭", "🎯 나만의 정리본"])

# --- [Tab 1: 분석] ---
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📚 족보 등록")
        exam_files = st.file_uploader("족보 PDF 업로드", type="pdf", accept_multiple_files=True)
        if st.button("족보 DB화"):
            db = []
            for f in exam_files:
                reader = PdfReader(f)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text: db.append({"source": f"{f.name} (p.{i+1})", "content": text.strip()})
            st.session_state.exam_db = db
            st.success(f"{len(db)}개의 족보 유닛이 등록되었습니다.")

    with col2:
        st.subheader("📖 강의록 분석")
        lec_file = st.file_uploader("강의록 PDF", type="pdf")
        if lec_file:
            st.session_state.pdf_bytes = lec_file.getvalue()
            if st.button("매칭 시작"):
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
                                "page": i+1, 
                                "score": sims.max(), 
                                "info": st.session_state.exam_db[idx]['source'], 
                                "content": st.session_state.exam_db[idx]['content']
                            })
                    st.session_state.pre_analysis = results
                    st.success(f"분석 완료! {len(results)}개 페이지 적중.")

# --- [Tab 2: 실시간 & 뷰어] ---
with tab2:
    if st.session_state.pdf_bytes:
        c1, c2 = st.columns([1.1, 0.9])
        with c1:
            st.subheader("📄 강의록 뷰어")
            page_num = st.number_input("페이지 선택", min_value=1, value=1)
            pdf_viewer(st.session_state.pdf_bytes, pages_to_render=[page_num], width=800, height=850)
        
        with c2:
            st.subheader("⚡ 실시간 족보 & AI 요약")
            page_hits = [h for h in st.session_state.pre_analysis if h['page'] == page_num]
            
            if page_hits:
                for h in page_hits:
                    with st.container(border=True):
                        st.error(f"🔥 기출 적중 ({int(h['score']*100)}% 일치)")
                        st.markdown(f"**📍 출처:** {h['info']}")
                        
                        st.markdown("**🤖 AI 족보 브리핑**")
                        summary_key = f"sum_res_{page_num}"
                        
                        # [개선] 버튼 클릭 시에만 AI 요약 실행
                        if st.button("🪄 AI 요약 요청하기", key=f"btn_sum_{page_num}"):
                            with st.spinner("AI가 분석 중..."):
                                summary = get_ai_summary(h['content'], api_key, ai_provider)
                                st.session_state[summary_key] = summary
                        
                        if summary_key in st.session_state:
                            st.info(st.session_state[summary_key])
                        else:
                            st.caption("위 버튼을 누르면 AI가 핵심 요약을 생성합니다.")
                        
                        with st.expander("📄 원문 전체 확인"):
                            st.write(h['content'])
                        
                        user_note = st.text_area("중요 메모 입력", key=f"note_{page_num}")
                        if st.button("📌 내 정리본에 추가", key=f"save_{page_num}"):
                            final_summary = st.session_state.get(summary_key, h['content'][:100] + "...")
                            st.session_state.notebook.append({
                                "page": page_num, 
                                "info": h['info'], 
                                "summary": final_summary, 
                                "note": user_note
                            })
                            st.toast("정리본에 저장되었습니다!")
            else:
                st.info("이 페이지는 관련 족보 기록이 없습니다.")
    else:
        st.warning("먼저 1단계에서 강의록을 업로드해주세요.")

# --- [Tab 3: 정리본] ---
with tab3:
    st.header("📝 나만의 스마트 정리본")
    if not st.session_state.notebook:
        st.info("수업 중 저장한 내용이 여기에 모입니다.")
    else:
        for i, item in enumerate(st.session_state.notebook):
            with st.expander(f"📔 [강의록 {item['page']}p] {item['info']}", expanded=True):
                st.markdown(f"**🤖 AI 요약:** {item['summary']}")
                st.success(f"**✏️ 나의 메모:** {item['note']}")
                if st.button("삭제", key=f"del_{i}"):
                    st.session_state.notebook.pop(i)
                    st.rerun()

