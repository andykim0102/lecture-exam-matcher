# app.py (UI: Modern Card / Logic: Show ALL Relevant + Instant Auto-Analysis + Styled Twin)
# app.py (Full Version: Auto-Analysis + Chat + Recording Restored)
import time
import re
import random
@@ -165,10 +165,7 @@
       line-height: 1.5;
   }

    /* 13. Hot Page Button */
    .hot-page-btn-score { font-size: 0.8em; color: #ff3b30; }

    /* 14. Sidebar Items */
    /* 13. Sidebar Items */
   .sidebar-subject {
       padding: 10px 15px;
       background-color: white;
@@ -199,7 +196,9 @@
# Interactive Parsing & Twin Gen
"parsed_items": {}, "twin_items": {},
# Hot Page Navigation
    "hot_pages": [], "hot_pages_analyzed": False, "analyzing_progress": 0
    "hot_pages": [], "hot_pages_analyzed": False, "analyzing_progress": 0,
    # Tab 3 Results
    "tr_res": None
}

for k, v in defaults.items():
@@ -304,7 +303,6 @@ def get_embedding_robust(text: str, status_placeholder=None):
def filter_db_by_subject(subject: str, db: list[dict]):
if not db: return []
if subject in ["전체", "ALL", ""]: return db
    # Strict filtering
return [x for x in db if x.get("subject") == subject]

def find_relevant_jokbo(query_text: str, db: list[dict], top_k: int = 5):
@@ -347,11 +345,12 @@ def transcribe_audio_gemini(audio_bytes, api_key):
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content([
            "Transcribe this audio.",
            "Please transcribe the following audio file into text accurately. Do not add any conversational text, just the transcription.",
{"mime_type": "audio/wav", "data": audio_bytes}
])
return response.text
    except Exception: return None
    except Exception as e:
        return None

def transcribe_image_to_text(image, api_key):
try:
@@ -434,7 +433,18 @@ def build_chat_prompt(history, context, related, q):
   """

def build_transcript_prompt(chunks, related_packs, subject):
    return "요약해."
    packed = ""
    for idx, (chunk, rel) in enumerate(zip(chunks, related_packs), 1):
        if not rel or rel[0]["score"] < 0.6: continue
        ctx = "\n".join([f"- {r['content']['text'][:200]}" for r in rel[:2]])
        packed += f"\n(구간 {idx})\n[강의] {chunk}\n[족보근거] {ctx}\n"
    if not packed: return "족보와 관련된 내용이 없습니다."
    return f"""
    당신은 의대 조교입니다. 강의 전사 내용을 족보 기반으로 요약하세요.
    과목: {subject}
    {packed}
    출력: [족보 적중 노트] 형식으로 요약.
    """

def chunk_transcript(text):
return [text[i:i+900] for i in range(0, len(text), 900)]
@@ -643,97 +653,169 @@ def get_subject_files(subject):
p_text = page.get_text().strip()
st.image(img, use_container_width=True)

            # RIGHT: Auto-Analysis
            # RIGHT: Auto-Analysis + Chat (RESTORED TABS)
with col_ai:
                if not p_text:
                    st.info("텍스트가 없는 페이지입니다.")
                else:
                    # Retrieve Related Items
                    psig = hash(p_text)
                    if psig != st.session_state.last_page_sig:
                        st.session_state.last_page_sig = psig
                        sub_db = filter_db_by_subject(target_subj, st.session_state.db)
                        st.session_state.last_related = find_relevant_jokbo(p_text, sub_db)
                    
                    rel = st.session_state.last_related
                    
                    if not rel:
                        st.info("💡 이 페이지와 직접 연관된 족보 문항이 없습니다.")
                ai_tab_match, ai_tab_chat = st.tabs(["📝 족보 매칭", "💬 AI 튜터"])
                
                # --- Tab 2-1: Jokbo Matching ---
                with ai_tab_match:
                    if not p_text:
                        st.info("텍스트가 없는 페이지입니다.")
else:
                        st.success(f"🔥 **{len(rel)}개의 관련 족보 문항이 발견되었습니다.**")
                        # Retrieve Related Items
                        psig = hash(p_text)
                        if psig != st.session_state.last_page_sig:
                            st.session_state.last_page_sig = psig
                            sub_db = filter_db_by_subject(target_subj, st.session_state.db)
                            st.session_state.last_related = find_relevant_jokbo(p_text, sub_db)

                        # Loop through ALL relevant items (Limit to top 5 to avoid infinite loop lag if many)
                        # The user wants to see "relevant items", not just top 2.
                        display_rel = rel[:10] # Display up to 10 relevant items

                        for i, r in enumerate(display_rel): 
                            content = r['content']
                            score = r['score']
                            raw_txt = content['text']
                            
                            # Split questions from raw text
                            questions = split_jokbo_text(raw_txt)
                            if not questions: questions = [raw_txt]
                        rel = st.session_state.last_related
                        
                        if not rel:
                            st.info("💡 이 페이지와 직접 연관된 족보 문항이 없습니다.")
                        else:
                            st.success(f"🔥 **{len(rel)}개의 관련 족보 문항이 발견되었습니다.**")

                            for q_idx, q_txt in enumerate(questions):
                                item_id = f"{psig}_{i}_{q_idx}"
                            display_rel = rel[:10] 

                            for i, r in enumerate(display_rel): 
                                content = r['content']
                                score = r['score']
                                raw_txt = content['text']

                                # 1. Display Original Exam Card
                                st.markdown(f"""
                                <div class="exam-card">
                                    <div class="exam-meta">
                                        <span><span class="exam-score-badge">유사도 {score:.0%}</span> &nbsp; {content['source']} (P.{content['page']})</span>
                                    </div>
                                    <div class="exam-question">
                                        {q_txt[:500] + ('...' if len(q_txt)>500 else '')}
                                # Split questions from raw text
                                questions = split_jokbo_text(raw_txt)
                                if not questions: questions = [raw_txt]
                                
                                for q_idx, q_txt in enumerate(questions):
                                    item_id = f"{psig}_{i}_{q_idx}"
                                    
                                    # 1. Display Original Exam Card
                                    st.markdown(f"""
                                    <div class="exam-card">
                                        <div class="exam-meta">
                                            <span><span class="exam-score-badge">유사도 {score:.0%}</span> &nbsp; {content['source']} (P.{content['page']})</span>
                                        </div>
                                        <div class="exam-question">
                                            {q_txt[:500] + ('...' if len(q_txt)>500 else '')}
                                        </div>
                                   </div>
                                </div>
                                """, unsafe_allow_html=True)

                                # 2. Instant Auto-Analysis Logic
                                if item_id not in st.session_state.parsed_items:
                                    with st.spinner(f"⚡ 문항 #{i+1}-{q_idx+1} 분석 중..."):
                                        parsed = parse_raw_jokbo_llm(q_txt)
                                        st.session_state.parsed_items[item_id] = parsed
                                    """, unsafe_allow_html=True)

                                    # 2. Instant Auto-Analysis Logic
                                    if item_id not in st.session_state.parsed_items:
                                        with st.spinner(f"⚡ 문항 #{i+1}-{q_idx+1} 분석 중..."):
                                            parsed = parse_raw_jokbo_llm(q_txt)
                                            st.session_state.parsed_items[item_id] = parsed
                                            if parsed["success"]:
                                                twin = generate_twin_problem_llm(parsed, target_subj)
                                                st.session_state.twin_items[item_id] = twin
                                            st.rerun()

                                    # 3. Render Analysis Results (Styled)
                                    if item_id in st.session_state.parsed_items:
                                        parsed = st.session_state.parsed_items[item_id]
if parsed["success"]:
                                            twin = generate_twin_problem_llm(parsed, target_subj)
                                            st.session_state.twin_items[item_id] = twin
                                        st.rerun() # Refresh to show results immediately

                                # 3. Render Analysis Results (Styled)
                                if item_id in st.session_state.parsed_items:
                                    parsed = st.session_state.parsed_items[item_id]
                                    if parsed["success"]:
                                        d = parsed["data"]
                                        
                                        # Use Tabs for cleaner layout
                                        t_ans, t_twin = st.tabs(["💡 정답 및 해설", "🧩 쌍둥이(변형) 문제"])
                                        
                                        with t_ans:
                                            st.markdown(f"""
                                            <div class="explanation-box">
                                                <div class="exp-title">✅ 정답</div>
                                                <div class="exp-text">{d.get('answer','정보 없음')}</div>
                                                <br>
                                                <div class="exp-title">📘 상세 해설</div>
                                                <div class="exp-text">{d.get('explanation','정보 없음')}</div>
                                            </div>
                                            """, unsafe_allow_html=True)
                                        
                                        with t_twin:
                                            twin_content = st.session_state.twin_items.get(item_id, "생성 실패")
                                            # Render Twin as a Card
                                            st.markdown(f"""
                                            <div class="twin-card">
                                                <div class="twin-badge">TWIN PROBLEM</div>
                                                <div class="exam-question">
                                                    {twin_content}
                                            d = parsed["data"]
                                            
                                            # Use Tabs for cleaner layout
                                            t_ans, t_twin = st.tabs(["💡 정답 및 해설", "🧩 쌍둥이(변형) 문제"])
                                            
                                            with t_ans:
                                                st.markdown(f"""
                                                <div class="explanation-box">
                                                    <div class="exp-title">✅ 정답</div>
                                                    <div class="exp-text">{d.get('answer','정보 없음')}</div>
                                                    <br>
                                                    <div class="exp-title">📘 상세 해설</div>
                                                    <div class="exp-text">{d.get('explanation','정보 없음')}</div>
                                               </div>
                                            </div>
                                            """, unsafe_allow_html=True)
                                    else:
                                        st.error("분석 실패 (텍스트가 불완전합니다)")

# --- TAB 3: 녹음 (Existing) ---
                                                """, unsafe_allow_html=True)
                                            
                                            with t_twin:
                                                twin_content = st.session_state.twin_items.get(item_id, "생성 실패")
                                                st.markdown(f"""
                                                <div class="twin-card">
                                                    <div class="twin-badge">TWIN PROBLEM</div>
                                                    <div class="exam-question">
                                                        {twin_content}
                                                    </div>
                                                </div>
                                                """, unsafe_allow_html=True)
                                        else:
                                            st.error("분석 실패 (텍스트가 불완전합니다)")
                
                # --- Tab 2-2: Chat Interface (Restored) ---
                with ai_tab_chat:
                    st.caption("현재 보고 있는 강의 페이지와 관련된 질문을 해보세요.")
                    for msg in st.session_state.chat_history:
                        with st.chat_message(msg["role"]):
                            st.markdown(msg["content"])
                    
                    if prompt := st.chat_input("질문 입력..."):
                        if not st.session_state.api_key_ok: st.error("API Key 필요")
                        else:
                            st.session_state.chat_history.append({"role": "user", "content": prompt})
                            with st.chat_message("user"): st.markdown(prompt)
                            
                            with st.chat_message("assistant"):
                                with st.spinner("답변 생성 중..."):
                                    # Provide context from current page & related items
                                    p_context = p_text if p_text else "No text"
                                    rel_context = st.session_state.last_related
                                    chat_prmt = build_chat_prompt(st.session_state.chat_history, p_context, rel_context, prompt)
                                    
                                    response_text, _ = generate_with_fallback(chat_prmt, st.session_state.text_models)
                                    st.markdown(response_text)
                                    st.session_state.chat_history.append({"role": "assistant", "content": response_text})

# --- TAB 3: 녹음 (Restored) ---
with tab3:
    st.info("녹음 기능 활성화 상태")
    with st.container(border=True):
        st.markdown("#### 🎙️ 강의 녹음/분석")
        
        c_in, c_out = st.columns(2)
        with c_in:
            sub_t3 = st.selectbox("과목", ["전체"] + sorted({x.get("subject", "") for x in st.session_state.db}), key="t3_s")
            t3_mode = st.radio("입력 방식", ["🎤 직접 녹음", "📂 파일 업로드 / 텍스트"], horizontal=True, label_visibility="collapsed")
            target_text = ""
            
            if t3_mode == "🎤 직접 녹음":
                audio_value = st.audio_input("녹음 시작")
                if audio_value:
                    if st.button("🚀 녹음 내용 분석하기", type="primary", use_container_width=True, key="btn_audio_analyze"):
                        if not st.session_state.api_key_ok: st.error("API Key 필요")
                        else:
                            with st.spinner("음성을 텍스트로 변환 중..."):
                                transcript = transcribe_audio_gemini(audio_value.getvalue(), st.session_state.api_key)
                                if transcript:
                                    st.session_state.transcribed_text = transcript
                                    target_text = transcript
                                else: st.error("변환 실패")
            else:
                f_txt = st.file_uploader("전사 파일(.txt)", type="txt", key="t3_f")
                area_txt = st.text_area("직접 입력", height=200, placeholder="강의 내용을 입력하세요...")
                if st.button("분석 실행", type="primary", use_container_width=True):
                    target_text = (f_txt.getvalue().decode() if f_txt else area_txt).strip()
            
            if target_text:
                if not st.session_state.api_key_ok: st.error("API Key 필요")
                else:
                    with st.spinner("족보 데이터와 대조하여 분석 중..."):
                        sdb = filter_db_by_subject(sub_t3, st.session_state.db)
                        chks = chunk_transcript(target_text)[:10]
                        rels = [find_relevant_jokbo(c, sdb, top_k=3) for c in chks]
                        pmt = build_transcript_prompt(chks, rels, sub_t3)
                        res, _ = generate_with_fallback(pmt, st.session_state.text_models)
                        st.session_state.tr_res = res
                    st.success("분석 완료!")

        with c_out:
            st.caption("분석 결과")
            if st.session_state.tr_res:
                st.info(st.session_state.tr_res)
                if st.session_state.transcribed_text:
                    with st.expander("📝 변환된 전체 텍스트 보기"):
                        st.text(st.session_state.transcribed_text)
            else:
                st.markdown("""<div style="height: 300px; background: #f9f9f9; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: #aaa;">결과가 여기에 표시됩니다.</div>""", unsafe_allow_html=True)
