# app.py

import streamlit as st
from agents.forensic_agent import ReActForensicAgent
from agents.ratio_agent import ReActRatioAgent
from agents.concall_agent import ReActConcallAgent
from router.router import RouterAgent
from scoring.scorer import ScoringEngine
from agents.rag_pipeline import process_document, query_document

# 📌 Setup
st.set_page_config(page_title="AI Fundamental Analyst", layout="wide")
st.title("💼 AI Fundamental Analyst")
st.caption("Built with OpenAI + Screener + Tavily")

# 🔑 API Inputs
# openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
# tavily_key = st.sidebar.text_input("Tavily API Key", type="password")

from dotenv import load_dotenv
import os

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")


# 📤 User Input
user_query = st.text_area("📩 Ask a financial analysis question", value="Give me a full score for INFY")
ticker = st.text_input("🏢 Company Ticker (e.g., INFY)", value="INFY")

if st.button("🔍 Analyze"):
    if not openai_key or not tavily_key:
        st.warning("Please enter both API keys.")
        st.stop()

    with st.spinner("🤖 Thinking..."):

        # ⚙ Initialize agents
        router = RouterAgent()
        forensic_agent = ReActForensicAgent()
        ratio_agent = ReActRatioAgent()
        concall_agent = ReActConcallAgent()
        scorer = ScoringEngine()

        # 🔀 Route
        route_result = router.route(user_query)
        st.markdown("### 🔀 Routing Decision")
        st.json(route_result)

        agents_to_call = route_result["agents"]

        # 🧠 Call agents
        forensic_out = ratio_out = concall_out = None

        if "FORENSIC_AGENT" in agents_to_call:
            st.subheader("🔎 Forensic Agent Running...")
            forensic_out = forensic_agent.run(ticker)
            st.text_area("📝 Forensic Output", forensic_out.final_answer, height=200)

        if "RATIO_AGENT" in agents_to_call:
            st.subheader("📊 Ratio Agent Running...")
            ratio_out = ratio_agent.run(ticker)
            st.text_area("📝 Ratio Summary", ratio_out.final_summary, height=200)

        if "CONCALL_AGENT" in agents_to_call:
            st.subheader("🗣️ Concall Agent Running...")
            concall_out = concall_agent.run(ticker)
            st.text_area("📝 Concall Summary", concall_out.summary, height=200)

        # 📈 Score
        st.header("🏁 Final Scorecard")
        result = scorer.score(
            ticker=ticker,
            forensic_output=forensic_out.dict() if forensic_out else None,
            ratio_output=ratio_out.dict() if ratio_out else None,
            concall_output=concall_out.dict() if concall_out else None
        )

        st.success(result.verdict)
        st.text_area("🧾 Full Summary", result.summary, height=300)

st.markdown("---")
st.header("📄 Ask Questions from Your Own Document")

uploaded_file = st.file_uploader("Upload a PDF document to enable RAG-based querying")

if uploaded_file:
    with open("temp_uploaded.pdf", "wb") as f:
        f.write(uploaded_file.read())

    process_document("temp_uploaded.pdf")
    st.success("✅ Document processed. You can now ask questions.")

    user_question = st.text_input("❓ Enter your question about the document")

    if user_question:
        with st.spinner("🔎 Searching with RAG..."):
            answer = query_document(user_question)
            st.markdown(f"**📘 Answer:** {answer}")
