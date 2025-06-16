from langchain_openai import ChatOpenAI

class ScoringEngine:
    def __init__(self):
        self.weights = {
            "forensic": 0.4,
            "ratio": 0.3,
            "concall": 0.3
        }
        self.llm = ChatOpenAI(model_name="gpt-4", temperature=0)

    def score(self,
              ticker: str,
              forensic_output: Optional[dict],
              ratio_output: Optional[dict],
              concall_output: Optional[dict]) -> Scorecard:

        # ... same score calculations as before ...
        # (keep the code that calculates forensic_score, ratio_score, etc.)

        total = round(
            forensic_score * self.weights["forensic"] +
            ratio_score * self.weights["ratio"] +
            concall_score * self.weights["concall"]
        )

        if total >= 80:
            verdict = "Good"
        elif 60 <= total < 80:
            verdict = "Average"
        else:
            verdict = "Risky"

        # --- Build LLM prompt for summary ---
        combined_text = ""

        if forensic_output:
            combined_text += f"Forensic Agent Output:\n{forensic_output.get('final_answer', '')}\n"
        if ratio_output:
            combined_text += f"Ratio Agent Output:\n{ratio_output.get('final_summary', '')}\n"
        if concall_output:
            combined_text += f"Concall Agent Output:\n{concall_output.get('summary', '')}\n"

        llm_prompt = (
            "Summarize the following financial analysis in 3–4 sentences. "
            "Make it easy to understand for a general investor. Highlight strengths, risks, and notable financial signals.\n\n"
            + combined_text
        )

        llm_summary = self.llm.invoke(llm_prompt).content.strip()

        summary = (
            f"📊 Forensic Score: {forensic_score}/100\n"
            f"📈 Ratio Score: {ratio_score}/100\n"
            f"🗣️ Concall Score: {concall_score}/100\n"
            f"✅ Final Score: {total}/100\n"
            f"🏁 Verdict: {verdict} – based on combined analysis\n\n"
            f"📝 Summary:\n{llm_summary}"
        )

        return Scorecard(
            ticker=ticker,
            forensic_score=forensic_score,
            ratio_score=ratio_score,
            concall_score=concall_score,
            total_score=total,
            summary=summary,
            verdict=verdict
        )
