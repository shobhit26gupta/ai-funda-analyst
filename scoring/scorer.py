import os
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

load_dotenv()

# --- Scorecard model --- #
class Scorecard(BaseModel):
    ticker: str
    forensic_score: int
    ratio_score: int
    concall_score: int
    total_score: int
    summary: str
    verdict: str

# --- Scoring engine --- #
class ScoringEngine:
    def __init__(self):
        self.weights = {
            "forensic": 0.4,
            "ratio": 0.3,
            "concall": 0.3
        }
        openai_key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(api_key=openai_key, model_name="gpt-4", temperature=0)

    def score(self,
              ticker: str,
              forensic_output: Optional[dict],
              ratio_output: Optional[dict],
              concall_output: Optional[dict]) -> Scorecard:

        # Forensic score
        forensic_score = 100
        if forensic_output:
            red_flags = [f for f in forensic_output.get("findings", []) if "red" in f["detail"].lower()]
            yellow_flags = [f for f in forensic_output.get("findings", []) if "yellow" in f["detail"].lower()]
            forensic_score -= len(red_flags) * 20 + len(yellow_flags) * 10
        forensic_score = max(forensic_score, 0)

        # Ratio score
        ratio_score = 80
        if ratio_output and len(ratio_output.get("dupont_breakdown", [])) >= 3:
            ratio_score += 10
        if ratio_output and "complex" in ratio_output.get("final_summary", "").lower():
            ratio_score -= 10
        ratio_score = min(max(ratio_score, 0), 100)

        # Concall score
        concall_score = 75
        if concall_output:
            sentiment = concall_output.get("sentiment", "").lower()
            confidence = concall_output.get("confidence", "").lower()
            if "positive" in sentiment:
                concall_score += 10
            elif "negative" in sentiment:
                concall_score -= 15
            if "high" in confidence:
                concall_score += 5
            elif "low" in confidence:
                concall_score -= 10
        concall_score = min(max(concall_score, 0), 100)

        # Final weighted score
        total = round(
            forensic_score * self.weights["forensic"] +
            ratio_score * self.weights["ratio"] +
            concall_score * self.weights["concall"]
        )

        # Verdict
        if total >= 80:
            verdict = "Good"
        elif total >= 60:
            verdict = "Average"
        else:
            verdict = "Risky"

        # Generate summary
        combined_text = ""
        if forensic_output:
            combined_text += "Forensic Agent Output:\n" + forensic_output.get("final_answer", "") + "\n"
        if ratio_output:
            combined_text += "Ratio Agent Output:\n" + ratio_output.get("final_summary", "") + "\n"
        if concall_output:
            combined_text += "Concall Agent Output:\n" + concall_output.get("summary", "") + "\n"

        summary_prompt = (
            "Summarize the following financial analysis in 3–4 sentences. "
            "Make it easy to understand for a general investor. Highlight strengths, risks, and signals:\n\n"
            + combined_text
        )
        llm_summary = self.llm.invoke(summary_prompt).content.strip()

        summary = (
            # f"📊 Forensic Score: {forensic_score}/100\n"
            # f"📈 Ratio Score: {ratio_score}/100\n"
            # f"🗣️ Concall Score: {concall_score}/100\n"
            # f"✅ Final Score: {total}/100\n"
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