# agents/forensic_agent.py

import os
import yfinance as yf
import requests
from typing import List
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()


# --- Output models --- #
class Finding(BaseModel):
    name: str
    severity: str
    detail: str


class Report(BaseModel):
    ticker: str
    findings: List[Finding]
    final_answer: str


# --- Single-pass Forensic Agent --- #
class ReActForensicAgent:
    def __init__(self):
        openai_key = os.getenv("OPENAI_API_KEY")
        tavily_key = os.getenv("TAVILY_API_KEY")

        self.llm = ChatOpenAI(api_key=openai_key, model_name="gpt-4", temperature=0)
        self.search = TavilySearchResults(api_key=tavily_key, k=5)

        self.base_prompt = """
You are a forensic accounting expert specializing in detecting accounting fraud and earnings manipulation.

You will receive basic financial data (cash flow, income, balance sheet) along with recent promoter-related news.

Your task is to analyze and summarize **potential red flags** in accounting, especially in:

- Cash flow discrepancies
- Revenue recognition
- Related party transactions
- Audit report concerns
- Contingent liabilities
- Unusual expenses
- Management commentary

Do not speculate — use only the data provided.

Return your answer as:

1. Summary of key concerns (3–5 lines)
2. Bullet checklist of potential issues
3. Label the accounting quality as: ✅ Good, ⚠️ Average, or ❌ Risky

Be clear, objective, and explain like you're presenting to a finance team.
"""

    def run(self, ticker: str) -> Report:
        # 1. Fetch data from yfinance
        try:
            stock = yf.Ticker(ticker)
            fin = stock.financials.T.iloc[-2:].to_string()
            bal = stock.balance_sheet.T.iloc[-2:].to_string()
            cf = stock.cashflow.T.iloc[-2:].to_string()
        except Exception as e:
            raise RuntimeError(f"Unable to fetch financials for {ticker}: {e}")

        # 2. Get promoter news via Tavily
        query = f"{ticker} promoter fraud audit red flags site:moneycontrol.com OR site:trendlyne.com"
        search_results = self.search.run(query)
        news_snippets = "\n".join([item.get("content", "") for item in search_results])

        # 3. Build prompt
        full_prompt = (
            self.base_prompt
            + f"\n\nFinancials:\n{fin}\n\nCash Flow:\n{cf}\n\nBalance Sheet:\n{bal}"
            + f"\n\nPromoter News:\n{news_snippets}"
        )

        # 4. Ask LLM
        final_answer = self.llm.invoke(full_prompt).content.strip()

        # 5. Heuristic finding detection
        findings = []
        if "red flag" in final_answer.lower() or "❌" in final_answer:
            findings.append(Finding(
                name="Potential red flags",
                severity="high",
                detail=final_answer[:300]
            ))
        elif "⚠️" in final_answer or "concern" in final_answer.lower():
            findings.append(Finding(
                name="Mild concerns",
                severity="medium",
                detail=final_answer[:300]
            ))
        else:
            findings.append(Finding(
                name="No red flags",
                severity="low",
                detail="No major issues detected in single-pass analysis."
            ))

        return Report(
            ticker=ticker,
            findings=findings,
            final_answer=final_answer
        )