import os
import requests
from typing import List
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
load_dotenv()


# --- Output model --- #
class ConcallInsight(BaseModel):
    ticker: str
    summary: str
    sentiment: str
    confidence: str
    raw_thoughts: List[str]

# --- ReAct-style Agent --- #
class ReActConcallAgent:
    def __init__(self, openai_api_key: str, tavily_api_key: str):
        os.getenv("TAVILY_API_KEY") = tavily_api_key
        self.llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        self.search = TavilySearchResults(k=5)

        self.base_prompt = """You are a financial research assistant analyzing a company’s earnings conference call transcript.
Your task is to extract all critical information objectively, using only the transcript — no assumptions or outside data.

Follow this structured output format strictly:

1. Management Commentary:
2. Future Outlook and Guidance:
3. Industry and Macro Trends:
4. Competitive Landscape / Peer Comparison:
5. Risks and Concerns:
6. Growth Drivers and Strategic Initiatives:
7. Product Mix and Portfolio Trends:
8. Financial Highlights:
9. Sentiment Analysis:
10. Final Summary:

Avoid speculation.
"""

    def fetch_transcript(self, ticker: str) -> str:
        """Use Tavily to search for a recent earnings call transcript."""
        query = f"{ticker} latest earnings conference call transcript site:trendlyne.com OR site:moneycontrol.com OR site:investorrelations.com"
        results = self.search.run(query)

        for r in results:
            url = r.get("url")
            if url and any(kw in url for kw in ["transcript", "earnings", "conference-call", "concall"]):
                try:
                    html = requests.get(url, timeout=10).text
                    # Naive way to strip tags. Replace with BeautifulSoup if needed.
                    plain = html.replace("<br>", "\n").replace("<p>", "\n")
                    return plain[:10000]  # truncate long transcripts
                except Exception as e:
                    continue
        return ""

    def run(self, ticker: str) -> ConcallInsight:
        transcript = self.fetch_transcript(ticker)

        if not transcript:
            return ConcallInsight(
                ticker=ticker,
                summary="Transcript could not be retrieved from public sources.",
                sentiment="Unknown",
                confidence="Unknown",
                raw_thoughts=[]
            )

        full_prompt = f"{self.base_prompt}\n\nCompany Ticker: {ticker}\n\nTranscript:\n{transcript}\n\nBegin your structured analysis:"
        response = self.llm.invoke(full_prompt).content.strip()

        # Extract sentiment + confidence
        sentiment = "Unknown"
        confidence = "Unknown"
        if "Sentiment Analysis:" in response:
            sent_block = response.split("Sentiment Analysis:")[1]
            if "Positive" in sent_block: sentiment = "Positive"
            elif "Negative" in sent_block: sentiment = "Negative"
            elif "Neutral" in sent_block: sentiment = "Neutral"

            if "High" in sent_block: confidence = "High"
            elif "Moderate" in sent_block: confidence = "Moderate"
            elif "Low" in sent_block: confidence = "Low"

        return ConcallInsight(
            ticker=ticker,
            summary=response,
            sentiment=sentiment,
            confidence=confidence,
            raw_thoughts=[response]
        )
