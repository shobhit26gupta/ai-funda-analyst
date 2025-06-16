# src/agents/ratio_agent.py

import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from typing import List, Dict
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

# --- Output models --- #
class DupontComponent(BaseModel):
    year: str
    roe: float
    roe_explanation: str
    roce: float
    roce_explanation: str

class RatioReport(BaseModel):
    ticker: str
    dupont_breakdown: List[DupontComponent]
    final_summary: str

# --- ReAct-style Ratio Analysis Agent --- #
class ReActRatioAgent:
    def __init__(self, openai_api_key: str, tavily_api_key: str):
        os.environ["TAVILY_API_KEY"] = tavily_api_key
        self.llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        self.search = TavilySearchResults(k=5)

        # System instructions for Du Pont analysis
        self.base_prompt = (
            "You are an expert financial analyst who specialises in financial statement analysis.\n"
            "Use the following Link for data: https://www.screener.in/company/{ticker}/consolidated/\n"
            "Analyse the ROE of the company and do a detailed Du Pont analysis in such a way "
            "that a 15 year old kid can understand. Break down the ROE, ROCE, and simplify what "
            "is driving the ROE, ROCE out of all the 3 parts of Du Pont. USE data between FY21–FY25 "
            "(treat March 2025 as FY25). Keep it simple and clear. I will reward you if you do well."
        )

    def fetch_screener_data(self, ticker: str) -> pd.DataFrame:
        """Scrape consolidated financials table from Screener."""
        search_query = f"{ticker} consolidated Screener.in"
        results = self.search.run(search_query)

        for r in results:
            url = r.get("url", "")
            if "screener.in/company" in url and "consolidated" in url:
                resp = requests.get(url, timeout=10)
                soup = BeautifulSoup(resp.text, "html.parser")
                table = soup.find("table", {"class": "data-table"})
                df = pd.read_html(str(table))[0]
                return df  # returns raw financials

        raise ValueError("Could not fetch Screener data for " + ticker)

    def run(self, ticker: str) -> RatioReport:
        # 1️⃣ Scrape data from Screener
        df = self.fetch_screener_data(ticker)

        # 2️⃣ Inject raw data into prompt (limit to FY21–FY25)
        html_data = df.to_html(index=False)
        full_prompt = self.base_prompt.format(ticker=ticker) + "\n\n" + html_data

        # 3️⃣ Ask GPT-4 to do the ratio analysis
        response = self.llm.invoke(full_prompt).content.strip()

        # 4️⃣ Parse the response into structured components
        dupont_breakdown = []
        for year in ["FY21", "FY22", "FY23", "FY24", "FY25"]:
            # Basic parsing – look for "FY21" block in text
            if year in response:
                desc = response.split(year)[1].split("FY" if year != "FY25" else "")[0].strip()
                # Optionally, use regex or smarter parsing here
                dupont_breakdown.append(DupontComponent(
                    year=year,
                    roe=0.0,  # You could parse the numeric value if present
                    roe_explanation=desc,
                    roce=0.0,
                    roce_explanation=desc,
                ))

        return RatioReport(
            ticker=ticker,
            dupont_breakdown=dupont_breakdown,
            final_summary=response
        )
