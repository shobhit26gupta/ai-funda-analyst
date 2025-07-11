import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from typing import List
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

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

# --- Ratio Agent (uses Moneycontrol) --- #
class ReActRatioAgent:
    def __init__(self):
        openai_key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(api_key=openai_key, model_name="gpt-4", temperature=0)

    def fetch_moneycontrol_ratios(self, slug: str, code: str) -> pd.DataFrame:
        """
        Scrape ratios table from Moneycontrol using soup instead of read_html.
        """
        url = f"https://www.moneycontrol.com/financials/{slug}/ratiosVI/{code}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        table = soup.find("table")
        if not table:
            raise ValueError("❌ No table found in the page.")

        rows = table.find_all("tr")
        data = []

        for row in rows:
            cols = [col.get_text(strip=True) for col in row.find_all(["td", "th"])]
            if cols:
                data.append(cols)

        df = pd.DataFrame(data[1:], columns=data[0])
        return df

    def extract_relevant_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Keep only ROE and ROCE rows from the table.
        """
        df.columns = df.columns.str.replace("\n", " ").str.strip()
        df = df.rename(columns=lambda x: x.strip())

        df = df[df.iloc[:, 0].str.contains("Return on Equity|ROCE", case=False, na=False)]
        df = df.set_index(df.columns[0])
        return df

    def run(self, ticker: str) -> RatioReport:
        # Hardcoded map for known companies
        ticker_map = {
            "INFY": ("infosys", "IT"),
            "TCS": ("tataconsultancyservices", "ITE"),
            "HDFC": ("hdfcbank", "BF05"),
            "ADANIPORTS": ("adanienterprises", "AE17"),
        }

        if ticker not in ticker_map:
            raise ValueError(f"Ticker {ticker} not supported in hardcoded Moneycontrol map.")

        slug, code = ticker_map[ticker]

        # Step 1: Scrape and filter ratios
        df = self.fetch_moneycontrol_ratios(slug, code)
        ratio_df = self.extract_relevant_ratios(df)

        # Step 2: Prepare summary input text for LLM
        summary_text = f"ROE and ROCE data for {ticker} (FY21–FY25):\n"
        for year in ["Mar'21", "Mar'22", "Mar'23", "Mar'24", "Mar'25"]:
            try:
                roe = ratio_df.loc["Return on Equity / Networth", year]
                roce = ratio_df.loc["ROCE (%)", year]
                summary_text += f"FY{year[-2:]}: ROE = {roe}, ROCE = {roce}\n"
            except Exception:
                continue

        prompt = (
            "You are an expert financial analyst who specialises in financial statement analysis.\n"
            f"{summary_text}\n\n"
            "Analyse the ROE and ROCE of the company and do a detailed Du Pont analysis in such a way "
            "that a 15-year-old kid can understand. Break down what’s driving ROE and ROCE clearly. "
            "Use plain English and avoid jargon."
        )

        # Step 3: Ask LLM to explain
        response = self.llm.invoke(prompt).content.strip()

        # Step 4: Build output
        dummy_component = DupontComponent(
            year="FY25",
            roe=0.0,
            roe_explanation=response,
            roce=0.0,
            roce_explanation=response
        )

        return RatioReport(
            ticker=ticker,
            dupont_breakdown=[dummy_component],
            final_summary=response
        )