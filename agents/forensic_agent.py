# src/agents/forensic_agent.py

import os
import yfinance as yf
import pandas as pd
from typing import List
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

# --- Output models --- #
class Finding(BaseModel):
    name: str
    severity: str
    detail: str

class Report(BaseModel):
    ticker: str
    findings: List[Finding]
    final_answer: str

# --- ReAct Agent --- #
class ReActForensicAgent:
    def __init__(self, openai_api_key: str, tavily_api_key: str):
        self.llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        os.environ["TAVILY_API_KEY"] = tavily_api_key
        self.search = TavilySearchResults(k=5)

        self.base_prompt = """You are a forensic accounting expert who specialises in detecting accounting fraud, and
earnings manipulation. use the latest annual report for you to do thorough analysis (use consolidated financials).

Your goal is to conduct a detailed forensic check, examining the financial statements and disclosures for potential red flags that might indicate red flags in accounting, fraudulent accounting and earnings manipulation. Please make your analysis as follows:

1.⁠ ⁠Brief Summary 
Pointers on your key findings and a checklist based on green/yellow/red indicating accounting quality.

2.⁠ ⁠Revenue Recognition Analysis 
Aggressive revenue recognition practices, working capital analysis, channel stuffing.

3. Cash Flow Discrepancies 
Compare the cash flows with PAT and EBITDA, and see if earnings are being converted into cash flows or not, triangulate with debt on the balance sheet.

4. Related Party Transactions 
Provide details of all major RTPs, and highlight RTPs that seem suspicious, check for loans given to related parties.

5.⁠ ⁠Do a thorough check of the Balance Sheet 
Any write offs of assets or equity, inventory concerns, receivables aging, loans to other parties.

6.⁠ ⁠⁠Check the contingent liabilities and compare it with Net-worth 
If contingent liabilities size is 10% higher than the net-worth, flag it as a major red flag, also flag any major court cases/litigations.

7.⁠ ⁠⁠Do a check on Miscellaneous expenses and flag out any expenses which seem suspicious. 
See how much Miscellaneous expenses are as a % of sales. If it is higher than 3%, flag it out.

8.⁠ ⁠Management Discussion Analysis
Are any inconsistencies related to the guidance and financial metrics, or any other warning signs related to slowing growth?

9. Check the Auditors report and read the CARO report carefully. 
Check the Key Audit matters and see if the Auditor's opinion is Qualified or Unqualified. Assign a Green/Yellow/Red flag on the basis of Auditors Observations.

For each identified red flag, please:
•⁠ ⁠Quote the specific section, page number, and language from the annual report
•⁠ ⁠Explain why this represents a potential concern
•⁠ ⁠Quantify the financial impact where possible
•⁠ ⁠Make a checklist with Green (indicating clean), Yellow (indicating amber), Red (Indicating red) on all the points given above. Give the company a score in the end on the basis of 
Accounting Quality: Good, Average, Bad.

Avoid speculation without evidence. Your analysis should solely rely on information contained within the document.

Use precise accounting terminology while making your explanations accessible. Simply to understand any potential red flags. Signal out the pints which require further investigation. 

The idea is to understand the accounting quality and not to label everything as a fraud.

If you need any clarifications or require additional context to complete your analysis, please ask specific questions.

I will reward you if you do the task well.

"""
    
    def run(self, ticker: str) -> Report:
        # Load financials
        stock = yf.Ticker(ticker)
        fin = stock.financials
        cf = stock.cashflow

        # Simple data formatting
        fin = fin.T.iloc[-2:].fillna(0)
        cf = cf.T.iloc[-2:].fillna(0)

        reaccumulated = []
        conversation = []

        # Start sequence
        prompt = f"{self.base_prompt}\nCompany ticker: {ticker}"

        while True:
            response = self.llm.invoke(prompt).content.strip()
            conversation.append(response)

            # Parse latest Thought or Action
            if "Action: search_news" in response:
                # Extract query
                query = response.split("Action: search_news(")[1].split(")")[0]
                observation = self.search.run(query)
                prompt = response + f"\nObservation: {observation}\nThought:"
                continue

            if "Final Answer:" in response:
                final_answer = response.split("Final Answer:")[1].strip()
                break

            # No explicit tool called — let agent decide next
            prompt = response + "\nThought:"

        # Build findings from conversation – simple heuristic
        findings = []
        for msg in conversation:
            if "red flag" in msg.lower():
                findings.append(Finding(
                    name="Detected red flag",
                    severity="medium",
                    detail=msg[:200]
                ))

        if not findings:
            findings.append(Finding(
                name="No obvious red flags",
                severity="low",
                detail="No issues identified in conversation."
            ))

        return Report(ticker=ticker, findings=findings, final_answer=final_answer)
