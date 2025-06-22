# src/router/router.py

import os
import json
import re
from typing import List
from openai import OpenAIError
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

class RouterAgent:
    def __init__(self):
        openai_key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0,api_key=openai_key)
        
    def route(self, user_query: str) -> dict:
        system_prompt = f"""
You are a query routing assistant for a financial analysis system.

Your task is to classify the user query into one or more of the following agent categories:
1. "FORENSIC_AGENT": Use this when the query is about red flags, manipulation, fraud detection, promoter actions, audit concerns, forensic accounting checks.
2. "RATIO_AGENT": Use this for queries about financial ratios, ROE, ROCE, Du Pont analysis, performance benchmarking, or company financial strength.
3. "CONCALL_AGENT": Use this if the query is about management tone, sentiment, earnings call insights, forward guidance, risks, or communication trends.

You may return multiple agents if the query spans more than one domain (e.g., "give full score" may include all agents).

Always respond in this JSON format:
{{
    "agents": ["FORENSIC_AGENT", "RATIO_AGENT", "CONCALL_AGENT"],
    "reason": "Brief explanation of your routing choice"
}}

Respond only in JSON, without additional explanation or comments.

User: {user_query}
"""

        try:
            response = self.llm.invoke(system_prompt).content
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            json_text = json_match.group()
            parsed = json.loads(json_text)
            return parsed

        except OpenAIError as api_err:
            return {
                "agents": ["FORENSIC_AGENT"],
                "reason": f"OpenAI API error: {api_err}"
            }

        except json.JSONDecodeError as json_err:
            return {
                "agents": ["FORENSIC_AGENT"],
                "reason": f"JSON parsing error: {json_err}"
            }

        except Exception as err:
            return {
                "agents": ["FORENSIC_AGENT"],
                "reason": f"Unexpected error: {err}"
            }
