# ai-funda-analyst

Component	What it does	User-visible outcome
🧠 Router	Interprets query intent and routes it to the relevant agents	Efficiently handles requests like “Give score,” “Show red flags,” “Summarize call”

🔍 Forensic Accounting Agent (ReAct)	Identifies signs of financial manipulation using Beneish M-Score, Benford's Law, promoter news, audit changes, accruals	Returns suspicious findings with LLM-generated explanation

📊 Ratio Analysis Agent (ReAct)	Computes 20+ financial ratios and benchmarks vs sector peers	Highlights strong/weak areas (e.g., low ROE, high D/E) with plain-English insights

🗣️ Con-call Insight Agent (ReAct)	Scrapes + processes earnings-call transcripts, analyzes tone, sentiment, risk language	Shows optimism/pessimism and highlights red-flag statements

📈 Scoring Engine	Combines 3 agents’ outputs into a 0–100 score using user-defined weights	Returns scorecard + reason

💬 Explainability Layer (LLM)	Summarizes what happened and why in natural language	Human-readable output

🌐 Open-source APIs	Supplies all the structured and unstructured data for analysis	Fast, cost-effective, no PDFs
