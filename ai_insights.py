import os
from dotenv import load_dotenv
from google import genai

load_dotenv("llm_integration.env")

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

def generate_insights(
    campaign_budget,
    spend,
    impressions,
    clicks,
    campaign_type,
    platform,
    prediction
):
    ctr = (clicks / impressions) * 100 if impressions else 0
    cpc = spend / clicks if clicks else None
    budget_utilization = (spend / campaign_budget) * 100 if campaign_budget else 0
    roi = ((prediction - spend) / spend) * 100 if spend else None
    roas = prediction / spend if spend else None

    prompt = f"""
You are a senior digital marketing analyst reviewing a single ad campaign.

CAMPAIGN DATA:

Campaign Budget: ${campaign_budget:.2f}
Spend: ${spend:.2f}
Impressions: {impressions}
Clicks: {clicks}
Conversions: Not Available
Predicted Revenue: ${prediction:.2f}
Campaign Type: {campaign_type}
Platform: {platform}
Do not confuse ROI and ROAS.

If a metric cannot be calculated because data is missing (such as conversions), clearly state that instead of estimating it.

Recommendations must be data-driven and reference the campaign metrics.

Using this data, produce a report with exactly these 6 sections, in this order:

1. Performance Summary — 2 sentences, plain language, no jargon.
2. ROI Analysis — show your work. Calculate ROI as (Revenue − Spend) / Spend, and also report ROAS (Revenue / Spend) and CPA (Spend / Conversions). State each formula before the number.
3. Campaign Strengths — 2 bullet points max, each citing a specific metric.
4. Campaign Weaknesses — 2 bullet points max, each citing a specific metric.
5. Recommendations — exactly 3, each one sentence, each tied to a weakness above.
6. Final Verdict — one word only: Excellent / Good / Average / Poor. Follow with a 1-sentence justification.

Rules:
- If any required metric is missing from the data, state which one is missing instead of estimating it.
- Total response: 250–300 words.
- Use bullet points only in sections 3, 4, and 5.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text