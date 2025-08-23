import os
import json
import asyncio
import httpx
import uuid
from typing import Optional, List
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from collections import Counter

# --- Environment Setup ---
load_dotenv()
DB_FILE = "analysis_log.json"

# --- Pydantic Models ---
class Claim(BaseModel):
    claim: str
    status: str

class AnalysisInput(BaseModel):
    text: Optional[str] = None
    image_url: Optional[str] = None
    source: str

class CounterInput(BaseModel):
    claims: List[Claim]

class AnalysisResult(BaseModel):
    id: str
    source: str
    score: int
    summary: str
    claims: List[Claim]
    timestamp: str
    coach_feedback: Optional[str] = None

# --- FastAPI App Initialization ---
app = FastAPI()
origins = ["http://localhost", "http://localhost:8080", "null"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Gemini API Configuration ---
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: raise ValueError("GEMINI_API_KEY not found.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    model = None

safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# --- Database Functions ---
def save_analysis(result: AnalysisResult):
    try:
        results = []
        if os.path.exists(DB_FILE):
            with open(DB_FILE, 'r') as f: results = json.load(f)
        results.insert(0, result.dict())
        with open(DB_FILE, 'w') as f: json.dump(results[:50], f, indent=4)
    except Exception as e: print(f"Error saving analysis: {e}")

def load_recent_analyses() -> List[AnalysisResult]:
    if not os.path.exists(DB_FILE): return []
    try:
        with open(DB_FILE, 'r') as f: return [AnalysisResult(**item) for item in json.load(f)[:20]]
    except Exception as e:
        print(f"Error loading analyses: {e}")
        return []

# --- AI Functions ---
async def extract_text_from_image(image_url: str):
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(image_url, follow_redirects=True, timeout=15)
            r.raise_for_status()
        image_part = {"mime_type": r.headers.get("Content-Type", "image/jpeg"), "data": r.content}
        response = await model.generate_content_async(["Extract all text from this image.", image_part], safety_settings=safety_settings)
        return response.text.strip()
    except Exception as e: raise HTTPException(status_code=500, detail=f"AI model failed to process image: {e}")

async def get_analysis_from_text(text: str):
    extracted_claims = []
    # If the text is short, treat the whole thing as a single claim.
    if len(text.split()) < 15:
        print("--- Short text detected, treating as a single claim. ---")
        extracted_claims = [text]
    else:
        # For longer text, use the AI to extract claims.
        claims_prompt = f'Extract up to 3 verifiable factual claims from this text. Your response MUST be a valid JSON array of strings. Text: "{text}"'
        try:
            claims_response = await model.generate_content_async(claims_prompt, safety_settings=safety_settings)
            extracted_claims = json.loads(claims_response.text.strip().replace("```json", "").replace("```", ""))
        except Exception as e:
            print(f"Error parsing claims from long text: {e}")
            extracted_claims = []

    if not extracted_claims: return 50, "Could not identify verifiable claims.", [], None

    async def fact_check_claim(claim):
        prompt = f'Is this claim "Supported", "Refuted", or "Unverifiable"? Your response MUST be a single word. Claim: "{claim}"'
        response = await model.generate_content_async(prompt, safety_settings=safety_settings)
        status = response.text.strip().replace(".", "")
        return {"claim": claim, "status": status if status in ["Supported", "Refuted", "Unverifiable"] else "Unverifiable"}

    tasks = [fact_check_claim(claim) for claim in extracted_claims]
    results = await asyncio.gather(*tasks)
    
    total_score = sum(100 if r['status'] == 'Supported' else 50 if r['status'] == 'Unverifiable' else 0 for r in results)
    final_score = int(total_score / len(results)) if results else 50

    coach_feedback = None
    if final_score < 70:
        coach_prompt = f"""
        Act as an AI Coach specializing in media literacy. Based on the following analysis, provide a short, helpful tip for the user.
        Explain a common misinformation technique present in the claims and advise on how to spot it in the future.
        Keep the feedback to 2-3 sentences.

        Analysis:
        - Score: {final_score}
        - Claims: {results}
        """
        try:
            coach_response = await model.generate_content_async(coach_prompt, safety_settings=safety_settings)
            coach_feedback = coach_response.text.strip()
        except Exception as e:
            print(f"Error generating coach feedback: {e}")

    summary_prompt = f"Write a one-sentence summary for a credibility score of {final_score}/100 based on these claims: {results}"
    summary_response = await model.generate_content_async(summary_prompt, safety_settings=safety_settings)
    summary = summary_response.text.strip()
    
    return final_score, summary, results, coach_feedback

# --- API Endpoints ---
@app.post("/analyze", response_model=AnalysisResult)
async def analyze_content(payload: AnalysisInput):
    if not model: raise HTTPException(status_code=500, detail="Gemini API not configured.")
    text_to_analyze = payload.text or ""
    if payload.image_url:
        text_from_image = await extract_text_from_image(payload.image_url)
        if not text_from_image:
            result = AnalysisResult(id=str(uuid.uuid4()), source=payload.source, score=50, summary="No text found in image.", claims=[], timestamp=datetime.utcnow().isoformat())
            save_analysis(result)
            return result
        text_to_analyze = text_from_image
    if not text_to_analyze: raise HTTPException(status_code=400, detail="No content to analyze.")
    
    score, summary, claims, coach_feedback = await get_analysis_from_text(text_to_analyze)
    
    result = AnalysisResult(
        id=str(uuid.uuid4()),
        source=payload.source,
        score=score,
        summary=summary,
        claims=claims,
        timestamp=datetime.utcnow().isoformat(),
        coach_feedback=coach_feedback
    )
    save_analysis(result)
    return result

@app.get("/recent", response_model=List[AnalysisResult])
async def get_recent_analyses():
    return load_recent_analyses()

@app.post("/generate_counter")
async def generate_counter_narrative(payload: CounterInput):
    if not model: raise HTTPException(status_code=500, detail="Gemini API not configured.")
    refuted_claims = [c.claim for c in payload.claims if c.status == 'Refuted']
    if not refuted_claims: return {"narrative": "No refuted claims to counter."}
    
    formatted_claims = "\n- ".join(refuted_claims)
    prompt = f'Generate a short, polite, factual paragraph to counter these false claims:\n- {formatted_claims}'
    
    try:
        response = await model.generate_content_async(prompt, safety_settings=safety_settings)
        return {"narrative": response.text.strip()}
    except Exception as e: raise HTTPException(status_code=500, detail="Failed to generate counter-narrative.")

@app.get("/stats")
async def get_stats():
    if not os.path.exists(DB_FILE): return {"hot_topics": [], "propagation_graph": {"nodes": [], "links": []}}
    with open(DB_FILE, 'r') as f:
        all_analyses = json.load(f)

    refuted_claims_text = " ".join([claim['claim'] for analysis in all_analyses for claim in analysis['claims'] if claim['status'] == 'Refuted'])
    words = [word for word in refuted_claims_text.lower().split() if len(word) > 4 and word.isalpha()]
    hot_topics = [{"topic": topic, "count": count} for topic, count in Counter(words).most_common(5)]
    
    nodes, links, seen_nodes = [], [], set()
    for analysis in all_analyses:
        if analysis['score'] < 50:
            source_id = analysis['source']
            if source_id not in seen_nodes:
                nodes.append({"id": source_id, "type": "source"})
                seen_nodes.add(source_id)
            for claim in analysis['claims']:
                if claim['status'] == 'Refuted':
                    claim_id = claim['claim'][:50] + '...'
                    if claim_id not in seen_nodes:
                        nodes.append({"id": claim_id, "type": "claim"})
                        seen_nodes.add(claim_id)
                    links.append({"source": source_id, "target": claim_id})

    return {"hot_topics": hot_topics, "propagation_graph": {"nodes": nodes, "links": links}}
