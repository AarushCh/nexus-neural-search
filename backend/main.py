from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy.orm import Session
from backend.database import Base, engine, SessionLocal
from backend.models import User, WishlistItem
from backend.auth import get_current_user_db, login_user, hash_password
import requests
import json
import re
import os
import time
from dotenv import load_dotenv

# --- LOAD ENV VARIABLES ---
load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# 🚨 USING THE STANDARD STABLE ENDPOINT
HF_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "arcee-ai/trinity-large-preview:free")

app = FastAPI(title="FreeMe Engine (Robust API Mode)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)

# --- 🧠 ROBUST EMBEDDING FUNCTION ---
def get_embedding(text):
    """
    Retries up to 3 times if HuggingFace is 'Loading'.
    Prints EXACT error if it fails.
    """
    if not HF_TOKEN:
        print("❌ CRITICAL: HF_TOKEN is missing in Render Environment!")
        return [0.0] * 384

    payload = {"inputs": text, "options": {"wait_for_model": True}}
    
    for attempt in range(3): # Try 3 times
        try:
            response = requests.post(
                HF_API_URL,
                headers={"Authorization": f"Bearer {HF_TOKEN}"},
                json=payload,
                timeout=10
            )
            
            # 1. Success
            if response.status_code == 200:
                data = response.json()
                # Check if we got a list of floats (the vector)
                if isinstance(data, list) and isinstance(data[0], float):
                    return data
                # Sometimes it returns a list of list
                if isinstance(data, list) and isinstance(data[0], list):
                    return data[0]
            
            # 2. Model Loading (Wait and Retry)
            if response.status_code == 503:
                print(f"⚠️ Model Loading... Waiting {data.get('estimated_time', 2)}s")
                time.sleep(data.get('estimated_time', 2))
                continue
            
            # 3. Other Errors (Print them!)
            print(f"❌ HF API Error ({response.status_code}): {response.text}")
            break # Don't retry 400/401 errors
            
        except Exception as e:
            print(f"❌ Connection Error: {e}")
            break

    # If all retries fail, return Zero Vector (This causes the 'Saiki' bug)
    print("⚠️ FAILED to get embedding. Returning Zero Vector.")
    return [0.0] * 384

def get_qdrant():
    from qdrant_client import QdrantClient
    # Ensure URL is https://
    url = QDRANT_URL
    if url and url.startswith("ttps://"): url = url.replace("ttps://", "https://")
    return QdrantClient(url=url, api_key=QDRANT_API_KEY)

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

def safe_vector_search(vector, limit=50):
    try: 
        q_client = get_qdrant()
        return q_client.query_points(collection_name="freeme_collection", query=vector, limit=limit).points
    except Exception as e:
        print(f"Vector Search Error: {e}")
        return []

def get_llm_recommendations(query):
    try:
        if not OPENROUTER_API_KEY: return None
        
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "HTTP-Referer": "http://nexus-search.com"},
            data=json.dumps({
                "model": OPENROUTER_MODEL,
                "messages": [{"role": "user", "content": f"Recommend 10 movies strictly matching '{query}'. Return ONLY JSON list of strings."}]
            }), timeout=15
        )
        if resp.status_code != 200: return None
        content = resp.json()['choices'][0]['message']['content']
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if not match: return None
        titles = json.loads(match.group())
        
        results = []
        for t in titles:
            # THIS was failing before. Now it will print why.
            vector = get_embedding(str(t))
            hits = safe_vector_search(vector, limit=1)
            if hits:
                item = hits[0].payload
                item["id"] = hits[0].id
                item["score"] = 99
                results.append(item)
        return results
    except Exception as e:
        print(f"LLM Error: {e}")
        return None

class UserRequest(BaseModel): text: str; top_k: int = 12; model: str = "internal"
class PersonalizedRequest(BaseModel): text: str; top_k: int = 12; model: str = "internal"
class AuthRequest(BaseModel): username: str; email: str; password: str
class SimilarRequest(BaseModel): id: int

@app.get("/")
def health_check():
    return {"status": "online", "message": "Nexus Neural Engine v4.0 (Robust)"}

@app.post("/login")
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    return login_user(form, db)

@app.post("/signup")
def signup(data: AuthRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == data.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")
    user = User(username=data.username, email=data.email, hashed_password=hash_password(data.password))
    db.add(user)
    db.commit()
    return {"status": "created"}

@app.post("/recommend")
def recommend(req: UserRequest):
    if req.model == 'api': return get_llm_recommendations(req.text) or []
    
    vector = get_embedding(req.text)
    hits = safe_vector_search(vector, limit=50)
    
    results = []
    for h in hits:
        item = h.payload
        item["id"] = h.id
        item["score"] = int(h.score * 100) if h.score else 0 
        results.append(item)
        
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:req.top_k]

@app.post("/recommend/personalized")
def personalized(req: PersonalizedRequest, user=Depends(get_current_user_db), db: Session = Depends(get_db)):
    if req.model == 'api': return recommend(UserRequest(text=req.text, top_k=req.top_k, model='api'))
    
    q_client = get_qdrant()
    w_items = db.query(WishlistItem).filter_by(user_id=user.id).all()
    if not w_items: return recommend(UserRequest(text=req.text, top_k=req.top_k))
    
    try:
        points = q_client.retrieve("freeme_collection", ids=[i.media_id for i in w_items], with_vectors=True)
        vectors = [p.vector for p in points if p.vector]
        if not vectors: raise Exception
        avg_taste = [sum(col)/len(col) for col in zip(*vectors)]
        mood = get_embedding(req.text)
        final_vector = [0.2*t + 0.8*m for t, m in zip(avg_taste, mood)]
    except: 
        final_vector = get_embedding(req.text)
    
    hits = safe_vector_search(final_vector, limit=50)
    results = []
    for h in hits:
        item = h.payload
        item["id"] = h.id
        item["score"] = int(h.score * 100) if h.score else 0
        results.append(item)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:req.top_k]

@app.post("/similar")
def similar(req: SimilarRequest):
    q_client = get_qdrant()
    tgt = q_client.retrieve("freeme_collection", ids=[req.id], with_vectors=True)
    if not tgt: return []
    hits = safe_vector_search(tgt[0].vector, limit=15)
    return [{**h.payload, 'id': h.id, 'score': 95} for h in hits if h.id!=req.id][:4]

@app.post("/wishlist/add/{mid}")
def add_w(mid: int, u=Depends(get_current_user_db), db: Session = Depends(get_db)):
    if not db.query(WishlistItem).filter_by(user_id=u.id, media_id=mid).first():
        db.add(WishlistItem(user_id=u.id, media_id=mid))
        db.commit()
    return {"status": "ok"}

@app.delete("/wishlist/remove/{mid}")
def rem_w(mid: int, u=Depends(get_current_user_db), db: Session = Depends(get_db)):
    db.query(WishlistItem).filter_by(user_id=u.id, media_id=mid).delete()
    db.commit()
    return {"status": "ok"}

@app.get("/wishlist")
def get_w(u=Depends(get_current_user_db), db: Session = Depends(get_db)):
    q_client = get_qdrant()
    ids = [i.media_id for i in db.query(WishlistItem).filter_by(user_id=u.id).all()]
    if not ids: return []
    try: return [{**p.payload, 'id': p.id} for p in q_client.retrieve("freeme_collection", ids=ids)]
    except: return []

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=10000, reload=True)