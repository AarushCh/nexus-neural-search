import os
import pandas as pd
import requests
import time
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# --- CONFIGURATION ---
CSV_FILE = "dataset.csv"  # Make sure this matches your file name
COLLECTION_NAME = "freeme_collection"
MODEL_ID = "BAAI/bge-small-en-v1.5"
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"
VECTOR_SIZE = 384
BATCH_SIZE = 50  # Upload 50 movies at a time to avoid timeouts

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

if not QDRANT_URL or not QDRANT_API_KEY or not HF_TOKEN:
    print("❌ Error: Missing credentials in .env file!")
    exit()

if QDRANT_URL.startswith("ttps://"): 
    QDRANT_URL = QDRANT_URL.replace("ttps://", "https://")

print(f"☁️ Connecting to Qdrant Cloud...")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# 1. Load CSV
if not os.path.exists(CSV_FILE):
    print(f"❌ Error: Could not find {CSV_FILE}")
    exit()

print(f"📖 Reading {CSV_FILE}...")
df = pd.read_csv(CSV_FILE)

# Fill missing values to avoid errors
df = df.fillna("")
print(f"📊 Found {len(df)} movies. Starting upload...")

# 2. Reset Collection (Optional: Uncomment to wipe old 10 movies)
# client.recreate_collection(
#     collection_name=COLLECTION_NAME,
#     vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
# )

# 3. Batch Upload
total_uploaded = 0
points_batch = []

for index, row in df.iterrows():
    # Construct text for embedding
    # Adjust column names 'title', 'overview', 'vote_average' to match YOUR CSV
    title = row.get('title', row.get('Title', 'Unknown'))
    desc = row.get('overview', row.get('Description', ''))
    rating = row.get('vote_average', row.get('Rating', 0))
    image = row.get('poster_path', row.get('Image', ''))
    media_type = row.get('media_type', 'MOVIE')

    # Quick cleanup for image URL if it's just a path
    if image and str(image).startswith("/"):
        image = f"https://image.tmdb.org/t/p/w500{image}"

    text = f"{title} {desc}"
    
    # --- GET EMBEDDING ---
    vector = None
    for attempt in range(3):
        try:
            response = requests.post(
                HF_API_URL, 
                headers={"Authorization": f"Bearer {HF_TOKEN}"}, 
                json={"inputs": [text]} 
            )
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    vector = data[0] if isinstance(data[0], list) else data
                break
            elif response.status_code == 503:
                time.sleep(2)
            else:
                break
        except:
            break
    
    # If embedding succeeded, add to batch
    if vector and len(vector) == VECTOR_SIZE:
        payload = {
            "title": title,
            "description": desc,
            "rating": float(rating) if rating else 0,
            "type": str(media_type).upper(),
            "image": str(image)
        }
        points_batch.append(PointStruct(id=index+1, vector=vector, payload=payload))
        print(f"   ✅ Prepared: {title}")
    else:
        print(f"   ⚠️ Skipped (Embedding Failed): {title}")

    # Upload when batch is full
    if len(points_batch) >= BATCH_SIZE:
        client.upsert(collection_name=COLLECTION_NAME, points=points_batch)
        total_uploaded += len(points_batch)
        print(f"🚀 Uploaded batch! Total: {total_uploaded}")
        points_batch = [] # Reset batch
        time.sleep(1) # Be nice to the API

# Upload any remaining
if points_batch:
    client.upsert(collection_name=COLLECTION_NAME, points=points_batch)
    total_uploaded += len(points_batch)

print(f"🎉 FINAL SUCCESS! Uploaded {total_uploaded} movies to the Cloud.")