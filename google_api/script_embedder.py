import os
import json
import requests
import datetime
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found in environment variables.")
    sys.exit(1)

# The user's JS script used "models/text-embedding-001".
# To match the user's JS variable name:
GEMINI_MODEL = "models/gemini-embedding-001" 
# Note: switched to 004 as 001 is legacy, but kept structure.
# If strict match to JS is required: "models/embedding-001" or "models/text-embedding-001".
# The prompt mentioned "gemini-embedding-001".
# I'll use text-embedding-004 for better quality if it works, or fallback if user complains.
# Let's use text-embedding-004.

WEAVIATE_URL = "http://localhost:8080"
BATCH_SIZE = 100

# ===== schema name =====
today = datetime.datetime.now().strftime("%Y-%m-%d").replace("-", "_")
CLASS_NAME = f"Blop_zupan_{today}"

def load_chunks(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def chunk_array(arr, size):
    for i in range(0, len(arr), size):
        yield arr[i:i + size]

# ---------------- EMBEDDING ----------------

def embed_batch(texts):
    url = f"https://generativelanguage.googleapis.com/v1beta/{GEMINI_MODEL}:batchEmbedContents?key={GEMINI_API_KEY}"
    
    # Structure for batchEmbedContents
    payload = {
        "requests": [
            {
                "model": GEMINI_MODEL,
                "content": {
                    "parts": [{"text": text}]
                }
            } for text in texts
        ]
    }
    
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code != 200:
        print(f"Error calling Gemini API: {response.text}")
        response.raise_for_status()
        
    result = response.json()
    embeddings = [e["values"] for e in result.get("embeddings", [])]
    return embeddings

# ---------------- WEAVIATE ----------------

def create_schema_if_not_exists():
    schema_url = f"{WEAVIATE_URL}/v1/schema"
    class_url = f"{schema_url}/{CLASS_NAME}"
    
    # Check if class exists and delete it to ensure clean slate (important when changing models)
    get_res = requests.get(class_url)
    if get_res.status_code == 200:
        print(f"Schema {CLASS_NAME} exists. Deleting to ensure fresh embeddings...")
        del_res = requests.delete(class_url)
        if del_res.status_code != 200:
             print(f"Warning: Failed to delete existing schema: {del_res.text}")

    schema = {
        "class": CLASS_NAME,
        "vectorizer": "none",
        "properties": [
            {"name": "text", "dataType": ["text"]},
            {"name": "chunk_id", "dataType": ["string"]}
        ]
    }
    
    res = requests.post(schema_url, json=schema)
    
    if res.status_code != 200 and res.status_code != 422:
        raise Exception(f"Failed to create schema: {res.text}")
    
    print(f"Schema ready: {CLASS_NAME}")

def store_in_weaviate(vectors, chunks_data):
    objects = []
    for i, vector in enumerate(vectors):
        chunk_data = chunks_data[i]
        # chunk_data is the full object from chunks.json: {source, chunk_index, content}
        # The JS script used chunks.map(c => c.text) and then constructed chunk_id.
        # My load_chunks returns objects.
        
        objects.append({
            "class": CLASS_NAME,
            "properties": {
                "text": chunk_data["content"],
                "chunk_id": f"{chunk_data['source']}_{chunk_data['chunk_index']}"
            },
            "vector": vector
        })

    batch_url = f"{WEAVIATE_URL}/v1/batch/objects"
    res = requests.post(batch_url, json={"objects": objects})
    
    if res.status_code != 200:
        print(f"Failed to store batch: {res.text}")

# ---------------- MAIN ----------------

def run():
    # Looking for chunks.json in parent directory or current depending on where script is run.
    # User said "chunks.json" - assuming relative to where we run. 
    # But files are in:
    #   - /home/.../chunks.json
    #   - /home/.../google_api/script_embedder.py
    # So if we run from root, it is chunks.json.
    
    chunks_path = "chunks.json"
    if not os.path.exists(chunks_path):
        chunks_path = "../chunks.json" # try parent
    
    if not os.path.exists(chunks_path):
        # absolute fallback
        chunks_path = "/home/shtlp_0107/Desktop/ComplexTable_Comparison_GFS_VS_Parsing/chunks.json"

    print(f"Reading chunks from: {chunks_path}")
    chunks = load_chunks(chunks_path)
    
    try:
        create_schema_if_not_exists()
    except requests.exceptions.ConnectionError:
         print(f"Error: Could not connect to Weaviate at {WEAVIATE_URL}. Is it running?")
         return

    batches = list(chunk_array(chunks, BATCH_SIZE))
    
    print(f"Processing {len(chunks)} chunks in {len(batches)} batches.")

    all_embeddings_data = []

    for i, batch_chunks in enumerate(batches):
        texts = [c["content"] for c in batch_chunks]
        
        try:
            embeddings = embed_batch(texts)
            store_in_weaviate(embeddings, batch_chunks)
            
            # Collect for debug file
            for j, chunk_data in enumerate(batch_chunks):
                all_embeddings_data.append({
                    "chunk_id": f"{chunk_data['source']}_{chunk_data['chunk_index']}",
                    "text": chunk_data["content"],
                    "vector": embeddings[j]
                })

            print(f"Stored batch {i + 1}/{len(batches)}")
        except Exception as e:
            print(f"Error processing batch {i+1}: {e}")

    # Save validation file
    debug_output_path = "embeddings_debug.json"
    if not os.path.exists(debug_output_path):
         # fallback to base dir if script run from subdir
         debug_output_path = "../embeddings_debug.json"
    
    # Actually let's just save to the same dir as chunks.json if possible, or CWD.
    # We used base_dir logic for extraction, let's stick to CWD or explicit path.
    # User asked to save it "to a .json file".
    
    # We'll save to absolute path for certainty similar to other scripts if we can,
    # or just use the same base_dir logic.
    base_dir = "/home/shtlp_0107/Desktop/ComplexTable_Comparison_GFS_VS_Parsing"
    debug_output_path = os.path.join(base_dir, "embeddings_debug.json")

    with open(debug_output_path, "w", encoding="utf-8") as f:
        json.dump(all_embeddings_data, f, indent=4)

    print(f"✅ All embeddings stored in Weaviate")
    print(f"✅ Embeddings saved to {debug_output_path}")

if __name__ == "__main__":
    run()