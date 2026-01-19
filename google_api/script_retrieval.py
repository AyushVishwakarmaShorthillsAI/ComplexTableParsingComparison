import os
import json
import requests
import re
import ast
import sys
import time
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found in environment variables.")
    sys.exit(1)

GEMINI_MODEL = "models/gemini-embedding-001"
WEAVIATE_URL = "http://localhost:8080"
TOP_K = 5

# Hardcoded queries as requested
QUERIES = [
    "Common features in people suffering from KPTN Syndrome and their percentage",
    "List of the most common facial and developmental features exhibited by individuals with KPTN Syndrome with percentages",
    "Uncommon features in people with KPTN Syndrome and their percentage"
]

def get_latest_schema_class():
    """Finds the most recent Blop_zupan schema class."""
    try:
        res = requests.get(f"{WEAVIATE_URL}/v1/schema")
        if res.status_code != 200:
            print(f"Error fetching schema: {res.text}")
            return None
        
        data = res.json()
        classes = data.get("classes", [])
        
        # Filter for our specific classes
        zupan_classes = [c["class"] for c in classes if c["class"].startswith("KPTN_syndrome_")]
        
        if not zupan_classes:
            return None
            
        # Sort by name (which acts as date sort due to YYYY_MM_DD format) and get the latest
        zupan_classes.sort(reverse=True)
        return zupan_classes[0]
        
    except Exception as e:
        print(f"Error connecting to Weaviate: {e}")
        return None

def embed_content(text):
    """Embeds a single query string."""
    url = f"https://generativelanguage.googleapis.com/v1beta/{GEMINI_MODEL}:embedContent?key={GEMINI_API_KEY}"
    
    payload = {
        "model": GEMINI_MODEL,
        "content": {
            "parts": [{"text": text}]
        }
    }
    
    headers = {"Content-Type": "application/json"}
    
    max_retries = 5
    for attempt in range(max_retries):
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            return result.get("embedding", {}).get("values")
        elif response.status_code == 429:
            retry_after = 5 * (attempt + 1)
            print(f"Rate limit hit. Retrying in {retry_after}s...")
            time.sleep(retry_after)
        else:
            print(f"Error embedding text: {response.text}")
            return None
            
    print("Max retries exceeded for embedding query.")
    return None

def search_weaviate(class_name, vector):
    """Searches Weaviate for the vector."""
    query = """
    {
      Get {
        %s(
          nearVector: {
            vector: %s
          }
          limit: %d
        ) {
          text
          chunk_id
          _additional {
            distance
          }
        }
      }
    }
    """ % (class_name, json.dumps(vector), TOP_K)

    headers = {"Content-Type": "application/json"}
    response = requests.post(f"{WEAVIATE_URL}/v1/graphql", headers=headers, json={"query": query})
    
    if response.status_code != 200:
        print(f"Error searching Weaviate: {response.text}")
        return []
        
    data = response.json()
    
    # Safely navigate the response
    try:
        results = data.get("data", {}).get("Get", {}).get(class_name)
        # Ensure we return a list, as Weaviate might return None if no matches found in some versions
        return results if results is not None else []
    except (KeyError, TypeError) as e:
        print(f"Unexpected response format or error: {e}")
        print(f"Full response: {data}")
        return []

def main():
    # 1. Identify Class Name
    class_name = get_latest_schema_class()
    if not class_name:
        print("❌ Could not find a 'KPTN_syndrome_' class in Weaviate. Did you run the embedder?")
        return
    print(f"🔎 Using Weaviate Class: {class_name}")

    print(f"📝 Processing {len(QUERIES)} specific queries.")

    results = []

    # 2. Process Queries
    for i, query in enumerate(QUERIES):
        print(f"[{i+1}/{len(QUERIES)}] Processing: {query[:50]}...")
        
        vector = embed_content(query)
        if not vector:
            print(f"Skipping query due to embedding failure: {query[:30]}")
            continue
            
        hits = search_weaviate(class_name, vector)
        
        # Format results
        retrieved_chunks = []
        for hit in hits:
            retrieved_chunks.append({
                "chunk_id": hit.get("chunk_id"),
                "text": hit.get("text"),
                "distance": hit.get("_additional", {}).get("distance")
            })
            
        results.append({
            "query": query,
            "retrieved_chunks": retrieved_chunks
        })

    # 3. Save Results
    output_file = "res_KPTN.json"
    
    base_dir = "/home/shtlp_0107/Desktop/ComplexTable_Comparison_GFS_VS_Parsing"
    output_path = os.path.join(base_dir, output_file)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print(f"✅ Saved results to {output_path}")

if __name__ == "__main__":
    main()
