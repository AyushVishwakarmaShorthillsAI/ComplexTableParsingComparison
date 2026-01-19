import os
import json
import requests
import sys
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

load_dotenv()

# ======================
# CONFIG
# ======================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found.")
    sys.exit(1)

GEMINI_MODEL = "models/gemini-embedding-001"
WEAVIATE_URL = "http://localhost:8080"

TOP_K = 15        # recall
TOP_RERANK = 5    # precision

# ======================
# QUERIES
# ======================

QUERIES = [
    "According to Table 1, what is the DMFT score (Decayed, Missing, Filled Teeth) for patient 5?",
    "Which patient in Table 1 has the mutation c.1954C>T?",
    "What type of 'Structure' anomaly regarding enamel is listed for Patient 16 in Table 1?",
    "In Table 2, what is the specific 'Facial axis' measurement recorded for Patient 16?",
    "According to Table 1, does Patient 1 exhibit dental crowding?"
]

# ======================
# LOAD RERANKER
# ======================

print("🔁 Loading reranker model...")
reranker = CrossEncoder("BAAI/bge-reranker-base")
print("✅ Reranker loaded")


# ======================
# UTILS
# ======================

def get_latest_schema_class():
    try:
        res = requests.get(f"{WEAVIATE_URL}/v1/schema")
        data = res.json()
        classes = data.get("classes", [])

        zupan = [c["class"] for c in classes if c["class"].startswith("Blop_zupan_")]
        if not zupan:
            return None

        zupan.sort(reverse=True)
        return zupan[0]

    except Exception as e:
        print("Schema error:", e)
        return None


def embed_content(text):
    url = f"https://generativelanguage.googleapis.com/v1beta/{GEMINI_MODEL}:embedContent?key={GEMINI_API_KEY}"

    payload = {
        "model": GEMINI_MODEL,
        "content": {
            "parts": [{"text": text}]
        }
    }

    r = requests.post(url, json=payload)
    if r.status_code != 200:
        print("Embedding error:", r.text)
        return None

    return r.json().get("embedding", {}).get("values")


def search_weaviate(class_name, vector):
    query = f"""
    {{
      Get {{
        {class_name}(
          nearVector: {{
            vector: {json.dumps(vector)}
          }}
          limit: {TOP_K}
        ) {{
          text
          chunk_id
          _additional {{
            distance
          }}
        }}
      }}
    }}
    """

    r = requests.post(
        f"{WEAVIATE_URL}/v1/graphql",
        json={"query": query},
        headers={"Content-Type": "application/json"}
    )

    if r.status_code != 200:
        print("Weaviate error:", r.text)
        return []

    return r.json().get("data", {}).get("Get", {}).get(class_name, [])


def rerank_chunks(query, chunks):
    pairs = [(query, c["text"]) for c in chunks]
    scores = reranker.predict(pairs)

    for i, score in enumerate(scores):
        chunks[i]["rerank_score"] = float(score)

    chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
    return chunks[:TOP_RERANK]


# ======================
# MAIN
# ======================

def main():
    class_name = get_latest_schema_class()
    if not class_name:
        print("❌ No Blop_zupan schema found.")
        return

    print(f"📦 Using class: {class_name}")
    print(f"🔍 TOP_K = {TOP_K} → rerank → TOP_{TOP_RERANK}\n")

    results = []

    for i, query in enumerate(QUERIES):
        print(f"[{i+1}/{len(QUERIES)}] {query}")

        vector = embed_content(query)
        if not vector:
            continue

        hits = search_weaviate(class_name, vector)

        chunks = []
        for h in hits:
            chunks.append({
                "chunk_id": h.get("chunk_id"),
                "text": h.get("text"),
                "distance": h.get("_additional", {}).get("distance")
            })

        # ✅ RERANK HERE
        top_chunks = rerank_chunks(query, chunks)

        results.append({
            "query": query,
            "top_5_chunks": top_chunks
        })

        print("   → reranked top 5 selected\n")

    output_path = "/home/shtlp_0107/Desktop/ComplexTable_Comparison_GFS_VS_Parsing/res_reranked.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"✅ Saved reranked results to:\n{output_path}")


if __name__ == "__main__":
    main()
