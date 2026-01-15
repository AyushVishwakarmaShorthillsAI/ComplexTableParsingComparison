import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import weaviate

load_dotenv()

# =========================
# 🔧 CONFIG
# =========================
EMBEDDING_MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"
RESULTS_FOLDER = "queries_results"
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
LITELLM_PROXY_API_BASE = os.getenv("LITELLM_PROXY_API_BASE")
MODEL_NAME = "gemini-2.5-flash-unique"

if not LITELLM_API_KEY or not LITELLM_PROXY_API_BASE:
    raise RuntimeError("⚠️ LITELLM_API_KEY or LITELLM_PROXY_API_BASE missing in .env")

# Initialize LiteLLM (OpenAI) Client
client = OpenAI(api_key=LITELLM_API_KEY, base_url=LITELLM_PROXY_API_BASE)

# Initialize Embedding Model
print(f"🚀 Loading embedding model: {EMBEDDING_MODEL_NAME}...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu", trust_remote_code=True)

# Initialize Weaviate (Embedded)
print("🌐 Connecting to Weaviate...")
WEAVIATE_PERSIST_DIR = os.path.join(os.getcwd(), "weaviate_data")
os.makedirs(WEAVIATE_PERSIST_DIR, exist_ok=True)

weaviate_client = weaviate.connect_to_embedded(
    persistence_data_path=WEAVIATE_PERSIST_DIR
)

def query_system(query_text):
    """Retrieve relevant chunks and generate an answer using LiteLLM."""
    print(f"\n🔎 Querying: {query_text}")
    
    # 1. Embed Query
    query_vector = embedding_model.encode(query_text).tolist()

    # 2. Retrieve from Weaviate
    collection = weaviate_client.collections.get("TableChunk")
    response = collection.query.near_vector(
        near_vector=query_vector,
        limit=5,
        return_properties=["content", "source", "page"]
    )

    if not response.objects:
        print("❌ No relevant information found in vector DB.")
        return None, []

    context = ""
    chunks_for_json = []
    for idx, obj in enumerate(response.objects):
        props = obj.properties
        context += f"--- Source: {props['source']}, Page: {props['page']} ---\n"
        context += f"{props['content']}\n\n"
        chunks_for_json.append({
            "content": props['content'],
            "source": props['source'],
            "page": props['page']
        })

    # 3. Generate Answer with LiteLLM
    prompt = f"""You are a helpful research assistant. Use the following retrieved chunks from table analysis to answer the query accurately. 
If the information is not in the context, say you don't know.

Context:
{context}

Query: {query_text}

Answer:"""

    print("🤖 Generating summary with LiteLLM...")
    answer_response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    
    answer = answer_response.choices[0].message.content
    print(f"\n💡 Answer:\n{answer}")
    print("\n📚 Sources:")
    for obj in response.objects:
        print(f"- {obj.properties['source']} (Page {obj.properties['page']})")
        
    return answer, chunks_for_json

def save_all_results(data, model_name):
    """Save query results to a JSON file."""
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    filename = "table_rag_results.json"
    filepath = os.path.join(RESULTS_FOLDER, filename)
    
    output_data = {
        "model": model_name,
        "total_queries": len(data),
        "queries": data
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"\n💾 All results saved to {filepath}")
    return filepath

if __name__ == "__main__":
    QUERIES = [
        # --- Zhang.pdf Queries ---
        "What is the most frequent genetic variant listed for the patients in Table 1?",
        "Which patient in Table 1 carries the p.Glu211Lys variant instead of the more common p.Glu209Lys?",
        "Which patient listed in Table 1 had the latest onset of seizures (at 17 years old)?",
        "How many patients in the table are reported to have microcephaly?",
        "What specific structural abnormality is listed in the 'Others' column for Case 2?",

        # --- Calmels.pdf Queries ---
        "Which patient from Morocco described in Table 1 carries the p.(Glu13*) protein alteration?",
        "According to Table 2, which protein alteration is associated with the mutation c.2008C>T in patient CS18LO?",
        "Based on Table 4, what is the percentage of dental anomalies found specifically in CS-B patients?",
        "Which homozygous patient in Table 1 has a clinical classification of 'I' and is originally from India?",
        "In Table 2, which patient from Lebanon is listed with the mutation c.640G>T?",

        # --- VANSICKLE.pdf Queries ---
        "What is the specific ODC1 gene variant listed for Patient 8 in Table 1?",
        "According to Table 1, at what age did Patient 6 start walking?",
        "Which patient in Table 1 is reported to have epilepsy with an onset at 14 years?",
        "What specific hair phenotype description is listed for Patient 7 in Table 1?",
        "Does Patient 9 in Table 1 exhibit hypotonia according to the clinical data?",

        # --- Bertoli-Avella.pdf Queries ---
        "In Table 1, which gene is associated with the phenotype including 'Alagille-like syndrome' for Patient 14?",
        "What is the cDNA variant listed for Patient 17 in the PLK1 gene section of Table 1?",
        "According to Table 1, is consanguinity present in the family history of Patient 26 with the ZNF699 variant?",
        "What specific skeletal abnormality is listed for Patient 10 in the 'Phenotype' column of Table 1?",
        "Which patient in Table 1 presents with 'congenital heart defects' associated with the RAP1GDS1 gene?",

        # --- Bloch-Zupan.pdf Queries ---
        "According to Table 1, what is the DMFT score (Decayed, Missing, Filled Teeth) for patient 5?",
        "Which patient in Table 1 has the mutation c.1954C>T?",
        "What type of 'Structure' anomaly regarding enamel is listed for Patient 16 in Table 1?",
        "In Table 2, what is the specific 'Facial axis' measurement recorded for Patient 16?",
        "According to Table 1, does Patient 1 exhibit dental crowding?"
    ]
    
    results_path = os.path.join(RESULTS_FOLDER, "table_rag_results.json")
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    try:
        # Load existing results if any (to resume or append)
        all_query_results = []
        if os.path.exists(results_path):
            try:
                with open(results_path, "r", encoding="utf-8") as f:
                    old_data = json.load(f)
                    all_query_results = old_data.get("queries", [])
            except:
                pass

        for q in QUERIES:
            # Check if already answered (optional, but good for restartability)
            if any(res["query"] == q for res in all_query_results):
                print(f"⏩ Query already completed: {q[:50]}...")
                continue

            ans, chunks = query_system(q)
            if ans:
                all_query_results.append({
                    "query": q,
                    "answer": ans,
                    "results": chunks
                })
                
                # Incremental Save
                with open(results_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "model": MODEL_NAME,
                        "total_queries": len(all_query_results),
                        "queries": all_query_results
                    }, f, indent=4)
                print(f"✅ Progress saved to {results_path}")
        
    except Exception as e:
        print(f"❌ Error during retrieval: {e}")
        import traceback
        traceback.print_exc()
    finally:
        weaviate_client.close()
