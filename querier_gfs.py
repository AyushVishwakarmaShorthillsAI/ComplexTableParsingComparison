import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# =========================
# 🔧 CONFIG
# =========================
STORE_INFO_FILE = "store_info.json"
RESULTS_FOLDER = "queries_results"

# Hardcoded Queries (Can be modified by user)
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

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("⚠️ API Key missing in .env (expected GOOGLE_API_KEY or GEMINI_API_KEY)")

client = genai.Client(api_key=api_key)

def load_store_id():
    """Loads the store ID from the persisted file."""
    if not os.path.exists(STORE_INFO_FILE):
        raise RuntimeError(f"❌ {STORE_INFO_FILE} not found. Run embedder.py first.")
    
    with open(STORE_INFO_FILE, 'r') as f:
        data = json.load(f)
        return data.get("store_id")

def query_store(store_id, query_text):
    """Runs query against store and returns (answer, chunks_list)"""
    print(f"\n🔎 Running query: {query_text}")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash", 
                contents=query_text,
                config=types.GenerateContentConfig(
                    tools=[
                        types.Tool(
                            file_search=types.FileSearch(
                                file_search_store_names=[store_id]
                            )
                        )
                    ]
                )
            )
            
            answer = response.text
            chunks = []
            
            # Extract ALL chunks
            if response.candidates and response.candidates[0].grounding_metadata:
                grounding = response.candidates[0].grounding_metadata
                if grounding.grounding_chunks:
                    for chunk in grounding.grounding_chunks:
                        ctx = chunk.retrieved_context
                        if ctx:
                            chunks.append({
                                "content": ctx.text,
                                "title": ctx.title,
                                "score": 0.0 # GFS doesn't expose score directly
                            })
                        
            return answer, chunks

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print(f"⚠️ Quota exhausted (429). Attempt {attempt + 1}/{max_retries}. Waiting 45s...")
                import time
                time.sleep(45)
            else:
                raise e
    
    raise RuntimeError("❌ Max retries exceeded due to quota limits.")

def save_all_results(data, model_name):
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    filename = "gfs_multi_query_results.json"
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
    results_path = os.path.join(RESULTS_FOLDER, "gfs_multi_query_results.json")
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    try:
        store_id = load_store_id()
        print(f"📂 Using Store ID: {store_id}")
        
        # Load existing results if any (to resume or append)
        all_query_results = []
        if os.path.exists(results_path):
            try:
                with open(results_path, "r", encoding="utf-8") as f:
                    old_data = json.load(f)
                    all_query_results = old_data.get("queries", [])
            except:
                pass

        for query in QUERIES:
            # Check if already answered
            if any(res["query"] == query for res in all_query_results):
                print(f"⏩ Query already completed: {query[:50]}...")
                continue

            ans, chunks = query_store(store_id, query)
            
            all_query_results.append({
                "query": query,
                "answer": ans,
                "results": chunks
            })
            
            # Incremental Save
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump({
                    "model": "Gemini-File-Search",
                    "total_queries": len(all_query_results),
                    "queries": all_query_results
                }, f, indent=4)
            print(f"✅ Progress saved to {results_path}")
             
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
