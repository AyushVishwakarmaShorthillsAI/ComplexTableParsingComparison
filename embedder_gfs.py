import os
import time
import json
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv()

# =========================
# 🔧 CONFIG
# =========================
INPUT_PDF_DIR = "input_pdfs"
STORE_INFO_FILE = "store_info.json"
# We can keep a default display name, but we don't need syndrome specific variables here anymore
FILE_SEARCH_STORE_DISPLAY_NAME = "Genomic_File_Search_Store"

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("⚠️ API Key missing in .env (expected GOOGLE_API_KEY or GEMINI_API_KEY)")

client = genai.Client(api_key=api_key)

def setup_store():
    """Creates store and ingests PDFs, then saves store ID to a file."""
    print("📦 Creating File Search Store...")

    file_search_store = client.file_search_stores.create(
        config={
            "display_name": FILE_SEARCH_STORE_DISPLAY_NAME
        }
    )

    store_id = file_search_store.name
    print(f"✅ Store created: {store_id}")

    # Ingest PDFs
    pdf_paths = list(Path(INPUT_PDF_DIR).glob("*.pdf"))
    if not pdf_paths:
        raise RuntimeError(f"❌ No PDFs found in {INPUT_PDF_DIR}/")

    print(f"📄 Found {len(pdf_paths)} PDF(s)")

    operations = []

    for pdf in pdf_paths:
        print(f"⬆️ Ingesting: {pdf.name}")

        op = client.file_search_stores.upload_to_file_search_store(
            file=str(pdf),
            file_search_store_name=store_id,
            config={
                "display_name": pdf.name,
                'chunking_config': {
                    'white_space_config': {
                        'max_tokens_per_chunk': 400,
                        'max_overlap_tokens': 50
                    }
                }
            }
        )
        operations.append(op)

    print("⏳ Waiting for ingestion & indexing to complete...")
    for op in operations:
        retries = 10
        while not op.done and retries > 0:
            try:
                time.sleep(5) 
                op = client.operations.get(op)
            except Exception as e:
                print(f"⚠️ Polling error: {e}. Retrying... ({retries} left)")
                retries -= 1
                time.sleep(5)

    print("✅ All PDFs ingested and indexed")
    
    # Save store info
    with open(STORE_INFO_FILE, 'w') as f:
        json.dump({"store_id": store_id}, f)
    print(f"💾 Store ID saved to {STORE_INFO_FILE}")

    return store_id

if __name__ == "__main__":
    try:
        setup_store()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
