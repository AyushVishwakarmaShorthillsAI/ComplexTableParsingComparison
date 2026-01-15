import os
import json
import time
import io
import base64
from pathlib import Path
from typing import List, Dict

import fitz
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import weaviate

load_dotenv()

# =========================
# 🔧 CONFIG
# =========================
INPUT_PDF_DIR = "input_pdfs"
TABLE_PAGES_FILE = "table_pages.json"
TABLE_IMAGES_DIR = "table_images"
PARSED_TABLE_DIR = "parsed_table"
EMBEDDING_MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
LITELLM_PROXY_API_BASE = os.getenv("LITELLM_PROXY_API_BASE")
MODEL_NAME = "gemini-2.5-pro-unique" # Use target model here

if not LITELLM_API_KEY or not LITELLM_PROXY_API_BASE:
    raise RuntimeError("⚠️ LITELLM_API_KEY or LITELLM_PROXY_API_BASE missing in .env")

# Initialize LiteLLM (OpenAI) Client with a default timeout
client = OpenAI(
    api_key=LITELLM_API_KEY, 
    base_url=LITELLM_PROXY_API_BASE,
    timeout=300.0  # 5 minute timeout for all requests
)

# Initialize Embedding Model
print(f"🚀 Loading embedding model: {EMBEDDING_MODEL_NAME}...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu", trust_remote_code=True)

# Initialize Weaviate (Embedded)
print("🌐 Connecting to Weaviate (local embedded)...")
WEAVIATE_PERSIST_DIR = os.path.join(os.getcwd(), "weaviate_data")
os.makedirs(WEAVIATE_PERSIST_DIR, exist_ok=True)

weaviate_client = weaviate.connect_to_embedded(
    persistence_data_path=WEAVIATE_PERSIST_DIR
)

def setup_weaviate_schema():
    """Create the TableChunk collection in Weaviate."""
    print("🏗️  Setting up Weaviate schema...")
    
    if weaviate_client.collections.exists("TableChunk"):
        # For this task, we will overwrite/recreate if it exists to ensure freshness
        weaviate_client.collections.delete("TableChunk")
        
    weaviate_client.collections.create(
        name="TableChunk",
        description="Chunks of semantically described tables from PDFs",
        properties=[
            weaviate.classes.config.Property(name="content", data_type=weaviate.classes.config.DataType.TEXT),
            weaviate.classes.config.Property(name="source", data_type=weaviate.classes.config.DataType.TEXT),
            weaviate.classes.config.Property(name="page", data_type=weaviate.classes.config.DataType.INT),
        ],
        vectorizer_config=None # We will provide our own vectors (GTE)
    )
    print("✅ Weaviate schema ready.")

def render_page(pdf_path, page_num, dpi=300):
    """Render a PDF page to a Pillow Image."""
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num - 1)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img_data = pix.tobytes("png")
    doc.close()
    return Image.open(io.BytesIO(img_data))

def encode_image(image):
    """Encode Pillow image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def extract_table_description(image):
    """Send image to LiteLLM for semantic description with retries."""
    prompt = """You are a research paper assistant. This page contains a table. Extract and describe the table content in a format optimized for semantic search and RAG. 
1. Table Title/Caption. 
2. Table Purpose. 
3. Table Structure (columns/rows). 
4. Table Data: Present the data in natural language paragraphs (NOT as a raw table). Group related data, use complete sentences, include all values. 
5. Key Findings. 
6. Context (footnotes/notes)."""

    base64_image = encode_image(image)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"    📡 Sending request to LiteLLM ({MODEL_NAME})...")
            start_time = time.time()
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0
            )
            duration = time.time() - start_time
            print(f"    ✨ Received response in {duration:.1f}s")
            return response.choices[0].message.content
        except Exception as e:
            err_msg = str(e).lower()
            if "429" in err_msg or "resource_exhausted" in err_msg or "quota" in err_msg:
                print(f"⚠️ Quota exhausted (429). Attempt {attempt + 1}/{max_retries}. Waiting 45s...")
                time.sleep(45)
            else:
                raise e
    
    raise RuntimeError("❌ Max retries exceeded due to quota limits.")

def chunk_text(text, size, overlap):
    """Chunk text into fixed size strings with overlap."""
    chunks = []
    if not text:
        return chunks
    
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
        if start >= len(text):
            break
    return chunks

def ingest_tables():
    """Main ingestion pipeline."""
    if not os.path.exists(TABLE_PAGES_FILE):
        print(f"❌ {TABLE_PAGES_FILE} not found.")
        return

    with open(TABLE_PAGES_FILE, "r") as f:
        table_config = json.load(f)

    setup_weaviate_schema()
    collection = weaviate_client.collections.get("TableChunk")

    for pdf_name, pages in table_config.items():
        pdf_path = os.path.join(INPUT_PDF_DIR, pdf_name)
        if not os.path.exists(pdf_path):
            print(f"⚠️ PDF not found: {pdf_path}")
            continue

        print(f"\n📄 Processing {pdf_name}...")
        for page_num in pages:
            print(f"  📸 Rendering page {page_num}...")
            img = render_page(pdf_path, page_num)
            
            # Save image to folder for inspection
            img_filename = f"{pdf_name}_page_{page_num}.png"
            img_save_path = os.path.join(TABLE_IMAGES_DIR, img_filename)
            img.save(img_save_path)
            print(f"  💾 Image saved to {img_save_path}")
            
            # Check if parsed description already exists on disk
            txt_filename = f"{pdf_name}_{page_num}.txt"
            txt_save_path = os.path.join(PARSED_TABLE_DIR, txt_filename)
            
            if os.path.exists(txt_save_path):
                print(f"  ⏭️  Skipping LiteLLM: Found existing description at {txt_save_path}")
                with open(txt_save_path, "r", encoding="utf-8") as f_txt:
                    description = f_txt.read()
            else:
                print(f"  🧠 Extracting semantic description with LiteLLM...")
                description = extract_table_description(img)
                
                # Save parsed description to disk
                with open(txt_save_path, "w", encoding="utf-8") as f_txt:
                    f_txt.write(description)
                print(f"  📝 Saved parsed description to {txt_save_path}")
            
            print(f"  ✂️  Chunking and Embedding...")
            chunks = chunk_text(description, CHUNK_SIZE, CHUNK_OVERLAP)
            embeddings = embedding_model.encode(chunks)

            print(f"  📦 Storing {len(chunks)} chunks in Weaviate...")
            with collection.batch.dynamic() as batch:
                for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
                    batch.add_object(
                        properties={
                            "content": chunk,
                            "source": pdf_name,
                            "page": page_num
                        },
                        vector=vector.tolist()
                    )
            print(f"  ✅ Page {page_num} finished.")

if __name__ == "__main__":
    try:
        os.makedirs(TABLE_IMAGES_DIR, exist_ok=True)
        os.makedirs(PARSED_TABLE_DIR, exist_ok=True)
        ingest_tables()
        print("\n🎉 All tables ingested successfully into Weaviate!")
    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        import traceback
        traceback.print_exc()
    finally:
        weaviate_client.close()
