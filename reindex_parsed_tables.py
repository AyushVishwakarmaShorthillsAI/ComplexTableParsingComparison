import os
import io
import json
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import weaviate

load_dotenv()

# =========================
# 🔧 CONFIG
# =========================
PARSED_TABLE_DIR = "parsed_table"
EMBEDDING_MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

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
        print("🗑️  Deleting existing 'TableChunk' collection for re-indexing...")
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

def reindex_everything():
    """Read all text files in parsed_table and index them."""
    if not os.path.exists(PARSED_TABLE_DIR):
        print(f"❌ {PARSED_TABLE_DIR} directory not found.")
        return

    setup_weaviate_schema()
    collection = weaviate_client.collections.get("TableChunk")

    txt_files = list(Path(PARSED_TABLE_DIR).glob("*.txt"))
    if not txt_files:
        print(f"⚠️  No text files found in {PARSED_TABLE_DIR}")
        return

    print(f"📄 Found {len(txt_files)} parsed table(s). Starting indexing...")

    for txt_file in txt_files:
        file_basename = txt_file.name # e.g., Calmels_2018.pdf_6.txt
        
        # Parse source and page
        try:
            # Expected format: {pdf_name}_{page}.txt
            # Using rsplit to handle cases where pdf_name might contain underscores
            name_part, page_part = txt_file.stem.rsplit('_', 1)
            pdf_name = name_part
            page_num = int(page_part)
        except ValueError:
            print(f"⚠️  Could not parse page number from filename: {file_basename}. Using default.")
            pdf_name = file_basename
            page_num = 0

        print(f"  📥 Processing {pdf_name} (Page {page_num})...")
        
        with open(txt_file, "r", encoding="utf-8") as f:
            description = f.read()
        
        if not description.strip():
            print(f"  ⏭️  Skipping empty file: {file_basename}")
            continue

        chunks = chunk_text(description, CHUNK_SIZE, CHUNK_OVERLAP)
        embeddings = embedding_model.encode(chunks)

        print(f"    📦 Storing {len(chunks)} chunks in Weaviate...")
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
    
    print("\n🎉 Re-indexing complete!")

if __name__ == "__main__":
    try:
        reindex_everything()
    except Exception as e:
        print(f"❌ Error during re-indexing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        weaviate_client.close()
