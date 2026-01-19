import nltk
import os
import json

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

def chunk_text(text, chunk_size=500, overlap_size=50):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        # Calculate length added by this sentence (plus a space if not first)
        sent_len = len(sentence)
        added_len = sent_len + (1 if current_chunk else 0)
        
        if current_length + added_len <= chunk_size:
            current_chunk.append(sentence)
            current_length += added_len
        else:
            # If the current chunk is not empty, finalize it
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            
            # Create the overlap for the new chunk from the end of the previous chunk
            # We want at least `overlap_size` characters from the end
            overlap_buffer = []
            overlap_len = 0
            
            # Iterate backwards through the current chunk to find sufficient overlap
            for s in reversed(current_chunk):
                overlap_buffer.insert(0, s)
                overlap_len += len(s) + (1 if len(overlap_buffer) > 1 else 0)
                if overlap_len >= overlap_size:
                    break
            
            # Start new chunk with the overlap sentences + the current sentence
            # Note: We must handle the case where the overlap IS the entire previous chunk
            # to avoid infinite loops if a single sentence is huge, but here we added 'sentence' 
            # which guarantees progress.
            
            current_chunk = list(overlap_buffer)
            current_length = overlap_len
            
            # Add the current sentence
            current_chunk.append(sentence)
            current_length += len(sentence) + 1
            
    # Add any remaining text
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def process_folder(folder_path, output_file):
    all_chunks = []
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return

    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    print(f"Found {len(files)} text files in {folder_path}.")
    
    for filename in sorted(files):
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            file_chunks = chunk_text(text)
            
            for i, chunk_content in enumerate(file_chunks):
                all_chunks.append({
                    "source": filename,
                    "chunk_index": i,
                    "content": chunk_content
                })
                
            print(f"Processed {filename}: {len(file_chunks)} chunks.")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=4)
    
    print(f"Successfully saved {len(all_chunks)} chunks to {output_file}")

if __name__ == "__main__":
    folder_name = "BlopZupan"
    # Assuming the script is run from the parent directory of BlopZupan
    # or provided with the absolute path.
    # Given the user's setup, I will use the absolute path to be safe.
    base_dir = "/home/shtlp_0107/Desktop/ComplexTable_Comparison_GFS_VS_Parsing"
    target_folder = os.path.join(base_dir, folder_name)
    output_json = os.path.join(base_dir, "chunks.json")
    
    process_folder(target_folder, output_json)
