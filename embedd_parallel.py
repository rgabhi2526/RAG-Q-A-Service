import sqlite3
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import multiprocessing
import math

def encode_batch(batch_texts):
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    print(f"  -> Process {os.getpid()} encoding a batch of {len(batch_texts)} chunks...")
    return model.encode(batch_texts, show_progress_bar=False, normalize_embeddings=True)

def generate_and_store_embeddings_multiprocess(db_path: str, index_path: str, id_map_path: str):
    if not os.path.exists(db_path):
        print(f"Error: Database not found at '{db_path}'")
        return
    print(f"Connecting to the database at '{db_path}'...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT rowid, chunk_text FROM chunks ORDER BY rowid")
    results = cursor.fetchall()
    conn.close()
    
    if not results:
        print("No data found in 'chunks' table. Aborting.")
        return
        
    db_row_ids = [row[0] for row in results]
    texts = [row[1] for row in results]
    print(f"Fetched {len(texts)} chunks from the database.")

    num_processes = multiprocessing.cpu_count()
    print(f"Starting a multiprocessing pool with {num_processes} workers.")

    chunk_size = math.ceil(len(texts) / num_processes)
    text_batches = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]

    with multiprocessing.Pool(num_processes) as pool:
        batch_embeddings_list = pool.map(encode_batch, text_batches)
    print("\nAll batches processed. Consolidating results...")
    embeddings = np.concatenate(batch_embeddings_list, axis=0)
    print(f"Embeddings generated successfully. Shape: {embeddings.shape}")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype('float32'))
    
    print(f"Saving FAISS index to '{index_path}'...")
    faiss.write_index(index, index_path)
    
    print(f"Saving database row ID mapping to '{id_map_path}'...")
    with open(id_map_path, 'w') as f:
        json.dump(db_row_ids, f)
        
    print("\n✅ Multiprocess embedding generation complete!")

if __name__ == '__main__':
    multiprocessing.set_start_method('fork', force=True)

    db_file = "knowledge_base.db"
    faiss_index_file = "faiss_index.idx"
    id_map_file = "faiss_ids.json"
    
    generate_and_store_embeddings_multiprocess(db_file, faiss_index_file, id_map_file)