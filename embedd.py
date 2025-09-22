import sqlite3
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

def generate_and_store_embeddings_batched(db_path: str, index_path: str, id_map_path: str, batch_size: int = 128):
    if not os.path.exists(db_path):
        print(f"Error: Database not found at '{db_path}'")
        return
    print(f"Connecting to the database at '{db_path}'...")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT rowid, chunk_text FROM chunks ORDER BY rowid")
    results = cursor.fetchall()
    conn.close()
    
    if not results:
        print("No data found in 'chunks' table. Aborting.")
        return
        
    db_row_ids = [row['rowid'] for row in results]
    texts = [row['chunk_text'] for row in results]
    print(f"Fetched {len(texts)} chunks from the database.")

    print("Loading the sentence-transformer model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

    dimension = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dimension)
    
    print(f"Starting embedding generation in batches of {batch_size}...")
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        print(f"  -> Processing batch {i//batch_size + 1}/{-(-len(texts)//batch_size)}...")
        batch_embeddings = model.encode(
            batch_texts, 
            show_progress_bar=False, 
            normalize_embeddings=True
        )
        
        index.add(batch_embeddings.astype('float32'))

    print("\nAll batches processed successfully.")
    print(f"FAISS index now contains {index.ntotal} vectors.")

    print(f"Saving FAISS index to '{index_path}'...")
    faiss.write_index(index, index_path)
    
    print(f"Saving database row ID mapping to '{id_map_path}'...")
    with open(id_map_path, 'w') as f:
        json.dump(db_row_ids, f)
        
    print("\n Embedding generation and storage complete!")

if __name__ == '__main__':
    db_file = "knowledge_base.db"
    faiss_index_file = "faiss_index.idx"
    id_map_file = "faiss_ids.json"
    generate_and_store_embeddings_batched(db_file, faiss_index_file, id_map_file)