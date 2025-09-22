import sqlite3
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re

class SearchEngine:
    def __init__(self, db_path, index_path, id_map_path, model_name='all-MiniLM-L6-v2'):
        print("Loading search engine components...")
        self.model = SentenceTransformer(model_name, device='cpu')
        self.index = faiss.read_index(index_path)
        with open(id_map_path, 'r') as f:
            self.faiss_to_db_id = json.load(f)
        self.db_conn = sqlite3.connect(db_path)
        self.db_conn.row_factory = sqlite3.Row
        print("✅ Search engine loaded successfully.")

    def _normalize_scores(self, scores: list) -> list:
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [1.0 for _ in scores]
        return [(score - min_score) / (max_score - min_score) for score in scores]

    def search_baseline(self, query: str, k: int = 5) -> list:
        query_vector = self.model.encode([query], normalize_embeddings=True).astype('float32')
        scores, faiss_indices = self.index.search(query_vector, k)
        
        results = []
        cursor = self.db_conn.cursor()
        for score, faiss_idx in zip(scores[0], faiss_indices[0]):
            if faiss_idx == -1: continue
            
            db_id = self.faiss_to_db_id[faiss_idx]
            cursor.execute("SELECT * FROM chunks WHERE rowid = ?", (db_id,))
            row = cursor.fetchone()
            
            if row:
                results.append({
                    'db_id': db_id,
                    'vector_score': float(score),
                    'text': row['chunk_text'],
                    'source': row['source_file'],
                    'page': row['page_number']
                })
        return results

    def search_hybrid(self, query: str, k: int = 5, alpha: float = 0.6) -> list:
        candidate_k = k * 6 # Retrieve more candidates to give the reranker options
        initial_results = self.search_baseline(query, k=candidate_k)
        if not initial_results:
            return []

        cursor = self.db_conn.cursor()
        sanitized_query = re.sub(r'[^\w\s]', '', query)
        sql = f"SELECT rowid, rank FROM chunks WHERE chunks MATCH '{sanitized_query}' ORDER BY rank"
        cursor.execute(sql)
        fts_scores = {row['rowid']: row['rank'] for row in cursor.fetchall()}

        vector_scores = [res['vector_score'] for res in initial_results]
        # FTS ranks are negative, so a lower number (e.g., -50) is worse than a higher one (e.g., -10)
        min_fts_score = min(fts_scores.values()) - 1 if fts_scores else -100
        keyword_scores = [fts_scores.get(res['db_id'], min_fts_score) for res in initial_results]

        norm_vector_scores = self._normalize_scores(vector_scores)
        norm_keyword_scores = self._normalize_scores(keyword_scores)

        for i, res in enumerate(initial_results):
            vec_score = norm_vector_scores[i]
            key_score = norm_keyword_scores[i]
            res['final_score'] = alpha * vec_score + (1 - alpha) * key_score

        reranked_results = sorted(initial_results, key=lambda x: x['final_score'], reverse=True)

        return reranked_results[:k]

    def __del__(self):
        if self.db_conn:
            self.db_conn.close()

def print_results(results: list, title: str):
    print(f"--- {title} ---")
    if not results:
        print("No results found.")
        return
    for i, res in enumerate(results):
        score_key = 'final_score' if 'final_score' in res else 'vector_score'
        print(f"\n[{i+1}] Score: {res[score_key]:.4f} | Source: {res['source']} (Page {res['page']})")
        print(f"    Text: {res['text'][:250]}...")

if __name__ == '__main__':
    engine = SearchEngine(
        db_path="knowledge_base.db",
        index_path="faiss_index.idx",
        id_map_path="faiss_ids.json"
    )
    
    query = "What is the CE marking of machinery?"
    print(f"\n\nQUERY: '{query}'")

    baseline_results = engine.search_baseline(query, k=3)
    print_results(baseline_results, "BASELINE SEARCH RESULTS (Vector Only)")
    
    hybrid_results = engine.search_hybrid(query, k=3, alpha=0.6)
    print_results(hybrid_results, "HYBRID SEARCH RESULTS (Reranked)")