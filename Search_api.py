from flask import Flask, request, jsonify
import sqlite3
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re

class SearchEngine:
    def __init__(self, db_path, index_path, id_map_path, sources_path, model_name='all-MiniLM-L6-v2'):
        print("Loading search engine components...")
        self.model = SentenceTransformer(model_name, device='cpu')
        self.index = faiss.read_index(index_path)
        with open(id_map_path, 'r') as f:
            self.faiss_to_db_id = json.load(f)
        with open(sources_path, 'r') as f:
            self.sources = {item['filename']: item for item in json.load(f)}
        self.db_conn = sqlite3.connect(db_path, check_same_thread=False)
        self.db_conn.row_factory = sqlite3.Row
        print("Search engine loaded successfully.")

    def _normalize_scores(self, scores: list) -> list:
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [1.0] * len(scores)
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
                    'db_id': db_id, 'vector_score': float(score), 'text': row['chunk_text'],
                    'source': row['source_file'], 'page': row['page_number']
                })
        return results

    def search_hybrid(self, query: str, k: int = 5, alpha: float = 0.6) -> list:
        candidate_k = k * 6
        initial_results = self.search_baseline(query, k=candidate_k)
        if not initial_results: return []

        cursor = self.db_conn.cursor()
        sanitized_query = re.sub(r'[^\w\s]', '', query)
        sql = f"SELECT rowid, rank FROM chunks WHERE chunks MATCH '{sanitized_query}' ORDER BY rank"
        cursor.execute(sql)
        fts_scores = {row['rowid']: row['rank'] for row in cursor.fetchall()}
        
        vector_scores = [res['vector_score'] for res in initial_results]
        min_fts_score = min(fts_scores.values()) - 1 if fts_scores else -100
        keyword_scores = [fts_scores.get(res['db_id'], min_fts_score) for res in initial_results]

        norm_vector_scores = self._normalize_scores(vector_scores)
        norm_keyword_scores = self._normalize_scores(keyword_scores)

        for i, res in enumerate(initial_results):
            res['final_score'] = alpha * norm_vector_scores[i] + (1 - alpha) * norm_keyword_scores[i]

        reranked_results = sorted(initial_results, key=lambda x: x['final_score'], reverse=True)
        return reranked_results[:k]

app = Flask(__name__)


engine = SearchEngine(
    db_path="knowledge_base.db",
    index_path="faiss_index.idx",
    id_map_path="faiss_ids.json",
    sources_path="source_updated.json"
)

def format_contexts(results: list) -> list:
    #Formats search results into the desired API response structure.
    contexts = []
    for res in results:
        source_info = engine.sources.get(res['source'], {})
        score_key = 'final_score' if 'final_score' in res else 'vector_score'
        contexts.append({
            'text': res['text'],
            'source_title': source_info.get('title'),
            'source_link': source_info.get('url'),
            'score': float(res[score_key])
        })
    return contexts

def get_extractive_answer(results: list, threshold: float = 0.5) -> (str, list):
    #Generates a simple extractive answer or abstains if confidence is low.
    if not results:
        return None

    top_result = results[0]
    score_key = 'final_score' if 'final_score' in top_result else 'vector_score'
    
    if top_result[score_key] < threshold:
        return None # Abstain if the top score is below the threshold
    
    return top_result['text']

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    if not data or 'q' not in data:
        return jsonify({"error": "Missing 'q' (query) in request body"}), 400

    query = data['q']
    k = data.get('k', 3)
    mode = data.get('mode', 'hybrid') 

    if mode == 'baseline':
        results = engine.search_baseline(query, k=k)
        reranker_used = False
    elif mode == 'hybrid':
        results = engine.search_hybrid(query, k=k)
        reranker_used = True
    else:
        return jsonify({"error": "Invalid mode. Choose 'baseline' or 'hybrid'."}), 400
    
    answer = get_extractive_answer(results)
    contexts = format_contexts(results)

    response = {
        'answer': answer,
        'contexts': contexts,
        'reranker_used': reranker_used
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)