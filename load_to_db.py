import sqlite3
import json
import os
from typing import Dict, Any

def create_database_and_load_data(json_path: str, db_path: str):
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at '{json_path}'")
        return
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    print(f"Successfully connected to SQLite database at '{db_path}'")
    cursor.execute("DROP TABLE IF EXISTS chunks")
    # Create the 'chunks' table using FTS5
    # FTS5 allows for fast full-text searching
    # We store source_file and page_number as unindexed columns to save space
    # but still have them available for retrieval.
    create_table_sql = """
    CREATE VIRTUAL TABLE chunks USING fts5(
        chunk_text,
        source_file UNINDEXED,
        page_number UNINDEXED
    );
    """
    cursor.execute(create_table_sql)
    print("'chunks' FTS5 table created successfully.")
    with open(json_path, 'r', encoding='utf-8') as f:
        data: Dict[str, Dict[str, Any]] = json.load(f)
    chunks_to_insert = []
    for source_file, pages in data.items():
        for page_number, chunk_list in pages.items():
            for chunk_text in chunk_list:
                chunks_to_insert.append(
                    (chunk_text, source_file, int(page_number))
                )
    print(f"Inserting {len(chunks_to_insert)} chunks into the database...")
    cursor.executemany(
        "INSERT INTO chunks (chunk_text, source_file, page_number) VALUES (?, ?, ?)",
        chunks_to_insert
    )
    conn.commit()
    print("Data committed to the database.")
    count_query = "SELECT count(*) FROM chunks"
    total_rows = cursor.execute(count_query).fetchone()[0]
    print(f"✅ Verification successful: Found {total_rows} rows in the 'chunks' table.")
    
    conn.close()

if __name__ == '__main__':
    input_json_file = "extracted_chunks_nested.json" 
    output_db_file = "knowledge_base.db"
    create_database_and_load_data(input_json_file, output_db_file)