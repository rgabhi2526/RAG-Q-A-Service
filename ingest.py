import pymupdf 
import re
import json
import os
from typing import List, Dict, Any
from pathlib import Path

def ingest_and_chunk_pdf(file_path: str) -> List[Dict[str, Any]]:
    doc = pymupdf.open(file_path)
    file_name = os.path.basename(file_path)
    
    all_chunks = []
    
    for page_num, page in enumerate(doc):
        blocks = sorted(page.get_text("blocks"), key=lambda b: b[1])
        
        current_paragraph = ""
        last_y1 = 0
        
        for b in blocks:
            text = b[4]
            cleaned_text = text.replace('\n', ' ').strip()
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
            
            if not cleaned_text:
                continue

            is_list_item = re.match(r'^\s*(?:•|-|\*|–|[a-zA-Z0-9]+\.)\s+', cleaned_text) is not None
            is_new_paragraph = not last_y1 or (b[1] > (last_y1 + 10) and not is_list_item)

            if is_new_paragraph:
                if current_paragraph:
                    current_paragraph = re.sub(r'-\s+', '', current_paragraph)
                    all_chunks.append({
                        'text': current_paragraph,
                        'page_number': page_num + 1,
                        'source_file': file_name
                    })
                current_paragraph = cleaned_text
            else:
                separator = "\n" if is_list_item else " "
                current_paragraph += separator + cleaned_text
            
            last_y1 = b[3]
        
        if current_paragraph:
            current_paragraph = re.sub(r'-\s+', '', current_paragraph)
            all_chunks.append({
                'text': current_paragraph,
                'page_number': page_num + 1,
                'source_file': file_name
            })

    final_chunks = [
        chunk for chunk in all_chunks if len(chunk['text'].split()) > 10
    ]

    return final_chunks


def process_pdf_folder(folder_path: str) -> List[Dict[str, Any]]:
    if not os.path.isdir(folder_path):
        print(f"Error: The path '{folder_path}' is not a valid directory.")
        return []

    all_docs_chunks = []
    pdf_files = list(Path(folder_path).glob('*.pdf'))
    
    print(f"Found {len(pdf_files)} PDF(s) to process...")

    for pdf_path in pdf_files:
        print(f"  -> Processing '{pdf_path.name}'...")
        try:
            chunks_from_one_pdf = ingest_and_chunk_pdf(str(pdf_path))
            all_docs_chunks.extend(chunks_from_one_pdf)
        except Exception as e:
            print(f"    [!] Failed to process {pdf_path.name}: {e}")
            
    return all_docs_chunks


def restructure_chunks(flat_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Restructures a flat list of chunk dictionaries into a nested dictionary.
    
    Args:
        flat_chunks: A list of dicts, e.g., [{'text': '...', 'page_number': 1, 'source_file': 'a.pdf'}]

    Returns:
        A nested dict, e.g., {'a.pdf': {'1': ['chunk text', ...]}}
    """
    nested_structure = {}
    for chunk in flat_chunks:
        source = chunk['source_file']
        page = str(chunk['page_number']) 
        text = chunk['text']

        if source not in nested_structure:
            nested_structure[source] = {}
 
        if page not in nested_structure[source]:
            nested_structure[source][page] = []
            
        nested_structure[source][page].append(text)
        
    return nested_structure

if __name__ == '__main__':
    folder_to_process = "./industrial-safety-pdfs"
    output_filename = "extracted_chunks_nested.json"

    total_chunks_flat = process_pdf_folder(folder_to_process)
    
    if total_chunks_flat:
        print("\nRestructuring data for nested JSON output...")
        nested_data = restructure_chunks(total_chunks_flat)

        print(f"Writing {len(total_chunks_flat)} chunks to '{output_filename}'...")
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(nested_data, f, indent=4, ensure_ascii=False)
            
        print(f"Successfully saved nested data to '{output_filename}'.")
    else:
        print("No chunks were extracted or the folder was empty.")