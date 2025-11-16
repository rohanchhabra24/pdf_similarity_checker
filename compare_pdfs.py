#!/usr/bin/env python3
"""
compare_pdfs_fullmatrix.py

Usage:
    python compare_pdfs_fullmatrix.py path/to/doc_A.pdf path/to/doc_B.pdf

Requirements:
    pip install google-genai PyPDF2 pandas scikit-learn numpy python-dotenv

Set credentials:
    - Create a .env file in the project folder with:
        GOOGLE_API_KEY=your_key_here
      OR set GOOGLE_APPLICATION_CREDENTIALS for a service account JSON.

This script:
 - extracts text from two PDFs,
 - splits into paragraphs and sentence-chunks,
 - attempts to embed chunks with Gemini embeddings (gemini-embedding-001),
 - if embeddings fail, falls back to TF-IDF,
 - aggregates chunk similarities to get paragraph-level N x M matrix,
 - writes CSV reports to comparison_output/.
"""

import sys
import os
import re
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# load env from .env if present
load_dotenv()

# import Gemini SDK
from google import genai

# initialize Gemini client (it will use GOOGLE_API_KEY or GOOGLE_APPLICATION_CREDENTIALS)
genai_client = genai.Client()

# ---------- PDF text extraction ----------
def extract_text_from_pdf_pypdf2(path):
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        raise RuntimeError("Install PyPDF2 (pip install PyPDF2)")
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return "\n".join(pages)

# ---------- paragraph splitting (robust) ----------
def split_paragraphs(text):
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # try blank-lines first
    paras = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    if len(paras) > 1:
        return paras

    # otherwise group lines heuristically
    lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
    if len(lines) > 1:
        grouped = []
        current = []
        for ln in lines:
            current.append(ln)
            if re.search(r'[.!?;:]$', ln):
                if len(current) >= 1:
                    grouped.append(" ".join(current))
                    current = []
        if current:
            grouped.append(" ".join(current))
        if len(grouped) > 1:
            return grouped

    # fallback: group every N sentences
    sents = re.split(r'(?<=[.!?;])\s+', text)
    sents = [s.strip() for s in sents if s.strip()]
    if not sents:
        return [text.strip()]
    n = 3
    paras = []
    for i in range(0, len(sents), n):
        paras.append(" ".join(sents[i:i+n]))
    return paras

# ---------- sentence chunking ----------
def split_sentences(paragraph):
    sents = re.split(r'(?<=[.!?;])\s+', paragraph.strip())
    sents = [s.strip() for s in sents if s.strip()]
    return sents

def chunk_paragraphs(paragraphs):
    chunks = []
    chunk_map = []
    for pi, p in enumerate(paragraphs):
        sents = split_sentences(p)
        if not sents:
            chunks.append(p)
            chunk_map.append({'para_idx': pi, 'chunk_idx': 0, 'text': p})
        else:
            for ci, s in enumerate(sents):
                chunks.append(s)
                chunk_map.append({'para_idx': pi, 'chunk_idx': ci, 'text': s})
    return chunks, chunk_map

# ---------- Gemini embedding helper ----------
def embed_with_gemini(chunks, model="gemini-embedding-001", batch_size=64, sleep_sec=0.08):
    """
    Embed a list of strings using Gemini embeddings (google-genai).
    Returns a NumPy array of normalized embeddings (unit vectors).
    """
    embeddings = []
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        result = genai_client.models.embed_content(model=model, contents=batch)
        # Parse result — handle SDK variants:
        if hasattr(result, "embeddings") and result.embeddings is not None:
            batch_embs = [e.values for e in result.embeddings]
        else:
            batch_embs = []
            try:
                for item in result:
                    if isinstance(item, dict) and "embedding" in item:
                        batch_embs.append(item["embedding"])
                    elif hasattr(item, "values"):
                        batch_embs.append(item.values)
                    else:
                        raise RuntimeError("Unexpected embedding response.")
            except Exception as e:
                raise RuntimeError("Failed to parse embedding response: " + str(e))
        embeddings.extend(batch_embs)
        time.sleep(sleep_sec)

    arr = np.array(embeddings, dtype=np.float32)
    # normalize to unit vectors
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    arr = arr / norms
    return arr

# ---------- TF-IDF fallback ----------
def compute_tfidf_similarity(A_chunks, B_chunks):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vec = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
    corpus = A_chunks + B_chunks
    tfidf = vec.fit_transform(corpus)
    A_vecs = tfidf[:len(A_chunks)]
    B_vecs = tfidf[len(A_chunks):]
    sim = cosine_similarity(A_vecs, B_vecs)
    return sim

# ---------- aggregation ----------
def aggregate_to_paragraph_level(chunk_sim, A_map, B_map, num_A_para, num_B_para):
    para_sim = np.zeros((num_A_para, num_B_para))
    evidence = [[None]*num_B_para for _ in range(num_A_para)]
    all_matches = []

    for a_idx, a_m in enumerate(A_map):
        for b_idx, b_m in enumerate(B_map):
            sim = float(chunk_sim[a_idx, b_idx])
            pa = a_m['para_idx']
            pb = b_m['para_idx']
            all_matches.append({
                'A_para': pa+1, 'B_para': pb+1,
                'A_chunk_idx': a_m['chunk_idx'], 'B_chunk_idx': b_m['chunk_idx'],
                'A_chunk_text': a_m['text'], 'B_chunk_text': b_m['text'],
                'similarity': round(sim*100, 4)
            })
            if sim > para_sim[pa, pb]:
                para_sim[pa, pb] = sim
                evidence[pa][pb] = {
                    'A_chunk_text': a_m['text'],
                    'B_chunk_text': b_m['text'],
                    'similarity': round(sim*100, 4)
                }
    return para_sim, evidence, all_matches

# ---------- main ----------
def main(pdfA, pdfB):
    # 1. extract
    textA = extract_text_from_pdf_pypdf2(pdfA)
    textB = extract_text_from_pdf_pypdf2(pdfB)

    # 2. paragraphs
    parasA = split_paragraphs(textA)
    parasB = split_paragraphs(textB)
    print(f"Paragraphs: A={len(parasA)}, B={len(parasB)}")

    # 3. chunks
    A_chunks, A_map = chunk_paragraphs(parasA)
    B_chunks, B_map = chunk_paragraphs(parasB)
    print(f"Chunks: A={len(A_chunks)}, B={len(B_chunks)}")

    if len(A_chunks) == 0 or len(B_chunks) == 0:
        print("No chunks found; exiting.")
        return

    # 4. similarity - try Gemini embeddings first, fallback to TF-IDF
    try:
        print("Attempting Gemini embeddings...")
        A_emb = embed_with_gemini(A_chunks, model="gemini-embedding-001", batch_size=64)
        B_emb = embed_with_gemini(B_chunks, model="gemini-embedding-001", batch_size=64)
        chunk_sim = np.dot(A_emb, B_emb.T)  # cosine because normalized
        print("Gemini embeddings succeeded.")
    except Exception as e:
        print("Gemini embedding failed, falling back to TF-IDF. Error:", e)
        chunk_sim = compute_tfidf_similarity(A_chunks, B_chunks)

    # 5. aggregate
    num_A_para = len(parasA)
    num_B_para = len(parasB)
    para_sim, evidence, all_matches = aggregate_to_paragraph_level(chunk_sim, A_map, B_map, num_A_para, num_B_para)

    # 6. save outputs (percentages)
    para_percent = (para_sim * 100).round(4)
    out_dir = 'comparison_output'
    os.makedirs(out_dir, exist_ok=True)

    matrix_df = pd.DataFrame(para_percent,
                             index=[f"A_para_{i+1}" for i in range(num_A_para)],
                             columns=[f"B_para_{j+1}" for j in range(num_B_para)])
    matrix_csv = os.path.join(out_dir, 'paragraph_similarity_matrix.csv')
    matrix_df.to_csv(matrix_csv)
    print("Saved paragraph similarity matrix:", matrix_csv)

    # best matches per A paragraph (for quick human review)
    best_rows = []
    for i in range(num_A_para):
        j = para_sim[i].argmax()
        ev = evidence[i][j]
        best_rows.append({
            'A_paragraph': f"A_para_{i+1}",
            'Best_matching_B_paragraph': f"B_para_{j+1}",
            'Similarity_%': float(para_percent[i, j]),
            'A_chunk_excerpt': ev['A_chunk_text'] if ev else '',
            'B_chunk_excerpt': ev['B_chunk_text'] if ev else ''
        })
    best_df = pd.DataFrame(best_rows)
    best_csv = os.path.join(out_dir, 'best_matches_with_evidence.csv')
    best_df.to_csv(best_csv, index=False)
    print("Saved best-match report:", best_csv)

    # all chunk matches (detailed)
    all_df = pd.DataFrame(all_matches)
    all_csv = os.path.join(out_dir, 'all_chunk_matches.csv')
    all_df.to_csv(all_csv, index=False)
    print("Saved detailed chunk matches:", all_csv)

    print("\nDone. Check the 'comparison_output' folder for CSVs.")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python compare_pdfs_fullmatrix.py path/to/doc_A.pdf path/to/doc_B.pdf")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
