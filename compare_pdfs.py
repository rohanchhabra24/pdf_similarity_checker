import sys, os, re
import numpy as np
import pandas as pd

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
    # Normalize newlines
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # First attempt: split on two or more newlines
    paras = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    if len(paras) > 1:
        return paras

    # If PDF had no blank lines, split by single newlines but group contiguous non-empty lines
    lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
    if len(lines) > 1:
        grouped = []
        current = []
        for ln in lines:
            current.append(ln)
            # Heuristic: if line ends with punctuation, treat it as end of small paragraph with some probability
            if re.search(r'[.!?;:]$', ln):
                # keep grouping small amount; we use short grouping to avoid huge paras
                if len(current) >= 1:
                    grouped.append(" ".join(current))
                    current = []
        if current:
            grouped.append(" ".join(current))
        if len(grouped) > 1:
            return grouped

    # Final fallback: split into pseudo-paragraphs by sentences (rough grouping)
    sents = re.split(r'(?<=[.!?;])\s+', text)
    sents = [s.strip() for s in sents if s.strip()]
    if len(sents) == 0:
        return [text.strip()]
    # Group every n sentences into a paragraph (n=3)
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
            # treat whole paragraph as one chunk
            chunks.append(p)
            chunk_map.append({'para_idx': pi, 'chunk_idx': 0, 'text': p})
        else:
            for ci, s in enumerate(sents):
                chunks.append(s)
                chunk_map.append({'para_idx': pi, 'chunk_idx': ci, 'text': s})
    return chunks, chunk_map

# ---------- embeddings / similarity ----------
def try_load_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception:
        return None

def compute_semantic_similarity(A_chunks, B_chunks, model):
    # model.encode(..., normalize_embeddings=True) available in recent versions
    A_emb = model.encode(A_chunks, convert_to_numpy=True, normalize_embeddings=True)
    B_emb = model.encode(B_chunks, convert_to_numpy=True, normalize_embeddings=True)
    sim = np.dot(A_emb, B_emb.T)  # cosine because normalized
    return sim

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
    # For producing a detailed all_chunk_matches list
    all_matches = []

    for a_idx, a_m in enumerate(A_map):
        for b_idx, b_m in enumerate(B_map):
            sim = float(chunk_sim[a_idx, b_idx])
            pa = a_m['para_idx']
            pb = b_m['para_idx']
            # store detailed row for potential inspection
            all_matches.append({
                'A_para': pa+1, 'B_para': pb+1,
                'A_chunk_idx': a_m['chunk_idx'], 'B_chunk_idx': b_m['chunk_idx'],
                'A_chunk_text': a_m['text'], 'B_chunk_text': b_m['text'],
                'similarity': round(sim*100, 4)
            })
            # update max for paragraph pair (we want max evidence)
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

    if len(A_chunks)==0 or len(B_chunks)==0:
        print("No chunks found; exiting.")
        return

    # 4. similarity
    model = try_load_sentence_transformer()
    if model:
        print("Using semantic embeddings (sentence-transformers).")
        chunk_sim = compute_semantic_similarity(A_chunks, B_chunks, model)
    else:
        print("Sentence-transformers not available; using TF-IDF fallback.")
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
            'Similarity_%': float(para_percent[i,j]),
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
