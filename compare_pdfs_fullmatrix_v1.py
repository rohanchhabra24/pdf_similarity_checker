import sys
import os
import re
import numpy as np
import pandas as pd

# --- Sentence Transformer import (required) ---
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    print("ERROR: sentence-transformers is not installed.")
    print("Install using: pip install sentence-transformers")
    sys.exit(1)

# --- Matplotlib (optional, for heatmap) ---
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# -------------------- PDF extraction --------------------
def extract_text_from_pdf(path):
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        print("Please install PyPDF2: pip install PyPDF2")
        sys.exit(1)

    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return "\n".join(pages)

# -------------------- Paragraph splitting --------------------
def split_paragraphs(text):
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Prefer blank-lines
    paras = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    if len(paras) > 1:
        return paras

    # Else fallback to sentence-grouping
    sentences = re.split(r'(?<=[.!?;])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return [text.strip()]

    paras = []
    n = 3  # every 3 sentences
    for i in range(0, len(sentences), n):
        paras.append(" ".join(sentences[i:i+n]))
    return paras

# -------------------- Sentence chunking --------------------
def split_sentences(paragraph):
    sents = re.split(r'(?<=[.!?;])\s+', paragraph.strip())
    return [s.strip() for s in sents if s.strip()]

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

# -------------------- Sentence-Transformer embedding --------------------
def embed_st(chunks):
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)

    print("Loading Sentence-Transformer model: all-MiniLM-L6-v2 ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Encoding chunks using Sentence-Transformers...")
    emb = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    return emb.astype(np.float32)

# -------------------- Aggregation --------------------
def aggregate_to_paragraph_level(chunk_sim, A_map, B_map, num_A_para, num_B_para):
    para_sim = np.zeros((num_A_para, num_B_para))
    evidence = [[None]*num_B_para for _ in range(num_A_para)]
    all_matches = []

    M, N = chunk_sim.shape
    for a_idx in range(M):
        for b_idx in range(N):
            sim = float(chunk_sim[a_idx, b_idx])
            pa = A_map[a_idx]['para_idx']
            pb = B_map[b_idx]['para_idx']

            all_matches.append({
                'A_para': pa+1,
                'B_para': pb+1,
                'A_chunk_idx': A_map[a_idx]['chunk_idx'],
                'B_chunk_idx': B_map[b_idx]['chunk_idx'],
                'A_chunk_text': A_map[a_idx]['text'],
                'B_chunk_text': B_map[b_idx]['text'],
                'similarity': round(sim*100, 4)
            })

            if sim > para_sim[pa, pb]:
                para_sim[pa, pb] = sim
                evidence[pa][pb] = {
                    'A_chunk_text': A_map[a_idx]['text'],
                    'B_chunk_text': B_map[b_idx]['text'],
                    'similarity': round(sim*100, 4)
                }

    return para_sim, evidence, all_matches

# -------------------- Heatmap --------------------
def save_heatmap(matrix_df, out_path):
    if not HAS_MPL:
        return None

    plt.figure(figsize=(10,6))
    plt.imshow(matrix_df.values, aspect="auto", interpolation="nearest")
    plt.xticks(np.arange(matrix_df.shape[1]), matrix_df.columns, rotation=45)
    plt.yticks(np.arange(matrix_df.shape[0]), matrix_df.index)
    plt.colorbar(label="Similarity (%)")
    plt.title("Paragraph Similarity Heatmap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path

# -------------------- Main --------------------
def main(pdfA, pdfB):

    print("\n=== Extracting text from PDFs ===")
    textA = extract_text_from_pdf(pdfA)
    textB = extract_text_from_pdf(pdfB)

    print("=== Splitting into paragraphs ===")
    parasA = split_paragraphs(textA)
    parasB = split_paragraphs(textB)
    print(f"Paragraphs: A={len(parasA)}, B={len(parasB)}")

    print("=== Splitting into chunks ===")
    A_chunks, A_map = chunk_paragraphs(parasA)
    B_chunks, B_map = chunk_paragraphs(parasB)
    print(f"Chunks: A={len(A_chunks)}, B={len(B_chunks)}")

    # STRICTLY USE SENTENCE-TRANSFORMERS ONLY
    print("\n=== Using Sentence-Transformers ONLY ===")
    A_emb = embed_st(A_chunks)
    B_emb = embed_st(B_chunks)

    print("Embedding shapes:", A_emb.shape, B_emb.shape)

    print("=== Computing cosine similarity ===")
    chunk_sim = np.dot(A_emb, B_emb.T)
    chunk_sim = np.clip(chunk_sim, 0.0, 1.0)

    print("=== Aggregating to paragraph-level ===")
    para_sim, evidence, all_matches = aggregate_to_paragraph_level(
        chunk_sim, A_map, B_map, len(parasA), len(parasB)
    )

    para_percent = (para_sim * 100).round(4)

    out_dir = "comparison_output"
    os.makedirs(out_dir, exist_ok=True)

    # Save matrix
    matrix_df = pd.DataFrame(
        para_percent,
        index=[f"A_para_{i+1}" for i in range(len(parasA))],
        columns=[f"B_para_{j+1}" for j in range(len(parasB))]
    )
    matrix_csv = os.path.join(out_dir, "paragraph_similarity_matrix.csv")
    matrix_df.to_csv(matrix_csv)
    print("Saved:", matrix_csv)

    # Save all chunk matches
    all_df = pd.DataFrame(all_matches)
    all_csv = os.path.join(out_dir, "all_chunk_matches.csv")
    all_df.to_csv(all_csv, index=False)
    print("Saved:", all_csv)

    # Best matches
    best_rows = []
    for i in range(len(parasA)):
        best_j = int(para_sim[i].argmax())
        ev = evidence[i][best_j]
        best_rows.append({
            'A_paragraph': f"A_para_{i+1}",
            'Best_matching_B_paragraph': f"B_para_{best_j+1}",
            'Similarity_%': float(para_percent[i, best_j]),
            'A_chunk_excerpt': ev['A_chunk_text'] if ev else "",
            'B_chunk_excerpt': ev['B_chunk_text'] if ev else ""
        })

    best_df = pd.DataFrame(best_rows)
    best_csv = os.path.join(out_dir, "best_matches_with_evidence.csv")
    best_df.to_csv(best_csv, index=False)
    print("Saved:", best_csv)

    # Heatmap
    if HAS_MPL:
        hm_path = os.path.join(out_dir, "paragraph_similarity_heatmap.png")
        save_heatmap(matrix_df, hm_path)
        print("Saved heatmap:", hm_path)

    print("\n=== DONE! All results saved in comparison_output/ ===\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_pdfs_st_only.py docA.pdf docB.pdf")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
