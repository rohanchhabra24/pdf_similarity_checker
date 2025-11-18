#!/usr/bin/env python3
"""
compare_pdfs_st_eval.py

Usage:
    python compare_pdfs_st_eval.py docA.pdf docB.pdf
    python compare_pdfs_st_eval.py docA.pdf docB.pdf --models all-MiniLM-L6-v2

This script:
 - extracts text from two PDFs,
 - splits into paragraphs and sentence-level chunks,
 - embeds chunks with sentence-transformers models (with per-chunk caching),
 - computes chunk-wise cosine similarities and aggregates to paragraph-level matrix,
 - writes CSV outputs under comparison_output/<model_name>/

Requirements:
    pip install sentence-transformers PyPDF2 pandas scikit-learn numpy python-dotenv
"""

import os
import sys
import argparse
import hashlib
import time
from pathlib import Path
import re
import numpy as np
import pandas as pd

# optional plotting imports are not required for this script (frontend will render heatmap)
# but keep matplotlib import optional
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


# -------------------------
# PDF text extraction
# -------------------------
def extract_text_from_pdf(path: str) -> str:
    try:
        from PyPDF2 import PdfReader
    except Exception as e:
        raise RuntimeError("Install PyPDF2: pip install PyPDF2") from e
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return "\n".join(pages)


# -------------------------
# paragraph / sentence splitting
# -------------------------
def split_paragraphs(text: str):
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    paras = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    if len(paras) > 1:
        return paras

    # fallback: group lines heuristically
    lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
    if len(lines) > 1:
        grouped = []
        cur = []
        for ln in lines:
            cur.append(ln)
            if re.search(r'[.!?;:]$', ln):
                grouped.append(" ".join(cur))
                cur = []
        if cur:
            grouped.append(" ".join(cur))
        if len(grouped) > 1:
            return grouped

    # final fallback: group every N sentences
    sents = re.split(r'(?<=[.!?;])\s+', text)
    sents = [s.strip() for s in sents if s.strip()]
    if not sents:
        return [text.strip()]
    n = 3
    paras = []
    for i in range(0, len(sents), n):
        paras.append(" ".join(sents[i:i+n]))
    return paras


def split_sentences(paragraph: str):
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


# -------------------------
# caching utilities
# -------------------------
def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode('utf-8')).hexdigest()


def ensure_cache_dir(base_cache_dir: str, model_name: str) -> Path:
    model_hash = hashlib.sha1(model_name.encode('utf-8')).hexdigest()[:10]
    d = Path(base_cache_dir) / f"{model_hash}_{model_name.replace('/', '_')}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_cached_embedding(cache_dir: Path, chunk_sha: str):
    p = cache_dir / (chunk_sha + ".npy")
    if p.exists():
        return np.load(str(p))
    return None


def save_cached_embedding(cache_dir: Path, chunk_sha: str, arr: np.ndarray):
    p = cache_dir / (chunk_sha + ".npy")
    np.save(str(p), arr)


# -------------------------
# embeddings with sentence-transformers
# -------------------------
def embed_chunks_with_model(chunks, model_name, cache_dir, batch_size=64):
    """
    Returns numpy array of embeddings shape (len(chunks), dim)
    Uses per-chunk caching to avoid re-embedding identical text.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError("Install sentence-transformers: pip install sentence-transformers") from e

    model = SentenceTransformer(model_name)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    embeddings = [None] * len(chunks)
    to_batch = []
    idx_map = []

    for i, ch in enumerate(chunks):
        sha = sha1_text(ch)
        emb = load_cached_embedding(cache_dir, sha)
        if emb is not None:
            embeddings[i] = emb
        else:
            idx_map.append(i)
            to_batch.append(ch)

    if to_batch:
        encoded = model.encode(to_batch, convert_to_numpy=True, normalize_embeddings=True, batch_size=batch_size)
        k = 0
        for pos in idx_map:
            emb_v = np.asarray(encoded[k], dtype=np.float32)
            embeddings[pos] = emb_v
            save_cached_embedding(cache_dir, sha1_text(chunks[pos]), emb_v)
            k += 1

    # stack and normalize (in case some cache entries were saved without normalization)
    arr = np.vstack(embeddings).astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    arr = arr / norms
    return arr


# -------------------------
# similarity & aggregation
# -------------------------
def compute_chunk_similarity(A_emb, B_emb):
    return np.dot(A_emb, B_emb.T)


def aggregate_chunk_to_paragraph(chunk_sim, A_map, B_map, num_A_para, num_B_para, agg="max", topk=1):
    para_sim = np.zeros((num_A_para, num_B_para))
    evidence = [[None]*num_B_para for _ in range(num_A_para)]
    all_matches = []

    M, N = chunk_sim.shape
    # create bins
    sims_bin = [[[] for _ in range(num_B_para)] for _ in range(num_A_para)]
    for a in range(M):
        for b in range(N):
            pa = A_map[a]['para_idx']; pb = B_map[b]['para_idx']
            val = float(chunk_sim[a, b])
            sims_bin[pa][pb].append((val, A_map[a]['text'], B_map[b]['text']))

            all_matches.append({
                'A_para': pa+1, 'B_para': pb+1,
                'A_chunk_idx': A_map[a]['chunk_idx'], 'B_chunk_idx': B_map[b]['chunk_idx'],
                'A_chunk_text': A_map[a]['text'], 'B_chunk_text': B_map[b]['text'],
                'similarity': round(val*100, 4)
            })

    for i in range(num_A_para):
        for j in range(num_B_para):
            lst = sims_bin[i][j]
            if not lst:
                para_sim[i, j] = 0.0
                evidence[i][j] = None
                continue
            lst_sorted = sorted(lst, key=lambda x: x[0], reverse=True)
            if agg == "max":
                val, a_txt, b_txt = lst_sorted[0]
            elif agg == "mean":
                val = float(np.mean([x[0] for x in lst_sorted]))
                a_txt, b_txt = lst_sorted[0][1], lst_sorted[0][2]
            elif agg == "topk_mean":
                k = min(topk, len(lst_sorted))
                val = float(np.mean([x[0] for x in lst_sorted[:k]]))
                a_txt, b_txt = lst_sorted[0][1], lst_sorted[0][2]
            else:
                val, a_txt, b_txt = lst_sorted[0]
            para_sim[i, j] = val
            evidence[i][j] = {'A_chunk_text': a_txt, 'B_chunk_text': b_txt, 'similarity': round(val*100, 4)}
    return para_sim, evidence, all_matches


# -------------------------
# save utilities
# -------------------------
def save_matrix_and_reports(out_dir: Path, model_name: str, para_percent, evidence, all_matches):
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_df = pd.DataFrame(para_percent, index=[f"A_para_{i+1}" for i in range(para_percent.shape[0])],
                             columns=[f"B_para_{j+1}" for j in range(para_percent.shape[1])])
    matrix_csv = out_dir / "paragraph_similarity_matrix.csv"
    matrix_df.to_csv(matrix_csv)
    pd.DataFrame(all_matches).to_csv(out_dir / "all_chunk_matches.csv", index=False)
    # best matches
    best_rows = []
    for i in range(para_percent.shape[0]):
        j = int(np.argmax(para_percent[i]))
        ev = evidence[i][j]
        best_rows.append({'A_paragraph': f"A_para_{i+1}", 'Best_matching_B_paragraph': f"B_para_{j+1}",
                          'Similarity_%': float(para_percent[i, j]),
                          'A_chunk_excerpt': ev['A_chunk_text'] if ev else '',
                          'B_chunk_excerpt': ev['B_chunk_text'] if ev else ''})
    pd.DataFrame(best_rows).to_csv(out_dir / "best_matches_with_evidence.csv", index=False)
    return str(matrix_csv), str(out_dir / "best_matches_with_evidence.csv"), str(out_dir / "all_chunk_matches.csv")


# -------------------------
# main experiment
# -------------------------
def run_models_and_compare(pdfA, pdfB, models, cache_dir=".emb_cache", out_root="comparison_output",
                           agg='max', topk=1, batch_size=64):
    textA = extract_text_from_pdf(pdfA)
    textB = extract_text_from_pdf(pdfB)
    parasA = split_paragraphs(textA)
    parasB = split_paragraphs(textB)
    A_chunks, A_map = chunk_paragraphs(parasA)
    B_chunks, B_map = chunk_paragraphs(parasB)
    num_A = len(parasA); num_B = len(parasB)
    print(f"Paragraphs: A={num_A}, B={num_B}; Chunks: A={len(A_chunks)}, B={len(B_chunks)}")

    summary_rows = []
    out_root_path = Path(out_root)
    out_root_path.mkdir(parents=True, exist_ok=True)

    for model_name in models:
        print("-----")
        print("Model:", model_name)
        model_cache = ensure_cache_dir(cache_dir, model_name)
        t0 = time.time()
        A_emb = embed_chunks_with_model(A_chunks, model_name, model_cache, batch_size=batch_size)
        B_emb = embed_chunks_with_model(B_chunks, model_name, model_cache, batch_size=batch_size)
        embed_time = time.time() - t0
        print(f"Encoded embeddings in {embed_time:.1f}s; shapes: {A_emb.shape}, {B_emb.shape}")

        chunk_sim = compute_chunk_similarity(A_emb, B_emb)
        para_sim, evidence, all_matches = aggregate_chunk_to_paragraph(chunk_sim, A_map, B_map, num_A, num_B, agg=agg, topk=topk)
        para_percent = para_sim  # values in 0..1

        # save outputs under model-specific folder
        safe_model_name = model_name.replace('/', '_')
        out_dir = out_root_path / safe_model_name
        matrix_csv, best_csv, all_csv = save_matrix_and_reports(out_dir, model_name, para_percent, evidence, all_matches)

        # summary metrics
        def compute_summary_metrics(para_sim_mat):
            m = para_sim_mat.shape
            min_n = min(m[0], m[1])
            diag_vals = [para_sim_mat[i, i] for i in range(min_n)]
            diag_mean = float(np.mean(diag_vals)) if diag_vals else 0.0
            off = []
            for i in range(m[0]):
                for j in range(m[1]):
                    if i < min_n and i == j:
                        continue
                    off.append(para_sim_mat[i, j])
            off_mean = float(np.mean(off)) if off else 0.0
            separation = diag_mean - off_mean
            max_per_A = [float(np.max(para_sim_mat[i])) for i in range(m[0])]
            mean_max = float(np.mean(max_per_A)) if max_per_A else 0.0
            return {'diag_mean': diag_mean, 'off_mean': off_mean, 'separation': separation, 'mean_max_per_A': mean_max}

        metrics = compute_summary_metrics(para_sim)
        metrics.update({'model': model_name, 'embed_time_s': round(embed_time, 2)})
        summary_rows.append(metrics)
        print("Saved results to:", out_dir)

    summary_df = pd.DataFrame(summary_rows).sort_values('separation', ascending=False)
    summary_csv = out_root_path / "models_comparison_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print("Wrote comparison summary:", summary_csv)
    return summary_df


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("pdfA")
    p.add_argument("pdfB")
    p.add_argument("--models", nargs='+', default=["all-MiniLM-L6-v2"],
                   help="one or more sentence-transformer model names (HuggingFace / sentence-transformers names)")
    p.add_argument("--cache-dir", default=".emb_cache", help="folder to cache embeddings")
    p.add_argument("--out-root", default="comparison_output", help="root output folder")
    p.add_argument("--agg", choices=['max', 'mean', 'topk_mean'], default='max', help="paragraph aggregation strategy")
    p.add_argument("--topk", type=int, default=1, help="k for topk_mean")
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df = run_models_and_compare(args.pdfA, args.pdfB, args.models, cache_dir=args.cache_dir,
                                out_root=args.out_root, agg=args.agg, topk=args.topk, batch_size=args.batch_size)
    print(df.to_string(index=False))
