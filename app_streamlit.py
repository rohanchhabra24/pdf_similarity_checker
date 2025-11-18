
import streamlit as st
import numpy as np
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
from streamlit_plotly_events import plotly_events

import plotly.express as px
import plotly.graph_objects as go

from compare_pdfs_st_eval import run_models_and_compare
import tempfile
import os

# -------------------------------------------------------------
# PAGE CONFIG — CLEAN LIGHT UI
# -------------------------------------------------------------
st.set_page_config(
    page_title="DocCompare",
    page_icon="📄",
    layout="wide"
)

# -------------------------------------------------------------
# HEADER
# -------------------------------------------------------------
st.markdown(
    """
    <style>
        .main-title {
            font-size: 38px;
            font-weight: 700;
            color: #222;
        }
        .sub-title {
            font-size: 20px;
            font-weight: 400;
            color: #555;
            margin-top: -10px;
            margin-bottom: 30px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='main-title'>DocCompare</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Using: Sentence-Transformer — all-MiniLM-L6-v2</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# UPLOAD SECTION
# -------------------------------------------------------------
colA, colB = st.columns(2)

with colA:
    st.subheader("Upload Document A (PDF)")
    fileA = st.file_uploader(" ", type=["pdf"], label_visibility="collapsed")

with colB:
    st.subheader("Upload Document B (PDF)")
    fileB = st.file_uploader("  ", type=["pdf"], label_visibility="collapsed")

# Centered run button
st.write("")
st.write("")

center_col = st.columns([3, 1, 3])[1]
with center_col:
    run_btn = st.button("Run comparison", use_container_width=True)

# -------------------------------------------------------------
# RUN COMPARISON
# -------------------------------------------------------------
if run_btn:
    if not (fileA and fileB):
        st.error("Please upload **both** PDF files.")
    else:
        st.info("Running comparison… First run may take time if model downloads.")

        # Save PDFs temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpA:
            tmpA.write(fileA.read())
            A_path = tmpA.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpB:
            tmpB.write(fileB.read())
            B_path = tmpB.name

        # Call ST evaluation engine
        summary = run_models_and_compare(
            A_path, B_path,
            models=["all-MiniLM-L6-v2"],
            cache_dir=".emb_cache",
            out_root="comparison_output"
        )

        st.success("Done!")

        # -------------------------------------------------------------
        # LOAD MATRIX CSV
        # -------------------------------------------------------------
        model_folder = "comparison_output/all-MiniLM-L6-v2"
        matrix_csv_path = os.path.join(model_folder, "paragraph_similarity_matrix.csv")
        matrix_df = pd.read_csv(matrix_csv_path, index_col=0)

        st.subheader("Paragraph similarity heatmap")

        # ------------------ ROBUST HEATMAP WITH CELL LABELS ------------------
        # Build values for heatmap: ensure we use 0..100 percentages for labels
        mat_vals = matrix_df.values.astype(float)
        # If matrix in 0..1 range, convert to 0..100 for display
        display_vals = mat_vals * 100.0 if mat_vals.max() <= 1.5 else mat_vals.copy()

        # Create text annotations (percentages rounded)
        text_vals = np.array([[f"{v:.1f}%" for v in row] for row in display_vals])

        # Create heatmap using plotly.graph_objects so we can control text annotations
        fig = go.Figure(data=go.Heatmap(
            z=display_vals,
            x=matrix_df.columns.tolist(),
            y=matrix_df.index.tolist(),
            text=text_vals,
            hovertemplate="A: %{y}<br>B: %{x}<br>Similarity: %{z:.2f}%<extra></extra>",
            colorscale="Blues",
            colorbar=dict(title="%")
        ))

        # Add annotations explicitly so percentages appear in each block
        annotations = []
        rows, cols = display_vals.shape
        for i in range(rows):
            for j in range(cols):
                annotations.append(dict(
                    x=matrix_df.columns.tolist()[j],
                    y=matrix_df.index.tolist()[i],
                    text=f"{display_vals[i,j]:.0f}%",
                    showarrow=False,
                    font=dict(
                        color="white" if display_vals[i,j] > (display_vals.max()/2) else "black",
                        size=10
                    )
                ))
        fig.update_layout(
            annotations=annotations,
            xaxis_title="Document B paragraphs",
            yaxis_title="Document A paragraphs",
            height=600,
            margin=dict(l=40, r=40, t=40, b=40)
        )

        # -------------------------------------------------------------
        # SHOW ONE HEATMAP, STILL CAPTURE CLICKS
        # -------------------------------------------------------------
        # 1) Visible working heatmap
        st.plotly_chart(fig, use_container_width=True)

        # 2) Hidden/narrow plotly_events to capture clicks only
        click_col1, click_col2 = st.columns([10, 1])
        with click_col2:
            selected_points = plotly_events(
                fig,
                click_event=True,
                key="heatmap_single"
            )

        # -------------------------------------------------------------
        # LOAD ALL CHUNK MATCHES
        # -------------------------------------------------------------
        all_matches_csv = os.path.join(model_folder, "all_chunk_matches.csv")
        all_matches_df = pd.read_csv(all_matches_csv)

        st.subheader("Paragraph-level match table")

        # Clean AgGrid table
        gb = GridOptionsBuilder.from_dataframe(all_matches_df)
        gb.configure_pagination()
        gb.configure_side_bar()
        gb.configure_default_column(resizable=True, filter=True, sortable=True)
        grid_options = gb.build()

        AgGrid(all_matches_df, gridOptions=grid_options, theme="light")

        # -------------------------------------------------------------
        # EVIDENCE SECTION
        # -------------------------------------------------------------
        st.subheader("Evidence (matching chunk text)")

        if selected_points:
            # plotly_events returns points where 'x' is x label and 'y' is y label
            pt = selected_points[0]
            # try to obtain labels; when using px.imshow or go.Heatmap we get x/y labels
            x_label = pt.get("x")
            y_label = pt.get("y")

            # find indices
            try:
                col_idx = matrix_df.columns.tolist().index(x_label)
                row_idx = matrix_df.index.tolist().index(y_label)
            except Exception:
                # fallback: if pointIndex available (rare), try decode
                col_idx = None
                row_idx = None
                pi = pt.get("pointIndex")
                if isinstance(pi, list) and len(pi) >= 2:
                    col_idx = pi[0]
                    row_idx = pi[1]

            if row_idx is not None and col_idx is not None:
                st.info(f"Selected: **{matrix_df.index[row_idx]} ↔ {matrix_df.columns[col_idx]}**")
                matches = all_matches_df[
                    (all_matches_df["A_para"] == row_idx+1) &
                    (all_matches_df["B_para"] == col_idx+1)
                ].sort_values("similarity", ascending=False)

                if matches.empty:
                    st.write("No chunk-level evidence available for this pair.")
                else:
                    for idx, m in matches.head(5).iterrows():
                        st.markdown(f"**Similarity: {m.get('similarity', m.get('similarity_%', None))}%**")
                        st.markdown("**A chunk:**")
                        st.write(m.get('A_chunk_text', ''))
                        st.markdown("**B chunk:**")
                        st.write(m.get('B_chunk_text', ''))
                        st.markdown("---")
            else:
                st.info("Clicked location couldn't be resolved to paragraph indices.")
        else:
            st.info("Click a heatmap cell to view detailed matching chunks.")

# -------------------------------------------------------------
# HOW DOES THIS MODEL WORK?
# -------------------------------------------------------------
st.write("---")
st.header("How does this model work?")
st.markdown(
    """
### Simple explanation  
Think of each paragraph like a shape. The model turns every paragraph into a number-pattern (called *embeddings*).  
If two paragraph-patterns look the same, it means the paragraphs talk about similar things.

We then compare these shapes using **cosine similarity**, which gives a score from 0 to 100%.

---

### Technical explanation  
1. Each paragraph → split into sentence chunks.  
2. Each chunk is converted into an embedding vector using  
   **Sentence-Transformer: all-MiniLM-L6-v2**.  
3. We compute cosine similarity for every A-chunk vs B-chunk.  
4. For each paragraph pair (Aᵢ, Bⱼ), we aggregate chunk similarities (max or top evidence).  
5. This forms the **N×M paragraph similarity matrix**, visualized in the heatmap.
    """
)
