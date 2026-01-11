# app.py

import streamlit as st
import json
import numpy as np
from PIL import Image

from clip_encoder import CLIPEncoder
from retriever import retrieve
from rocchio import update_query
from config import TOP_K

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(layout="wide")
st.title("Fashion Image Retrieval with Relevance Feedback")

# -------------------------------
# Caching utilities
# -------------------------------
@st.cache_resource
def load_encoder():
    return CLIPEncoder()

@st.cache_resource
def load_data():
    embeddings = np.load("features/image_embeddings.npy")
    with open("features/image_paths.json") as f:
        paths = json.load(f)
    return embeddings, paths

@st.cache_data
def load_image(path):
    return Image.open(path).convert("RGB")

encoder = load_encoder()
embeddings, image_paths = load_data()

# -------------------------------
# Session state initialization
# -------------------------------
if "query_vector" not in st.session_state:
    st.session_state.query_vector = None
    st.session_state.results = []
    st.session_state.rel = set()
    st.session_state.irrel = set()
    st.session_state.iteration = 0

# -------------------------------
# Query input
# -------------------------------
uploaded = st.file_uploader("Upload query image", type=["jpg", "png"])
text_feedback = st.text_input("Optional text feedback (e.g., 'more formal', 'long sleeves')")

if uploaded:
    query_image = Image.open(uploaded).convert("RGB")
    st.image(query_image, width=300, caption="Query Image")
    
    if st.button("Search"):
        query_image.save("query.jpg")
        q = encoder.encode_image("query.jpg").numpy()
       
        st.session_state.query_vector = q
        st.session_state.results = retrieve(q, embeddings, TOP_K)
        st.session_state.rel.clear()
        st.session_state.irrel.clear()
        st.session_state.iteration = 0

# -------------------------------
# Display results
# -------------------------------
if st.session_state.results:
    st.subheader(f"Results (Iteration {st.session_state.iteration})")

    cols = st.columns(4)

    for idx, (img_idx, score) in enumerate(st.session_state.results):
        with cols[idx % 4]:
            st.image(
                load_image(image_paths[img_idx]),
                width='stretch',
            )

            is_rel = st.checkbox(
                "Relevant",
                key=f"rel_{img_idx}",
            )
            is_irrel = st.checkbox(
                "Irrelevant",
                key=f"irr_{img_idx}",
            )

            # Update feedback sets (NO on_change)
            if is_rel:
                st.session_state.rel.add(img_idx)
            else:
                st.session_state.rel.discard(img_idx)

            if is_irrel:
                st.session_state.irrel.add(img_idx)
            else:
                st.session_state.irrel.discard(img_idx)

    # -------------------------------
    # Refinement step
    # -------------------------------
    if st.button("Refine Search"):
        rel_vecs = (
            embeddings[list(st.session_state.rel)]
            if st.session_state.rel
            else []
        )
        irr_vecs = (
            embeddings[list(st.session_state.irrel)]
            if st.session_state.irrel
            else []
        )

        text_vec = None
        if text_feedback.strip():
            text_vec = encoder.encode_text(text_feedback).numpy()

        q_new = update_query(
            st.session_state.query_vector,
            rel_vecs,
            irr_vecs,
            text_vec,
        )

        st.session_state.query_vector = q_new
        st.session_state.results = retrieve(q_new, embeddings, TOP_K)
        st.session_state.rel.clear()
        st.session_state.irrel.clear()
        st.session_state.iteration += 1
        st.rerun()

    # -------------------------------
    # Reset button (demo-friendly)
    # -------------------------------
    if st.button("Reset Search"):
        st.session_state.query_vector = None
        st.session_state.results = []
        st.session_state.rel.clear()
        st.session_state.irrel.clear()
        st.session_state.iteration = 0
        st.rerun()
