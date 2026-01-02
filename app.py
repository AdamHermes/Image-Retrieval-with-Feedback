# app.py

import streamlit as st
import json
import numpy as np
from PIL import Image

from clip_encoder import CLIPEncoder
from retriever import retrieve
from rocchio import update_query
from config import TOP_K

st.set_page_config(layout="wide")
st.title("Fashion Image Retrieval with Relevance Feedback")

@st.cache_resource
def load_data():
    embeddings = np.load("features/image_embeddings.npy")
    with open("features/image_paths.json") as f:
        paths = json.load(f)
    return embeddings, paths

embeddings, image_paths = load_data()
encoder = CLIPEncoder()

if "query_vector" not in st.session_state:
    st.session_state.query_vector = None
    st.session_state.results = []
    st.session_state.rel = set()
    st.session_state.irrel = set()

uploaded = st.file_uploader("Upload query image", type=["jpg", "png"])
text_feedback = st.text_input("Optional text feedback")

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, width=300)

    if st.button("Search"):
        image.save("query.jpg")
        q = encoder.encode_image("query.jpg").numpy()
        st.session_state.query_vector = q
        st.session_state.results = retrieve(q, embeddings, TOP_K)
        st.session_state.rel.clear()
        st.session_state.irrel.clear()

if st.session_state.results:
    st.subheader("Results")
    cols = st.columns(4)

    for idx, (img_idx, score) in enumerate(st.session_state.results):
        with cols[idx % 4]:
            st.image(image_paths[img_idx], use_container_width=True)
            st.checkbox(
                "Relevant",
                key=f"rel_{img_idx}",
                on_change=lambda i=img_idx: st.session_state.rel.add(i),
            )
            st.checkbox(
                "Irrelevant",
                key=f"irr_{img_idx}",
                on_change=lambda i=img_idx: st.session_state.irrel.add(i),
            )

    if st.button("Refine Search"):
        rel_vecs = embeddings[list(st.session_state.rel)] if st.session_state.rel else []
        irr_vecs = embeddings[list(st.session_state.irrel)] if st.session_state.irrel else []

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
