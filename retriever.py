# retriever.py

import numpy as np

def cosine_similarity(query, database):
    query = query / np.linalg.norm(query)
    database = database / np.linalg.norm(database, axis=1, keepdims=True)
    return database @ query

def retrieve(query_vector, embeddings, top_k=12):
    sims = cosine_similarity(query_vector, embeddings)
    indices = np.argsort(-sims)[:top_k]
    return [(i, sims[i]) for i in indices]
