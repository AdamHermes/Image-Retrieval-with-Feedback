# rocchio.py

import numpy as np
from config import ALPHA, BETA, GAMMA, DELTA

def update_query(
    query,
    relevant_vectors,
    irrelevant_vectors,
    text_vector=None,
):
    new_query = ALPHA * query

    if len(relevant_vectors) > 0:
        new_query += BETA * np.mean(relevant_vectors, axis=0)

    if len(irrelevant_vectors) > 0:
        new_query -= GAMMA * np.mean(irrelevant_vectors, axis=0)

    if text_vector is not None:
        new_query += DELTA * text_vector

    new_query = new_query / np.linalg.norm(new_query)
    return new_query
