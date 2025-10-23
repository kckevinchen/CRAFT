# src/preprocessing.py
import numpy as np
from sklearn.preprocessing import Normalizer
from scipy.sparse.base import spmatrix

def normalize(A: np.ndarray) -> np.ndarray:
    """Normalizes the rows of a dense matrix to unit L2 length."""
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    norms[norms == 0] = 1.0 # Avoid division by zero
    A = A / norms
    return A

def normalize_sparse(A: spmatrix) -> spmatrix:
    """Normalizes the rows of a sparse matrix to unit L2 length."""
    # This is the most efficient way for sparse matrices
    pre_normalizer = Normalizer(norm='l2')
    A = pre_normalizer.fit_transform(A)
    return A