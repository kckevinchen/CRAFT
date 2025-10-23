import numpy as np
import faiss
from tqdm import tqdm
from typing import Tuple, Dict, Any, Literal

def apply_faiss_exact_nn(
    A: np.ndarray, 
    k_neighbors: int = 10, 
    batch_size: int = 1024
) -> np.ndarray:
    """
    Applies Faiss for exact nearest neighbor search using batch processing.
    
    This optimized version removes the slow post-processing loop by
    vectorized slicing.
    """
    num_vectors, dimension = A.shape
    A_float32 = np.ascontiguousarray(A.astype('float32'))

    # Use IndexFlatIP for exact Inner Product (cosine similarity)
    index = faiss.IndexFlatIP(dimension)
    index.add(A_float32)

    # Pre-allocate array for *only* the k neighbors (not k+1)
    all_indices = np.zeros((num_vectors, k_neighbors), dtype='int64')

    for i in tqdm(range(0, num_vectors, batch_size), desc="Searching batches"):
        end_idx = min(i + batch_size, num_vectors)
        
        # Search for k+1 neighbors (to include the vector itself)
        _, idx = index.search(A_float32[i:end_idx], k_neighbors + 1)
        
        # Store the k neighbors, *skipping the first result (self-index)*
        all_indices[i:end_idx, :] = idx[:, 1:]

    return all_indices

def apply_faiss_lsh(
    A_tilde: np.ndarray, 
    k_neighbors: int = 50, 
    nbits_factor: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies Faiss LSH and returns the k nearest neighbors.
    
    This fixed version correctly removes the self-index from results.
    """
    num_vectors, dimension = A_tilde.shape
    nbits = dimension * nbits_factor
    A_tilde_float32 = np.ascontiguousarray(A_tilde.astype('float32'))

    index = faiss.IndexLSH(dimension, nbits)
    index.add(A_tilde_float32)

    # Search for k+1 to account for the self-index
    distances, indices = index.search(A_tilde_float32, k_neighbors + 1)

    # Return the k neighbors, skipping the first (self-index)
    return distances[:, 1:], indices[:, 1:]

def apply_faiss_hnsw_direct(
    A_reduced: np.ndarray, 
    k_neighbors: int = 50, 
    M: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies Faiss HNSW on an *already reduced* data matrix.
    
    This fixed version removes the redundant 'reduced_dim' parameter.
    """
    # Get dimension directly from the input matrix
    num_vectors, dimension = A_reduced.shape
    A_reduced_float32 = np.ascontiguousarray(A_reduced.astype('float32'))

    index = faiss.IndexHNSWFlat(dimension, M)
    index.add(A_reduced_float32)

    # Search for k+1 to account for the self-index
    distances, indices = index.search(A_reduced_float32, k_neighbors + 1)
    
    # Return the k neighbors, skipping the first (self-index)
    return distances[:, 1:], indices[:, 1:]

def build_candidate_graph(
    B_norm: np.ndarray, 
    k_neighbors: int, 
    method: Literal['hnsw', 'lsh'] = 'hnsw', 
    M: int = 32, 
    nbits_factor: int = 4
) -> Dict[int, Tuple[list, list]]:
    """
    Builds a Faiss index and returns a candidate graph as a dictionary.
    
    This function consolidates apply_faiss_hnsw and apply_faiss_lsh_complex.
    It handles complex inputs by taking their absolute value.
    """
    # Handle complex or real inputs, convert to float32
    if np.iscomplexobj(B_norm):
        data = np.ascontiguousarray(np.abs(B_norm).astype('float32'))
    else:
        data = np.ascontiguousarray(B_norm.astype('float32'))
        
    num_vectors, dimension = data.shape

    # 1. Build the appropriate index
    if method == 'hnsw':
        index = faiss.IndexHNSWFlat(dimension, M)
    elif method == 'lsh':
        nbits = dimension * nbits_factor
        index = faiss.IndexLSH(dimension, nbits)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'hnsw' or 'lsh'.")

    index.add(data)
    
    # 2. Search for k+1 neighbors
    distances, indices = index.search(data, k_neighbors + 1)
    
    # 3. Build the candidate graph (same as your original logic)
    candidate_graph = {}
    for i in range(num_vectors):
        filtered_neighbors = []
        filtered_distances = []
        for idx, dist in zip(indices[i], distances[i]):
            if idx != i:
                filtered_neighbors.append(idx)
                filtered_distances.append(dist)
        
        # Store the top k_neighbors (L)
        candidate_graph[i] = (
            filtered_neighbors[:k_neighbors], 
            filtered_distances[:k_neighbors]
        )
        
    return candidate_graph