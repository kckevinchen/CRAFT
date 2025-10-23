import numpy as np
from tqdm import tqdm
from scipy import stats
from typing import List, Dict, Any, Iterable, Set, Tuple
from .omp_solver import solve_OMP
from .statistical_test import fisher_z_test, benjamini_hochberg

def compute_similarity_matrix(norm_matrix: np.ndarray) -> np.ndarray:
    """
    Computes the cosine similarity matrix (Gram matrix) for both real and
    complex vectors. Assumes rows are already L2-normalized.
    """
    if np.iscomplexobj(norm_matrix):
        # For complex vectors, similarity is the magnitude of the dot product
        return np.abs(norm_matrix @ norm_matrix.conj().T)
    else:
        # For real vectors, similarity is a standard dot product
        return norm_matrix @ norm_matrix.T

def rank_pairs_by_similarity(
    similarity_matrix: np.ndarray, 
    vocabulary: List[str]
) -> List[Dict[str, Any]]:
    """
    (Baseline Method) Ranks all possible pairs by similarity.
    
    WARNING: This is a brute-force O(N^2) operation and is extremely slow
    for large vocabularies. Use only for small-scale tests or baselines.
    """
    num_terms = len(vocabulary)
    ranked_pairs = []
    print("Ranking all pairs (O(N^2) brute-force)...")
    for i in tqdm(range(num_terms)):
        for j in range(i + 1, num_terms): # Avoid self-pairs and duplicates
            pair = tuple(sorted((vocabulary[i], vocabulary[j])))
            similarity = similarity_matrix[i, j]
            ranked_pairs.append({'pair': pair, 'similarity': similarity})

    ranked_pairs.sort(key=lambda x: x['similarity'], reverse=True)
    return ranked_pairs

def generate_candidate_pairs(
    indices: np.ndarray, 
    A_norm: np.ndarray
) -> List[Dict[str, Any]]:
    """
    Generates candidate pairs and re-ranks them by true similarity.
    
    Takes Faiss neighbor indices and computes the true dot product
    for each candidate pair.
    """
    num_terms = len(indices)
    candidates = []
    candidate_pairs_set = set()
    
    for i in tqdm(range(num_terms), desc="Generating/re-ranking candidates"):
        # Iterate through the neighbors of term i
        for neighbor_idx in indices[i]:
            # Ensure the neighbor index is valid
            if neighbor_idx < 0 or neighbor_idx >= num_terms or neighbor_idx == i:
                continue
                
            pair = tuple(sorted((i, neighbor_idx)))
            if pair not in candidate_pairs_set:
                candidate_pairs_set.add(pair)
                # Re-calculate true similarity (dot product)
                similarity = np.abs(A_norm[i] @ A_norm[neighbor_idx].conj().T)
                candidates.append({'pair': pair, 'similarity': similarity})

    return candidates

def extract_candidate_pairs_from_graph(
    L_graph: Dict[int, Tuple[list, list]], 
    B_norm: np.ndarray
) -> List[Dict[str, Any]]:
    """
    Extracts unique candidate pairs from a graph and computes their true similarity.
    
    This is used when the candidate source is a graph (e.g., from OMP).
    """
    candidates = []
    tested_pairs = set()

    for i, (neighbor_indices, _) in tqdm(L_graph.items(), desc="Extracting Candidates"):
        for j in neighbor_indices:
            pair = tuple(sorted((i, j)))
            if pair not in tested_pairs:
                tested_pairs.add(pair)
                # Re-calculate true similarity (dot product)
                similarity = np.abs(B_norm[i] @ B_norm[j].conj().T)
                candidates.append({'pair': pair, 'similarity': similarity})
                
    return candidates

def solve_OMP_for_candidates(
    L_graph: Dict[int, Tuple[list, list]], 
    B: np.ndarray, 
    epsilon_scale: float = 1.0, 
    max_iters: int = 10
) -> List[Dict[str, Any]]:
    """
    Recovers sparse correlation vectors for each term using OMP.
    The similarity score is the recovered sparse coefficient.
    """
    N, k = B.shape
    
    # Pre-compute normalization factors
    norms_squared = np.sum(np.abs(B)**2, axis=1)
    P_diag = norms_squared / k
    norm_factors = np.sqrt(P_diag + 1e-12)
    epsilon = epsilon_scale * np.sqrt(k)

    candidates = []
    tested_pairs = set()

    for i in tqdm(range(N), desc="Solving OMP (Sequential)"):
        l_graph_i = L_graph.get(i)
        if not l_graph_i:
            continue

        candidate_indices = l_graph_i[0]
        if not candidate_indices:
            continue

        # Construct local matrices for the OMP solver
        B_local = B[candidate_indices, :]
        norm_factors_local = norm_factors[candidate_indices]
        Psi_local = (B_local.T / norm_factors_local)
        z_i = B[i, :] / norm_factors[i]

        x_hat = solve_OMP(Psi_local, z_i, epsilon, max_iters=max_iters)

        for c, j in enumerate(candidate_indices):
            pair = tuple(sorted((i, j)))
            similarity = np.abs(x_hat[c])
            
            if similarity > 0 and pair not in tested_pairs:
                candidates.append({'pair': pair, 'similarity': similarity})
                tested_pairs.add(pair)
    
    return candidates

def test_candidate_pairs(
    candidates: List[Dict[str, Any]], 
    k: int, 
    V: List[str], 
    alpha: float
) -> Tuple[List[Dict[str, Any]], Set[tuple]]:
    """
    Performs vectorized statistical testing (Fisher's Z-test) on candidate pairs,
    corrects for multiple comparisons (Benjamini-Hochberg), and augments
    the pairs with string names for evaluation.
    
    Args:
        candidates: List of dicts, e.g., [{'pair': (0, 1), 'similarity': 0.8}]
        k: Number of dimensions (for statistical test).
        V: Vocabulary list (maps index to string).
        alpha: Significance level.
        
    Returns:
        - A ranked list of all candidates, augmented with p-values and string pairs.
        - A set of *only* the statistically significant string pairs.
    """
    if k <= 3:
        raise ValueError(f"Fisher's Z-test requires dimensions k > 3, but got k={k}.")
    
    if not candidates:
        print("No candidate pairs to test.")
        return [], set()

    # --- Vectorized Pass 1: Calculate all p-values at once --- 
    similarities = np.array([c['similarity'] for c in candidates])
    all_p_values = fisher_z_test(similarities, k)

    # --- Prepare the ranked list for the evaluation function ---
    augmented_candidates = []
    for i, cand in enumerate(candidates):
        idx_pair = cand['pair']
        string_pair = tuple(sorted((V[idx_pair[0]], V[idx_pair[1]])))
        
        augmented_candidates.append({
            'pair': string_pair,
            'similarity': cand['similarity'],
            'p_value': all_p_values[i]
        })
    
    ranked_pairs_for_eval = sorted(
        augmented_candidates, 
        key=lambda x: x['similarity'], 
        reverse=True
    )

    # --- Pass 2: Find the set of statistically significant pairs ---
    significant_mask = benjamini_hochberg(all_p_values, alpha)
    significant_pairs_set = {
        augmented_candidates[i]['pair']
        for i, is_significant in enumerate(significant_mask) if is_significant
    }
    
    print(f"Found {len(significant_pairs_set)} statistically significant pairs "
          f"out of {len(candidates)} candidates (FDR={alpha}).")
          
    return ranked_pairs_for_eval, significant_pairs_set