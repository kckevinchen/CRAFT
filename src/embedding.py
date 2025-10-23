# src/embedding.py
import numpy as np
import bm25s
from tqdm import tqdm
from scipy.sparse import coo_matrix
from scipy.fft import fft  
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import IncrementalPCA, TruncatedSVD
from sklearn.kernel_approximation import RBFSampler
from typing import Literal
from .preprocessing import normalize, normalize_sparse

# Helper function to avoid repeating BM25 setup
def _build_bm25_model(corpus):
    """Initializes and indexes a BM25 model."""
    print("Indexing corpus with BM25s...")
    tokenized_corpus = bm25s.tokenize(corpus, stopwords="en")
    bm25_model = bm25s.BM25()
    bm25_model.index(tokenized_corpus)
    return bm25_model, tokenized_corpus

# Helper function to build the sparse matrix, used by multiple methods
def _build_sparse_bm25_matrix(corpus, vocabulary, bm25_model, batch_size):
    """Builds a sparse term-document matrix from BM25 scores incrementally."""
    num_terms = len(vocabulary)
    num_docs = len(corpus)
    
    data_list, row_indices, col_indices = [], [], []
    print("Building sparse BM25 matrix...")
    for i in tqdm(range(0, num_terms, batch_size), desc="Processing Batches"):
        start_index = i
        end_index = min(i + batch_size, num_terms)
        batch_vocab = vocabulary[start_index:end_index]

        queries_tokenized = [[term] for term in batch_vocab]
        batch_scores = np.array([bm25_model.get_scores(q) for q in queries_tokenized])

        non_zero_rows, non_zero_cols = batch_scores.nonzero()
        
        data_list.append(batch_scores[non_zero_rows, non_zero_cols])
        row_indices.append(non_zero_rows + start_index)
        col_indices.append(non_zero_cols)

    data = np.concatenate(data_list)
    rows = np.concatenate(row_indices)
    cols = np.concatenate(col_indices)

    bm25_sparse = coo_matrix((data, (rows, cols)), shape=(num_terms, num_docs))
    return bm25_sparse.tocsr()

def generate_embeddings(
    corpus: list, 
    vocabulary: list, 
    k: int, 
    batch_size: int = 1024,
    method: Literal['svd', 'rp', 'rbf'] = 'svd'
) -> np.ndarray:
    """
    Computes term embeddings from a corpus using various dimensionality reduction
    techniques on a sparse BM25 matrix.
    (Methods: 'svd', 'rp', 'rbf')
    """
    bm25_model, _ = _build_bm25_model(corpus)
    
    # All methods here start by building the full sparse matrix
    bm25_csr = _build_sparse_bm25_matrix(corpus, vocabulary, bm25_model, batch_size)
    bm25_csr = normalize_sparse(bm25_csr)
    
    print(f"Sparse matrix created. Shape: {bm25_csr.shape}, "
          f"Sparsity: {100 * bm25_csr.nnz / (bm25_csr.shape[0] * bm25_csr.shape[1]):.4f}%")

    print(f"Running embedding generation with method: '{method}'...")
    
    if method == 'svd':
        transformer = TruncatedSVD(n_components=k, algorithm='randomized')
    elif method == 'rp':
        transformer = GaussianRandomProjection(n_components=k)
    elif method == 'rbf':
        transformer = RBFSampler(n_components=k)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from ['svd', 'rp', 'rbf'].")

    term_embeddings = transformer.fit_transform(bm25_csr)
    
    print(f"{method.upper()} complete. Final shape: {term_embeddings.shape}")
    return term_embeddings

def generate_embeddings_ipca(corpus: list, vocabulary: list, k: int, batch_size: int = 1024) -> np.ndarray:
    """
    Computes term embeddings using IncrementalPCA (two-pass, dense batches).
    """
    bm25_model, tokenized_corpus = _build_bm25_model(corpus)
    num_terms = len(vocabulary)
    ipca = IncrementalPCA(n_components=k, batch_size=batch_size)

    # Pass 1: Train IncrementalPCA model
    print("Pass 1: Training IncrementalPCA model...")
    for i in tqdm(range(0, num_terms, batch_size), desc="Training Batches"):
        batch_vocab = vocabulary[i : i + batch_size]
        queries_tokenized = [[term] for term in batch_vocab]
        batch_scores = np.array([bm25_model.get_scores(q) for q in queries_tokenized])
        batch_scores = normalize(batch_scores)
        ipca.partial_fit(batch_scores)

    # Pass 2: Transform data to get embeddings
    print("\nPass 2: Transforming data to get embeddings...")
    term_embeddings_list = []
    for i in tqdm(range(0, num_terms, batch_size), desc="Transforming Batches"):
        batch_vocab = vocabulary[i : i + batch_size]
        queries_tokenized = [[term] for term in batch_vocab]
        batch_scores = np.array([bm25_model.get_scores(q, docs=tokenized_corpus) for q in queries_tokenized])
        batch_scores = normalize(batch_scores)
        batch_embeddings = ipca.transform(batch_scores)
        term_embeddings_list.append(batch_embeddings)

    term_embeddings = np.vstack(term_embeddings_list)
    print(f"IPCA complete. Final shape: {term_embeddings.shape}")
    return term_embeddings

def generate_embeddings_fft(corpus: list, vocabulary: list, k: int, batch_size: int = 1024) -> np.ndarray:
    """
    Computes the Randomized Fourier Embedding in batches using FFT.
    
    This function combines BM25 scoring, centering, and RFT, processing
    in dense batches. It produces complex-valued embeddings.
    """
    # Use the helper to set up BM25
    bm25_model, _ = _build_bm25_model(corpus)

    num_terms = len(vocabulary)
    num_docs = len(corpus)

    # Ensure k is not larger than the number of documents
    k = min(k, num_docs)

    # Final embedding matrix (complex)
    B = np.zeros((num_terms, k), dtype=np.complex128)

    # Pre-sample frequency indices so they are the same for all batches
    sampled_freq_indices = np.random.choice(num_docs, size=k, replace=False)

    print("Generating FFT embeddings in batches...")
    for i in tqdm(range(0, num_terms, batch_size), desc="Embedding Batches"):
        start_index = i
        end_index = min(i + batch_size, num_terms)
        batch_vocab = vocabulary[start_index:end_index]

        # 1. Compute BM25 scores
        queries_tokenized = [[term] for term in batch_vocab]
        batch_scores_list = []
        for query in queries_tokenized:
            scores = bm25_model.get_scores(query)
            batch_scores_list.append(scores)
        A_batch = np.array(batch_scores_list)

        # 2. Center the rows of the batch matrix
        A_tilde_batch = normalize(A_batch)

        # 3. Apply FFT along the document axis for the batch
        A_tilde_dft_batch = fft(A_tilde_batch, axis=1)

        # 4. Select the pre-sampled frequencies
        B_batch = A_tilde_dft_batch[:, sampled_freq_indices]

        # 5. Store the result in the final embedding matrix
        B[start_index:end_index, :] = B_batch

    print(f"FFT embeddings complete. Final shape: {B.shape}")
    return B