import numpy as np

def solve_OMP(Psi_L, z, epsilon, block_size=5, max_iters=2):
    """
    High-performance sparse recovery solver using the Block Sparse Recovery (BSR)
    algorithm with Gram-Schmidt orthogonalization.

    This algorithm is a generalization of OMP that selects the best block of
    'block_size' columns at each iteration, which can be more efficient and
    robust, especially for signals with clustered sparsity.

    Args:
        Psi_L (np.ndarray): The measurement matrix.
        z (np.ndarray): The measurement vector.
        epsilon (float): The target error tolerance. The algorithm stops when
                         the L2 norm of the residual is less than this value.
        block_size (int, optional): The number of best-matching columns to
                                    select at each iteration. Defaults to 5.
                                    Setting block_size=1 makes this algorithm
                                    equivalent to standard Fast OMP.
        max_iters (int, optional): The maximum number of non-zero coefficients
                                   to find. This acts as a safeguard.

    Returns:
        np.ndarray: The estimated sparse coefficient vector x.
    """
    # 1. Initialization
    L_dim = Psi_L.shape[1]
    residual = np.copy(z)
    support_indices = []
    x = np.zeros(L_dim, dtype=z.dtype)

    # Pre-allocate arrays for the QR decomposition parts
    Q = np.zeros((Psi_L.shape[0], max_iters), dtype=z.dtype)
    R = np.zeros((max_iters, max_iters), dtype=z.dtype)
    
    # Determine the number of main loops based on block size
    num_loops = int(np.ceil(max_iters / block_size))
    
    for _ in range(num_loops):
        current_support_size = len(support_indices)
        if current_support_size >= max_iters:
            break

        # --- Find the best BLOCK of columns ---
        correlations = np.abs(Psi_L.conj().T @ residual)
        correlations[support_indices] = -1.0 # Exclude already chosen columns
        
        # Determine how many columns to add in this block
        cols_to_add = min(block_size, max_iters - current_support_size)
        best_indices_block = np.argpartition(correlations, -cols_to_add)[-cols_to_add:]
        
        # --- Orthogonalize each new column in the block via Gram-Schmidt ---
        for idx in best_indices_block:
            k = len(support_indices)
            if k >= max_iters: break
            
            new_col = Psi_L[:, idx]
            # Project new column onto the existing orthogonal basis Q
            h = Q[:, :k].conj().T @ new_col
            # Subtract projections to get the orthogonal component
            v_perp = new_col - Q[:, :k] @ h
            
            norm_v_perp = np.linalg.norm(v_perp)
            # If column is linearly dependent, skip it
            if norm_v_perp < 1e-12: continue
            
            # Update the QR decomposition matrices
            R[k, k] = norm_v_perp
            Q[:, k] = v_perp / R[k, k] # New orthogonal basis vector
            R[:k, k] = h
            support_indices.append(idx)
        
        # --- Solve for coefficients and update residual ---
        current_support_size = len(support_indices)
        if current_support_size == 0: continue

        # Solve R * x = Q^H * z using the updated factorization.
        # This is very fast because R is upper-triangular.
        q_t_z = Q[:, :current_support_size].conj().T @ z
        x_support = np.linalg.solve(R[:current_support_size, :current_support_size], q_t_z)
        residual = z - Psi_L[:, support_indices] @ x_support

        # --- Check stopping condition ---
        if np.linalg.norm(residual) < epsilon:
            break

    # 3. Construct the final sparse solution from the calculated coefficients
    if support_indices:
        x[support_indices] = x_support
        
    return x 


