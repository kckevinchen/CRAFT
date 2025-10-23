import numpy as np
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from typing import Union, Iterable, List

def fisher_z_test(
    rho_hat: Union[float, Iterable[float], np.ndarray], 
    k: int
) -> Union[float, np.ndarray]:
    """
    Perform Fisher's Z-test to get a p-value.
    Works for both single float and numpy arrays.
    
    Requires k > 3.
    """
    # --- CRITICAL: Handle k <= 3 edge case ---
    if k <= 3:
        raise ValueError(f"Fisher's Z-test requires k > 3 (i.e., more than 3 samples/dimensions), but got k={k}.")
        
    # Ensure rho_hat is a numpy array for vectorized operations
    rho_hat = np.asarray(rho_hat)
    # Clip to avoid inf values from arctanh(1.0)
    rho_hat = np.clip(rho_hat, -1.0 + 1e-9, 1.0 - 1e-9)
    
    # Fisher's Z-transform (element-wise)
    z = np.arctanh(rho_hat)
    
    # Standard error for the Z-transform
    se = 1.0 / np.sqrt(k - 3)
    
    # Z-score (element-wise)
    z_score = z / se
    
    # Two-tailed p-value (element-wise)
    p_value = 2 * norm.sf(np.abs(z_score)) # norm.sf(x) is 1 - norm.cdf(x)
    
    return p_value

def benjamini_hochberg(
    p_values: Iterable[float], 
    alpha: float
) -> np.ndarray:
    """
    Apply the Benjamini-Hochberg (FDR) procedure.
    
    Returns:
        np.ndarray: A boolean mask where True indicates a significant p-value.
    """
    p_values_arr = np.asarray(p_values)
    
    # Handle empty input robustly
    if p_values_arr.size == 0:
        return np.array([], dtype=bool)
        
    # Returns a boolean mask for significant p-values
    reject, _, _, _ = multipletests(p_values_arr, alpha=alpha, method='fdr_bh')
    
    return reject