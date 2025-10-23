from typing import List, Set, Dict, Any, Iterable

def calculate_average_precision(ranked_pairs: List[Dict[str, Any]], ground_truth: Set[tuple]) -> float:
    """
    Calculates the Average Precision (AP) for a ranked list of pairs.
    
    The list is assumed to be pre-sliced to the desired cutoff (e.g., @R).
    Normalization is done by the total number of relevant items (len(ground_truth)).
    """
    if not ground_truth:
        return 0.0

    num_hits = 0
    precision_sum = 0
    for i, item in enumerate(ranked_pairs):
        rank = i + 1
        # Assumes ground_truth is a set for O(1) lookup
        if item['pair'] in ground_truth:
            num_hits += 1
            precision_at_k = num_hits / rank
            precision_sum += precision_at_k

    # Normalize by the total number of relevant items
    return precision_sum / len(ground_truth)


def evaluate_ranking(
    ranked_pairs: List[Dict[str, Any]], 
    ground_truth: Iterable[tuple], 
    evaluation_ks: List[int]
) -> Dict[str, Any]:
    """
    Evaluates the ranking based on ground truth and returns a results dictionary.
    
    Calculates Precision@k, Recall@k for each k in evaluation_ks,
    and Average Precision@R (where R = len(ground_truth)).
    """
    # 1. Ensure ground_truth is a set for efficient O(1) lookups
    ground_truth_set = set(ground_truth)
    num_ground_truth = len(ground_truth_set)

    results = {
        "metrics_at_k": {},
        "average_precision": 0.0
    }

    if num_ground_truth == 0:
        # No relevant items, all metrics are 0
        return results

    # 2. Calculate P@k and R@k
    for k_val in evaluation_ks:
        if k_val == 0:
            continue
            
        if k_val > len(ranked_pairs):
            # Warn or skip if k is larger than the ranked list
            # Using max available pairs instead
            k_val = len(ranked_pairs)

        top_k_pairs = {item['pair'] for item in ranked_pairs[:k_val]}
        hits_at_k = len(ground_truth_set.intersection(top_k_pairs))

        recall_at_k = hits_at_k / num_ground_truth
        precision_at_k = hits_at_k / k_val

        results["metrics_at_k"][k_val] = {
            "precision": precision_at_k,
            "recall": recall_at_k,
            "hits": hits_at_k
        }

    # 3. Calculate Average Precision @ R (where R = num_ground_truth)
    # We pass the list sliced at R, and the full ground truth set
    avg_precision = calculate_average_precision(
        ranked_pairs[:num_ground_truth], 
        ground_truth_set
    )
    results["average_precision"] = avg_precision

    # 4. Return the results dictionary (no printing)
    return results


def calculate_f1(set1: Iterable, set2: Iterable) -> float:
    """
    Calculates the F1 score between two sets of items (e.g., predicted vs. actual).
    """
    predicted_set = set(set1)
    actual_set = set(set2)

    true_positives = len(predicted_set.intersection(actual_set))

    # This check correctly handles all edge cases where
    # precision or recall would be 0, preventing division by zero.
    if true_positives == 0:
        return 0.0

    precision = true_positives / len(predicted_set)
    recall = true_positives / len(actual_set)

    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score