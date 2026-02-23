import numpy as np
from typing import Tuple, List, Dict


def linear_partition_greedy_refined(arr: np.ndarray, N: int) -> Tuple[int, np.ndarray]:
    """
    Refined Greedy Linear Partition with Label Output.

    Partitions a 1-D numpy array into N contiguous parts using a greedy
    approach with local search improvement. Returns partition labels.

    Parameters
    ----------
    arr : np.ndarray
        1-D numpy array of int32, non-negative integers
    N : int
        Number of partitions required

    Returns
    -------
    max_sum : int
        The maximum sum across all partitions
    labels : np.ndarray
        Array of same size as input, where labels[i] = partition ID (0 to N-1)
    """
    # Input validation
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr, dtype=np.int32)
    if arr.dtype != np.int32:
        arr = arr.astype(np.int32)
    if arr.ndim != 1:
        raise ValueError("Input must be a 1-D array")
    if N <= 0 or N > len(arr):
        raise ValueError(f"N must be between 1 and {len(arr)}")
    if np.any(arr < 0):
        raise ValueError("All elements must be non-negative")

    n = len(arr)

    # Edge cases
    if N == 1:
        return int(arr.sum()), np.zeros(n, dtype=np.int32)
    if N == n:
        return int(arr.max()), np.arange(n, dtype=np.int32)

    # Calculate target sum per partition
    total_sum = int(arr.sum())
    target_sum = total_sum / N

    # Greedy initialization: find cut positions
    cut_indices = []
    current_sum = 0
    parts_formed = 0

    for i in range(n - 1):
        current_sum += arr[i]

        if current_sum >= target_sum and parts_formed < N - 1:
            remaining_elements = n - 1 - i
            remaining_parts = N - 1 - parts_formed

            if remaining_elements >= remaining_parts:
                cut_indices.append(i + 1)
                parts_formed += 1
                current_sum = 0

    # Ensure we have exactly N-1 cuts
    while len(cut_indices) < N - 1:
        # Add cuts at the end if needed
        cut_indices.append(n - (N - 1 - len(cut_indices)))
    cut_indices = sorted(cut_indices[: N - 1])

    # Local search refinement
    improved = True
    max_iterations = 50
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        for i in range(len(cut_indices)):
            best_move = 0
            best_max_sum = _evaluate_cuts(arr, cut_indices)

            # Determine valid range for this cut
            min_pos = cut_indices[i - 1] + 1 if i > 0 else 1
            max_pos = cut_indices[i + 1] - 1 if i < len(cut_indices) - 1 else n - 1

            # Try moving left
            if cut_indices[i] > min_pos:
                test_cuts = cut_indices.copy()
                test_cuts[i] -= 1
                test_max = _evaluate_cuts(arr, test_cuts)
                if test_max < best_max_sum:
                    best_max_sum = test_max
                    best_move = -1

            # Try moving right
            if cut_indices[i] < max_pos:
                test_cuts = cut_indices.copy()
                test_cuts[i] += 1
                test_max = _evaluate_cuts(arr, test_cuts)
                if test_max < best_max_sum:
                    best_max_sum = test_max
                    best_move = 1

            # Apply best move
            if best_move != 0:
                cut_indices[i] += best_move
                improved = True

    # Convert cut indices to label array
    labels = np.zeros(n, dtype=np.int32)
    part_id = 0
    start = 0
    for cut in cut_indices:
        labels[start:cut] = part_id
        start = cut
        part_id += 1
    labels[start:] = part_id

    # Calculate max sum
    part_sums = np.bincount(labels, weights=arr.astype(np.float64))
    max_sum = int(part_sums.max())

    return max_sum, labels


def _evaluate_cuts(arr: np.ndarray, cut_indices: list) -> int:
    """Helper: Evaluate max sum for given cut positions"""
    if len(cut_indices) == 0:
        return int(arr.sum())

    sums = []
    start = 0
    for cut in cut_indices:
        sums.append(int(arr[start:cut].sum()))
        start = cut
    sums.append(int(arr[start:].sum()))
    return max(sums)


# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    arr = np.random.randint(1, 100, size=1000, dtype=np.int32)
    N = 10

    max_sum, labels = linear_partition_greedy_refined(arr, N)

    print(f"Input: {arr}")
    print(f"Labels: {labels}")
    print(f"Max Sum: {max_sum}")
    print(f"Partition sums: {[int(arr[labels == i].sum()) for i in range(N)]}")
