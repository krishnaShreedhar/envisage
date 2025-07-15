import numpy as np
import time
from memory_profiler import profile


# Standard Attention (for comparison)
@profile
def standard_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Standard scaled dot-product attention.
    Args:
        Q: Query matrix, shape (batch_size, query_seq_len, head_dim)
        K: Key matrix, shape (batch_size, kv_seq_len, head_dim)
        V: Value matrix, shape (batch_size, kv_seq_len, head_dim)
    Returns:
        np.ndarray: The output attention matrix, shape (batch_size, query_seq_len, head_dim)
    """
    scale_factor = 1.0 / np.sqrt(Q.shape[-1])
    # Compute attention scores S = QK^T
    # S shape: (batch_size, query_seq_len, kv_seq_len)
    scores = np.matmul(Q, np.swapaxes(K, -1, -2)) * scale_factor

    # Apply softmax to get attention weights
    # Subtract max for numerical stability (but entire N x N matrix is materialized)
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Compute output O = P V
    # O shape: (batch_size, query_seq_len, head_dim)
    output = np.matmul(attention_weights, V)
    return output


# Simplified Single-Head Flash Attention (from previous responses)
@profile
def single_head_flash_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                blk_q: int = 64, blk_kv: int = 64) -> np.ndarray:
    """
    A simplified implementation of single-head Flash Attention using NumPy,
    focusing on tiling and online softmax.

    Args:
        Q: Query matrix, shape (batch_size, query_seq_len, head_dim)
        K: Key matrix, shape (batch_size, kv_seq_len, head_dim)
        V: Value matrix, shape (batch_size, kv_seq_len, head_dim)
        blk_q: Block size for queries. Defaults to 64.
        blk_kv: Block size for keys/values. Defaults to 64.

    Returns:
        np.ndarray: The output attention matrix, shape (batch_size, query_seq_len, head_dim)
    """

    batch_size, query_seq_len, head_dim = Q.shape
    _, kv_seq_len, _ = K.shape

    O = np.zeros_like(Q)
    softmax_scale = 1.0 / np.sqrt(head_dim)

    for i in range(0, query_seq_len, blk_q):
        Q_i = Q[:, i: i + blk_q, :] * softmax_scale

        l_i = np.zeros((batch_size, Q_i.shape[1], 1))
        m_i = np.full((batch_size, Q_i.shape[1], 1), -np.inf)
        O_i = np.zeros((batch_size, Q_i.shape[1], head_dim))

        for j in range(0, kv_seq_len, blk_kv):
            K_j = K[:, j: j + blk_kv, :]
            V_j = V[:, j: j + blk_kv, :]

            S_ij = np.matmul(Q_i, np.swapaxes(K_j, -1, -2))

            m_ij_curr = np.max(S_ij, axis=-1, keepdims=True)
            P_ij = np.exp(S_ij - m_ij_curr)

            m_i_prev = m_i
            m_i = np.maximum(m_i_prev, m_ij_curr)

            alpha = np.exp(m_i_prev - m_i)
            beta = np.exp(m_ij_curr - m_i)

            l_i = alpha * l_i + beta * np.sum(P_ij, axis=-1, keepdims=True)
            O_i = alpha * O_i + beta * np.matmul(P_ij, V_j)

        O[:, i: i + blk_q, :] = O_i / l_i

    return O


# Function to run the benchmark
def run_benchmark(query_seq_len: int, kv_seq_len: int, head_dim: int, batch_size: int):
    """
    1. Benchmark both the standard and flash attention sample code for time and memory complexity

    :param query_seq_len:
    :param kv_seq_len:
    :param head_dim:
    :param batch_size:
    :return:
    """
    print(f"\n--- Benchmarking Sequence Length: {query_seq_len}, Head Dim: {head_dim} ---")
    Q_np = np.random.randn(batch_size, query_seq_len, head_dim)
    K_np = np.random.randn(batch_size, kv_seq_len, head_dim)
    V_np = np.random.randn(batch_size, kv_seq_len, head_dim)

    # Flash Attention
    print("\nRunning Flash Attention:")
    start_time_fa = time.time()
    output_fa_np = single_head_flash_attention(Q_np, K_np, V_np)
    end_time_fa = time.time()
    time_fa = end_time_fa - start_time_fa
    print(f"Flash Attention Time: {time_fa:.6f} seconds")

    # Standard Attention
    print("\nRunning Standard Attention:")
    start_time_std = time.time()
    output_std_np = standard_attention(Q_np, K_np, V_np)
    end_time_std = time.time()
    time_std = end_time_std - start_time_std
    print(f"Standard Attention Time: {time_std:.6f} seconds")

    # Verify numerical closeness
    # Using a higher tolerance for potential NumPy/floating point differences
    are_close = np.allclose(output_std_np, output_fa_np, atol=1e-5, rtol=1e-5)
    print(f"Outputs are numerically close: {are_close}")


def compare_results():
    batch_size = 4  # Keep batch size small for clarity
    head_dim = 64

    # Example 1: Smaller sequence length (Flash Attention benefits might be less pronounced)
    run_benchmark(query_seq_len=256, kv_seq_len=256, head_dim=head_dim, batch_size=batch_size)

    # Example 2: Larger sequence length (where Flash Attention shines in principle)
    # Be aware that very large sequence lengths will still be slow/memory-intensive with NumPy
    # due to Python/NumPy overhead, but the *relative* benefits should still be visible.
    # If this causes memory errors, reduce the sequence length.
    run_benchmark(query_seq_len=2048, kv_seq_len=2048, head_dim=head_dim, batch_size=batch_size)


# Main execution
if __name__ == "__main__":
    compare_results()

    # 1. Install memory_profiler: `pip install memory_profiler`
    # 3. Run from terminal: `python -m memory_profiler exp_flash_attention_2.py`
    # The `@profile` decorator will output line-by-line memory usage for the decorated functions.
