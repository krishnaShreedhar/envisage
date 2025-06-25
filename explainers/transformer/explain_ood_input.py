import numpy as np

np.random.seed(42)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def one_hot(indices, depth):
    out = np.zeros((indices.size, depth))
    out[np.arange(indices.size), indices] = 1
    return out

class SimpleTransformerExplainer:
    def __init__(self, d_model, vocab_size):
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Initialize weight matrices
        self.W_q = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_k = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_v = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_fc = np.random.randn(d_model, vocab_size) / np.sqrt(d_model)

    def attention(self, Q, K, V):
        print("\n=== Attention Computation ===")
        print("Q:\n", Q)
        print("K:\n", K)
        print("V:\n", V)

        # Compute raw scores (similarity between queries and keys)
        scores = Q @ K.T / np.sqrt(self.d_model)
        print("Raw attention scores (QK^T / sqrt(d)):\n", scores)

        # Apply softmax to get attention weights (probability distribution)
        attn_weights = softmax(scores)
        print("Attention weights (after softmax):\n", attn_weights)

        # Compute the weighted sum of V
        out = attn_weights @ V
        print("Attention output (weighted sum of V):\n", out)

        return out, attn_weights

    def forward(self, x):
        # Linear projections
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        attn_out, attn_weights = self.attention(Q, K, V)
        logits = attn_out @ self.W_fc
        probs = softmax(logits)

        return probs, attn_weights

# === Example setup ===
vocab_size = 5
embedding_dim = 4
batch_size = 3

# Create model
model = SimpleTransformerExplainer(d_model=embedding_dim, vocab_size=vocab_size)

# Generate in-distribution input: token IDs from vocab
in_tokens = np.array([1, 2, 3])
X_in = one_hot(in_tokens, vocab_size) @ np.random.randn(vocab_size, embedding_dim)

print("\n>>> IN-DISTRIBUTION INPUT")
probs_in, attn_in = model.forward(X_in)

# Generate OOD input: random noise instead of proper embeddings
X_ood = np.random.randn(batch_size, embedding_dim) * 10  # exaggerated noise

print("\n>>> OUT-OF-DISTRIBUTION INPUT")
probs_ood, attn_ood = model.forward(X_ood)

# Compare attention distributions
print("\n>>> In-distribution attention weights:")
print(attn_in)

print("\n>>> Out-of-distribution attention weights:")
print(attn_ood)

# Comment: The in-distribution input produces structured attention (tokens attend to related tokens)
# The OOD input produces attention weights that may look random or assign high weights arbitrarily
# because the QK^T similarities don't reflect learned relationships
