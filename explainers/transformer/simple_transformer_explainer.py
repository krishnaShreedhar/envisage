import numpy as np

np.random.seed(42)  # for reproducibility


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def cross_entropy_loss(pred, target):
    # pred: (batch, vocab_size), target: (batch,)
    log_likelihood = -np.log(pred[np.arange(len(target)), target] + 1e-9)
    return np.mean(log_likelihood)


def one_hot(indices, depth):
    out = np.zeros((indices.size, depth))
    out[np.arange(indices.size), indices] = 1
    return out


class SimpleTransformerExplainer:
    def __init__(self, d_model, vocab_size):
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Initialize weights
        self.W_q = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_k = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_v = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_o = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_fc = np.random.randn(d_model, vocab_size) / np.sqrt(d_model)

        # Gradients
        self.grads = {}

    def attention(self, Q, K, V):
        print("Q shape:", Q.shape)
        print("K shape:", K.shape)
        print("V shape:", V.shape)

        scores = Q @ K.T / np.sqrt(self.d_model)
        print("Attention scores:", scores)

        attn_weights = softmax(scores)
        print("Attention weights:", attn_weights)

        out = attn_weights @ V
        print("Attention output:", out)

        return out, attn_weights

    def forward(self, x):
        # x: (batch_size, d_model)
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        attn_out, self.attn_weights = self.attention(Q, K, V)

        o = attn_out @ self.W_o
        logits = o @ self.W_fc

        self.cache = (x, Q, K, V, attn_out, o, logits)

        probs = softmax(logits)
        return probs

    def backward(self, x, probs, targets):
        batch_size = x.shape[0]
        d_logits = probs.copy()
        d_logits[np.arange(batch_size), targets] -= 1
        d_logits /= batch_size

        x, Q, K, V, attn_out, o, logits = self.cache

        self.grads['W_fc'] = o.T @ d_logits
        d_o = d_logits @ self.W_fc.T

        self.grads['W_o'] = attn_out.T @ d_o
        d_attn_out = d_o @ self.W_o.T

        d_attn_weights = d_attn_out @ V.T
        d_V = self.attn_weights.T @ d_attn_out

        d_scores = d_attn_weights * self.attn_weights * (1 - self.attn_weights)

        d_Q = d_scores @ K / np.sqrt(self.d_model)
        d_K = d_scores.T @ Q / np.sqrt(self.d_model)

        self.grads['W_q'] = x.T @ d_Q
        self.grads['W_k'] = x.T @ d_K
        self.grads['W_v'] = x.T @ d_V

    def step(self, lr):
        for name, grad in self.grads.items():
            setattr(self, name, getattr(self, name) - lr * grad)


# Dummy dataset
vocab_size = 10
d_model = 4
batch_size = 2

model = SimpleTransformerExplainer(d_model=d_model, vocab_size=vocab_size)

# random one-hot input as embeddings
X = one_hot(np.random.randint(0, vocab_size, size=batch_size), vocab_size) @ np.random.randn(vocab_size, d_model)
targets = np.random.randint(0, vocab_size, size=batch_size)

# Training loop
for epoch in range(10):
    probs = model.forward(X)
    loss = cross_entropy_loss(probs, targets)
    print(f"Epoch {epoch}, Loss: {loss:.4f}")
    model.backward(X, probs, targets)
    model.step(lr=0.1)
