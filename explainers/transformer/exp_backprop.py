import numpy as np

np.random.seed(42)

# --- Helper functions ---
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def cross_entropy(probs, targets):
    n = probs.shape[0]
    log_likelihood = -np.log(probs[np.arange(n), targets] + 1e-9)
    return np.mean(log_likelihood)

# --- Our tiny neural network ---
class TinyNet:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
        self.b2 = np.zeros((1, output_dim))

    def forward(self, X):
        # Linear 1
        self.X = X
        self.Z1 = X @ self.W1 + self.b1
        print("Z1 (pre-ReLU):\\n", self.Z1)

        # ReLU
        self.A1 = np.maximum(0, self.Z1)
        print("A1 (ReLU output):\\n", self.A1)

        # Linear 2
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.probs = softmax(self.Z2)
        print("Z2 (logits):\\n", self.Z2)
        print("Probs (softmax):\\n", self.probs)

        return self.probs

    def backward(self, y):
        n = y.shape[0]

        # Gradient of loss wrt logits
        dZ2 = self.probs
        dZ2[np.arange(n), y] -= 1
        dZ2 /= n
        print("dZ2 (grad loss w.r.t logits):\\n", dZ2)

        # Gradients for W2, b2
        dW2 = self.A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # Gradient back to hidden layer
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (self.Z1 > 0)  # ReLU grad
        print("dZ1 (grad loss w.r.t pre-ReLU):\\n", dZ1)

        # Gradients for W1, b1
        dW1 = self.X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Store gradients
        self.grads = { 'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2 }

    def step(self, lr=0.1):
        for param in self.grads:
            setattr(self, param, getattr(self, param) - lr * self.grads[param])

# --- Example usage ---
X = np.random.randn(5, 3)  # 5 samples, 3 features
y = np.array([0, 1, 2, 1, 0])

model = TinyNet(input_dim=3, hidden_dim=4, output_dim=3)

for epoch in range(5):
    print(f"\\n=== Epoch {epoch} ===")
    probs = model.forward(X)
    loss = cross_entropy(probs, y)
    print(f"Loss: {loss:.4f}")
    model.backward(y)
    model.step(lr=0.1)
