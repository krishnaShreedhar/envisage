import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# === Concept 1: Matrix Multiplication (Linear transformation) ===
def demo_matrix_multiplication():
    print("\n=== Matrix Multiplication ===")
    X = np.random.randn(2, 3)  # Example input (batch_size=2, features=3)
    W = np.random.randn(3, 4)  # Weights (features=3, output_dim=4)
    Y = X @ W                  # Linear transformation
    print("Input X:\n", X)
    print("Weights W:\n", W)
    print("Output Y = XW:\n", Y)

# === Concept 2: Dot product similarity (used in attention) ===
def demo_dot_product_similarity():
    print("\n=== Dot Product Similarity ===")
    A = np.random.randn(3)
    B = np.random.randn(3)
    similarity = A @ B
    print("Vector A:", A)
    print("Vector B:", B)
    print("Dot product (similarity):", similarity)

# === Concept 3: Broadcasting and element-wise operations ===
def demo_broadcasting():
    print("\n=== Broadcasting ===")
    X = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([10, 20, 30])
    Y = X + b
    print("Matrix X:\n", X)
    print("Vector b:", b)
    print("Broadcasted X + b:\n", Y)

# === Concept 4: Softmax visualization ===
def demo_softmax():
    print("\n=== Softmax ===")
    x = np.linspace(-2, 2, 5)
    e_x = np.exp(x - np.max(x))
    softmax_out = e_x / e_x.sum()
    print("Input:", x)
    print("Softmax output:", softmax_out)

    plt.figure()
    plt.bar(range(len(softmax_out)), softmax_out)
    plt.title("Softmax Output Distribution")
    plt.xlabel("Class index")
    plt.ylabel("Probability")
    plt.show()

# === Concept 5: Attention matrix heatmap ===
def demo_attention_heatmap():
    print("\n=== Attention Heatmap ===")
    Q = np.random.randn(3, 4)
    K = np.random.randn(3, 4)
    scores = Q @ K.T / np.sqrt(4)
    attn = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
    print("Attention scores:\n", scores)
    print("Attention weights (softmax):\n", attn)

    plt.figure()
    plt.imshow(attn, cmap='viridis')
    plt.colorbar()
    plt.title("Attention Weights Heatmap")
    plt.xlabel("Key index")
    plt.ylabel("Query index")
    plt.show()

# === Run demos ===
demo_matrix_multiplication()
demo_dot_product_similarity()
demo_broadcasting()
demo_softmax()
demo_attention_heatmap()
