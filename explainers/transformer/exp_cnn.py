import numpy as np


# --- Conv2D ---
class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.k = kernel_size
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1
        self.b = np.zeros(out_channels)

    def forward(self, x):
        self.x = x
        n, c, h, w = x.shape
        out_h = h - self.k + 1
        out_w = w - self.k + 1
        out = np.zeros((n, self.W.shape[0], out_h, out_w))

        for i in range(out_h):
            for j in range(out_w):
                patch = x[:, :, i:i + self.k, j:j + self.k]
                out[:, :, i, j] = np.tensordot(patch, self.W, axes=([1, 2, 3], [1, 2, 3])) + self.b
        return out

    def backward(self, dout, lr):
        n, c, h, w = self.x.shape
        _, oc, oh, ow = dout.shape
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dx = np.zeros_like(self.x)

        for i in range(oh):
            for j in range(ow):
                patch = self.x[:, :, i:i + self.k, j:j + self.k]
                for n_idx in range(n):
                    dW += dout[n_idx, :, i, j][:, None, None, None] * patch[n_idx]
                db += np.sum(dout[:, :, i, j], axis=0)

                for n_idx in range(n):
                    dx[n_idx, :, i:i + self.k, j:j + self.k] += np.sum(
                        self.W * dout[n_idx, :, i, j][:, None, None, None],
                        axis=0
                    )

        self.W -= lr * dW
        self.b -= lr * db
        return dx


# --- ReLU ---
class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask


# --- Flatten ---
class Flatten:
    def forward(self, x):
        self.orig_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.orig_shape)


# --- Linear ---
class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * 0.1
        self.b = np.zeros(out_features)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout, lr):
        dW = self.x.T @ dout
        db = np.sum(dout, axis=0)
        dx = dout @ self.W.T

        self.W -= lr * dW
        self.b -= lr * db
        return dx


# --- Loss ---
def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / exps.sum(axis=1, keepdims=True)


def cross_entropy(pred, target):
    n = pred.shape[0]
    loss = -np.log(pred[np.arange(n), target] + 1e-9).mean()
    return loss


def softmax_backward(pred, target):
    grad = pred.copy()
    grad[np.arange(pred.shape[0]), target] -= 1
    return grad / pred.shape[0]


# --- CNN model ---
class MiniCNN:
    def __init__(self):
        self.conv = Conv2D(1, 2, 3)  # 1 in channel, 2 out, 3x3 kernel
        self.relu = ReLU()
        self.flat = Flatten()
        self.fc = Linear(2 * 6 * 6, 2)  # output after conv: (8-3+1)=6

    def forward(self, x):
        x = self.conv.forward(x)
        x = self.relu.forward(x)
        x = self.flat.forward(x)
        x = self.fc.forward(x)
        return x

    def backward(self, dout, lr):
        dout = self.fc.backward(dout, lr)
        dout = self.flat.backward(dout)
        dout = self.relu.backward(dout)
        dout = self.conv.backward(dout, lr)


# --- Train dummy CNN ---
np.random.seed(0)
model = MiniCNN()

# Dummy data: 10 samples of 8x8 grayscale images
X = np.random.randn(10, 1, 8, 8)
y = np.random.randint(0, 2, 10)

for epoch in range(10):
    logits = model.forward(X)
    probs = softmax(logits)
    loss = cross_entropy(probs, y)
    print(f"Epoch {epoch} Loss: {loss:.4f}")

    dout = softmax_backward(probs, y)
    model.backward(dout, lr=0.01)

    # --- Inference ---
    test_X = np.random.randn(2, 1, 8, 8)
    test_logits = model.forward(test_X)
    test_probs = softmax(test_logits)
    print("\\nTest probs:\\n", test_probs)
