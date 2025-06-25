import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.asarray(data, dtype=float)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set()
        self._op = ''

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}
        out._op = '+'

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad if self.grad is not None else out.grad
            if other.requires_grad:
                other.grad = other.grad + out.grad if other.grad is not None else out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}
        out._op = '*'

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + other.data * out.grad if self.grad is not None else other.data * out.grad
            if other.requires_grad:
                other.grad = other.grad + self.data * out.grad if other.grad is not None else self.data * out.grad

        out._backward = _backward
        return out

    def matmul(self, other):
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}
        out._op = '@'

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad @ other.data.T if self.grad is not None else out.grad @ other.data.T
            if other.requires_grad:
                other.grad = other.grad + self.data.T @ out.grad if other.grad is not None else self.data.T @ out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad)
        out._prev = {self}
        out._op = 'ReLU'

        def _backward():
            if self.requires_grad:
                relu_grad = (self.data > 0) * out.grad
                self.grad = self.grad + relu_grad if self.grad is not None else relu_grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_topo(child)
                topo.append(t)

        build_topo(self)

        self.grad = np.ones_like(self.data)

        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"


# Create tensors
x = Tensor([[1, 2], [3, 4]], requires_grad=True)
w = Tensor([[0.5], [1.0]], requires_grad=True)
b = Tensor([[0.1]], requires_grad=True)

# Forward pass
y = x.matmul(w) + b
out = y.relu()

print("Forward output:", out)

# Backward pass
out.backward()

print("\\nGradient wrt x:", x.grad)
print("Gradient wrt w:", w.grad)
print("Gradient wrt b:", b.grad)


def show_graph(tensor):
    import networkx as nx
    import matplotlib.pyplot as plt
    G = nx.DiGraph()
    nodes = set()

    def build(t):
        if t not in nodes:
            nodes.add(t)
            for c in t._prev:
                G.add_edge(str(c._op) + str(id(c)), str(t._op) + str(id(t)))
                build(c)

    build(tensor)
    nx.draw(G, with_labels=True)
    plt.show()
