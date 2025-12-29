```.py

import math
import random
from typing import Callable, Set, Tuple, List


class Value:
    """
    A scalar value node for autodiff (reverse-mode automatic differentiation).

    Attributes:
        data: float value of this node
        grad: accumulated gradient d(output)/d(this)
        _prev: parent nodes in the computational graph
        _op: operation label (for graph display)
        _backward: function that backpropagates local gradients to parents
    """
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None  # set by ops

    def __repr__(self):
        return f"Value(data={self.data:.6f}, grad={self.grad:.6f}, op={self._op})"

    # -----------------------
    # Basic arithmetic ops
    # -----------------------
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        out = Value(-self.data, (self,), "neg")

        def _backward():
            self.grad += -1.0 * out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return Value(other) + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, power):
        assert isinstance(power, (int, float)), "power must be int/float"
        out = Value(self.data ** power, (self,), f"**{power}")

        def _backward():
            self.grad += (power * (self.data ** (power - 1.0))) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return Value(other) * (self ** -1)

    # -----------------------
    # Non-linearities (bonus)
    # -----------------------
    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "ReLU")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            # d/dx tanh(x) = 1 - tanh(x)^2
            self.grad += (1.0 - t * t) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        # Numerically stable-ish sigmoid for moderate ranges
        x = self.data
        if x >= 0:
            z = math.exp(-x)
            s = 1.0 / (1.0 + z)
        else:
            z = math.exp(x)
            s = z / (1.0 + z)
        out = Value(s, (self,), "sigmoid")

        def _backward():
            # d/dx sigmoid(x) = s(1-s)
            self.grad += (s * (1.0 - s)) * out.grad
        out._backward = _backward
        return out

    # -----------------------
    # Backprop (reverse-mode)
    # -----------------------
    def backward(self):
        # Topological order by DFS
        topo: List[Value] = []
        visited: Set[Value] = set()

        def build(v: "Value"):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        # Seed gradient
        self.grad = 1.0

        # Traverse in reverse topo order
        for v in reversed(topo):
            v._backward()


# =========================
# Bonus 2: Graph display
# =========================
def trace(root: Value) -> Tuple[Set[Value], Set[Tuple[Value, Value]]]:
    nodes: Set[Value] = set()
    edges: Set[Tuple[Value, Value]] = set()

    def build(v: Value):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_graph_text(root: Value, max_nodes: int = 60):
    nodes, edges = trace(root)
    nodes_list = list(nodes)
    # Avoid printing gigantic graphs
    if len(nodes_list) > max_nodes:
        nodes_list = nodes_list[:max_nodes]

    print("\n=== Computation Graph (text) ===")
    for n in nodes_list:
        print(f"op={n._op:>6} | data={n.data:>10.6f} | grad={n.grad:>10.6f} | prev={len(n._prev)}")
    print("=== Edges (child -> parent) ===")
    cnt = 0
    for a, b in edges:
        print(f"{a._op or 'leaf':>6} -> {b._op:>6}")
        cnt += 1
        if cnt >= max_nodes:
            print("...(edges truncated)")
            break
    print("===============================\n")


# =========================
# Bonus 3: Gradient check
# =========================
def numerical_grad(f: Callable[[], Value], x: Value, eps: float = 1e-6) -> float:
    """
    Approximate df/dx using central difference:
    (f(x+eps) - f(x-eps)) / (2eps)
    """
    orig = x.data

    x.data = orig + eps
    f1 = f().data

    x.data = orig - eps
    f2 = f().data

    x.data = orig
    return (f1 - f2) / (2.0 * eps)


# =========================
# Demo 1: Linear Regression
# =========================
def demo_linear_regression():
    print("=== Demo 1: Linear Regression (y = a*x + b) ===")
    random.seed(42)

    true_a, true_b = 2.5, -1.0
    xs = [i / 10 for i in range(-50, 51)]
    ys = [true_a * x + true_b + random.uniform(-0.2, 0.2) for x in xs]

    a = Value(random.uniform(-1, 1))
    b = Value(random.uniform(-1, 1))

    lr = 0.05
    for epoch in range(1, 201):
        # Forward: MSE
        loss = Value(0.0)
        for x, y in zip(xs, ys):
            xV = Value(x)
            yhat = a * xV + b
            loss = loss + (yhat - y) ** 2
        loss = loss * (1.0 / len(xs))

        # Backward
        a.grad = 0.0
        b.grad = 0.0
        loss.backward()

        # Update
        a.data += -lr * a.grad
        b.data += -lr * b.grad

        if epoch % 20 == 0 or epoch == 1:
            print(f"epoch={epoch:3d} loss={loss.data:.6f}  a={a.data:.4f} b={b.data:.4f}")

    print("\nTrue params:", true_a, true_b)
    print("Learned params:", a.data, b.data)
    return a, b


# =========================
# Demo 2: Nonlinear + Graph + GradCheck
# =========================
def demo_nonlinear_graph_and_gradcheck():
    print("\n=== Demo 2: Nonlinear function + Graph + Gradient Check ===")
    # We test a nonlinear function to validate tanh/sigmoid backward rules.
    x = Value(1.2345)

    # Define function: f(x) = tanh(x*x + 3x - 1) + sigmoid(2x)
    def f():
        return (x * x + 3 * x - 1).tanh() + (2 * x).sigmoid()

    y = f()
    # Backprop
    x.grad = 0.0
    y.backward()

    # Numerical gradient
    ng = numerical_grad(f, x, eps=1e-6)

    print(f"x.data = {x.data:.6f}")
    print(f"f(x)   = {y.data:.6f}")
    print(f"autodiff grad df/dx = {x.grad:.10f}")
    print(f"numerical grad      = {ng:.10f}")
    print(f"abs diff            = {abs(x.grad - ng):.10e}")

    # Graph display (text)
    draw_graph_text(y, max_nodes=60)


def main():
    demo_linear_regression()
    demo_nonlinear_graph_and_gradcheck()
    print("Done ✅")


if __name__ == "__main__":
    main()

```
