import numpy as np

p = np.array([0.5, 0.2, 0.3], dtype=float)

def entropy_bits(p):
    return -np.sum(p * np.log2(p + 1e-12))

target_entropy = entropy_bits(p)
print(f"Target p: {p}")
print(f"Target Min Loss (Entropy): {target_entropy:.5f}\n")

def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)

def cross_entropy_bits(p, q):
    """Cross Entropy 使用 log2（和 entropy 單位一致）"""
    return -np.sum(p * np.log2(q + 1e-12))

np.random.seed(0)

z = np.zeros(3)
q = softmax(z)
loss = cross_entropy_bits(p, q)

print("Start Gradient Descent...")
print(f"Initial q: {q.round(4)}, Loss: {loss:.5f}")

max_iters = 20000
base_step = 0.4    
decay = 0.8        

for t in range(max_iters + 1):
    q = softmax(z)
    loss = cross_entropy_bits(p, q)

    if t % 1000 == 0:
        step = base_step * (decay ** (t // 1000))
        print(f"{t:05d}: Loss={loss:.5f} q={q.round(4)} step={step:.5f}")

    grad = q - p

    step = base_step * (decay ** (t // 1000))

    z -= step * grad

print("-" * 60)
final_q = softmax(z)
final_loss = cross_entropy_bits(p, final_q)

print("Final Result:")
print("Optimized q :", final_q.round(4))
print("Target    p :", p)
print(f"Final Loss  : {final_loss:.5f}")
print("Diff (q - p):", final_q - p)
