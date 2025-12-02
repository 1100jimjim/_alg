import numpy as np

p = np.array([0.5, 0.2, 0.3])

def cross_entropy_bits(p, q):
    return -np.sum(p * np.log2(q + 1e-12))

print("Random Search for CE(p,q) minimum")
print("Target p:", p)
print()

best_q = np.random.dirichlet(alpha=np.ones(3))
best_loss = cross_entropy_bits(p, best_q)

print(f"Initial q: {best_q.round(4)}, Loss: {best_loss:.5f}")

max_iters = 20000

for t in range(max_iters + 1):
    q_candidate = np.random.dirichlet(alpha=np.ones(3))
    loss_candidate = cross_entropy_bits(p, q_candidate)

    if loss_candidate < best_loss:
        best_q = q_candidate
        best_loss = loss_candidate

    if t % 1000 == 0:
        print(f"{t:05d}: Loss={best_loss:.5f} q={best_q.round(4)}")

print("-" * 60)
print("Final Result:")
print("Optimized q :", best_q.round(4))
print("Target    p :", p)
print("Final Loss  :", best_loss)
print("Diff (q - p):", best_q - p)
