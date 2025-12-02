import numpy as np

p = np.array([0.5, 0.2, 0.3], dtype=float)

def entropy_bits(p):
    return -np.sum(p * np.log2(p + 1e-12))

target_entropy = entropy_bits(p)
print(f"Target p: {p}")
print(f"Target Min Loss (Entropy): {target_entropy:.5f}\n")

def cross_entropy_bits(p, q):
    return -np.sum(p * np.log2(q + 1e-12))

def propose_neighbor(q, step_size):

    q_new = q.copy()
    K = len(q_new)
    
    i, j = np.random.choice(K, size=2, replace=False)
    
    delta = (np.random.rand() * 2 - 1) * step_size
    
    q_new[i] += delta
    q_new[j] -= delta
    
    if np.any(q_new <= 0):
        return q  
    
    q_new = q_new / q_new.sum()
    return q_new

np.random.seed(0)

q = np.array([1/3, 1/3, 1/3], dtype=float)
loss = cross_entropy_bits(p, q)

print("Start Hill Climbing...")
print(f"Initial q: {q.round(4)}, Loss: {loss:.5f}")

max_iters = 20000
base_step = 0.2   
decay = 0.8       

for t in range(max_iters + 1):
    
    if t % 1000 == 0:
        step_size = base_step * (decay ** (t // 1000))
        print(f"{t:05d}: Loss={loss:.5f} q={q.round(4)} step={step_size:.5f}")
    
    step_size = base_step * (decay ** (t // 1000))
    
    q_candidate = propose_neighbor(q, step_size)
    loss_candidate = cross_entropy_bits(p, q_candidate)
    
    if loss_candidate < loss:
        q = q_candidate
        loss = loss_candidate

print("-" * 60)
final_q = q
final_loss = loss

print("Final Result:")
print("Optimized q :", final_q.round(4))
print("Target    p :", p)
print(f"Final Loss  : {final_loss:.5f}")
print("Diff (q - p):", final_q - p)
