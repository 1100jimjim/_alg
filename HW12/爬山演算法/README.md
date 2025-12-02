程式碼:
---
```.py
import numpy as np

# =========================
# 1. 目標分佈 p
# =========================
p = np.array([0.5, 0.2, 0.3], dtype=float)

def entropy_bits(p):
    return -np.sum(p * np.log2(p + 1e-12))

target_entropy = entropy_bits(p)
print(f"Target p: {p}")
print(f"Target Min Loss (Entropy): {target_entropy:.5f}\n")

# =========================
# 2. Cross Entropy（log2，單位 bits）
# =========================
def cross_entropy_bits(p, q):
    return -np.sum(p * np.log2(q + 1e-12))

# =========================
# 3. 產生鄰居：在 simplex 上做小擾動
# =========================
def propose_neighbor(q, step_size):
    """
    給現在的 q，產生一個鄰居 q_new：
    - 對兩個不同維度 i, j 做 +δ, -δ
    - 確保所有元素 > 0
    - 再做一次 normalize 避免浮點誤差
    """
    q_new = q.copy()
    K = len(q_new)
    
    # 隨機挑兩個不同 index
    i, j = np.random.choice(K, size=2, replace=False)
    
    # 在 [-step_size, step_size] 範圍內取一個擾動
    delta = (np.random.rand() * 2 - 1) * step_size
    
    q_new[i] += delta
    q_new[j] -= delta
    
    # 如果有變成負數就直接回原 q（放棄這個候選）
    if np.any(q_new <= 0):
        return q  # 不接受負數的候選
    
    # 正規化避免數值飄移
    q_new = q_new / q_new.sum()
    return q_new

# =========================
# 4. Hill Climbing 主流程
# =========================
np.random.seed(0)

# 初始 q：均勻分布
q = np.array([1/3, 1/3, 1/3], dtype=float)
loss = cross_entropy_bits(p, q)

print("Start Hill Climbing...")
print(f"Initial q: {q.round(4)}, Loss: {loss:.5f}")

max_iters = 20000
base_step = 0.2   # 初始擾動大小
decay = 0.8       # 每 1000 步衰減一次

for t in range(max_iters + 1):
    # 每 1000 步印出狀態
    if t % 1000 == 0:
        step_size = base_step * (decay ** (t // 1000))
        print(f"{t:05d}: Loss={loss:.5f} q={q.round(4)} step={step_size:.5f}")
    
    # 目前的 step_size（擾動大小）
    step_size = base_step * (decay ** (t // 1000))
    
    # 產生鄰居
    q_candidate = propose_neighbor(q, step_size)
    loss_candidate = cross_entropy_bits(p, q_candidate)
    
    # Hill Climbing：只要比較好就接受
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
```
