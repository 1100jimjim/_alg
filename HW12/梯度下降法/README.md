```.py
import numpy as np

# =========================
# 1. 設定目標分佈 p
# =========================
p = np.array([0.5, 0.2, 0.3], dtype=float)

def entropy_bits(p):
    """用 log2 的 entropy，單位 bits。"""
    return -np.sum(p * np.log2(p + 1e-12))

target_entropy = entropy_bits(p)
print(f"Target p: {p}")
print(f"Target Min Loss (Entropy): {target_entropy:.5f}\n")

# =========================
# 2. softmax + cross entropy
# =========================
def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)

def cross_entropy_bits(p, q):
    """Cross Entropy 使用 log2（和 entropy 單位一致）"""
    return -np.sum(p * np.log2(q + 1e-12))

# =========================
# 3. 對 z 做梯度下降
# =========================

np.random.seed(0)

# 初始 z 設成全 0 → q = [1/3, 1/3, 1/3]
z = np.zeros(3)
q = softmax(z)
loss = cross_entropy_bits(p, q)

print("Start Gradient Descent...")
print(f"Initial q: {q.round(4)}, Loss: {loss:.5f}")

max_iters = 20000
base_step = 0.4    # 初始學習率
decay = 0.8        # step 衰減因子（每 1000 步）

for t in range(max_iters + 1):
    q = softmax(z)
    loss = cross_entropy_bits(p, q)

    # 每 1000 步印出狀態（模仿你的範例格式）
    if t % 1000 == 0:
        step = base_step * (decay ** (t // 1000))
        print(f"{t:05d}: Loss={loss:.5f} q={q.round(4)} step={step:.5f}")

    # 對 z 的梯度：q - p
    grad = q - p

    # 衰減後的學習率
    step = base_step * (decay ** (t // 1000))

    # 更新 z
    z -= step * grad

print("-" * 60)
final_q = softmax(z)
final_loss = cross_entropy_bits(p, final_q)

print("Final Result:")
print("Optimized q :", final_q.round(4))
print("Target    p :", p)
print(f"Final Loss  : {final_loss:.5f}")
print("Diff (q - p):", final_q - p)
```
