程式碼:
---
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
執行結果:
---
```
PS C:\ccc\py2cs> & C:/Users/user/AppData/Local/Programs/Python/Python312/python.exe c:/ccc/py2cs/HW/HW12.py
Target p: [0.5 0.2 0.3]
Target Min Loss (Entropy): 1.48548

Start Gradient Descent...
Initial q: [0.3333 0.3333 0.3333], Loss: 1.58496
00000: Loss=1.58496 q=[0.3333 0.3333 0.3333] step=0.40000
01000: Loss=1.48548 q=[0.5 0.2 0.3] step=0.32000
02000: Loss=1.48548 q=[0.5 0.2 0.3] step=0.25600
03000: Loss=1.48548 q=[0.5 0.2 0.3] step=0.20480
04000: Loss=1.48548 q=[0.5 0.2 0.3] step=0.16384
05000: Loss=1.48548 q=[0.5 0.2 0.3] step=0.13107
06000: Loss=1.48548 q=[0.5 0.2 0.3] step=0.10486
07000: Loss=1.48548 q=[0.5 0.2 0.3] step=0.08389
08000: Loss=1.48548 q=[0.5 0.2 0.3] step=0.06711
09000: Loss=1.48548 q=[0.5 0.2 0.3] step=0.05369
10000: Loss=1.48548 q=[0.5 0.2 0.3] step=0.04295
11000: Loss=1.48548 q=[0.5 0.2 0.3] step=0.03436
12000: Loss=1.48548 q=[0.5 0.2 0.3] step=0.02749
13000: Loss=1.48548 q=[0.5 0.2 0.3] step=0.02199
14000: Loss=1.48548 q=[0.5 0.2 0.3] step=0.01759
15000: Loss=1.48548 q=[0.5 0.2 0.3] step=0.01407
16000: Loss=1.48548 q=[0.5 0.2 0.3] step=0.01126
17000: Loss=1.48548 q=[0.5 0.2 0.3] step=0.00901
18000: Loss=1.48548 q=[0.5 0.2 0.3] step=0.00721
19000: Loss=1.48548 q=[0.5 0.2 0.3] step=0.00576
20000: Loss=1.48548 q=[0.5 0.2 0.3] step=0.00461
------------------------------------------------------------
Final Result:
Optimized q : [0.5 0.2 0.3]
Target    p : [0.5 0.2 0.3]
Final Loss  : 1.48548
Diff (q - p): [0.00000000e+00 5.55111512e-17 0.00000000e+00]
```
