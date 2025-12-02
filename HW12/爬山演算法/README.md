程式碼:
---
```.py
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
```
執行結果
---
```
PS C:\ccc\py2cs> & C:/Users/user/AppData/Local/Programs/Python/Python312/python.exe c:/ccc/py2cs/HW/HW12-Mountain.py
Target p: [0.5 0.2 0.3]
Target Min Loss (Entropy): 1.48548

Start Hill Climbing...
Initial q: [0.3333 0.3333 0.3333], Loss: 1.58496
00000: Loss=1.58496 q=[0.3333 0.3333 0.3333] step=0.20000
01000: Loss=1.48548 q=[0.5011 0.1998 0.2991] step=0.16000
02000: Loss=1.48548 q=[0.5001 0.1999 0.3   ] step=0.12800
03000: Loss=1.48548 q=[0.5001 0.2    0.2999] step=0.10240
04000: Loss=1.48548 q=[0.5001 0.2    0.2999] step=0.08192
05000: Loss=1.48548 q=[0.5001 0.2    0.2999] step=0.06554
06000: Loss=1.48548 q=[0.5 0.2 0.3] step=0.05243
07000: Loss=1.48548 q=[0.5 0.2 0.3] step=0.04194
08000: Loss=1.48548 q=[0.5 0.2 0.3] step=0.03355
09000: Loss=1.48548 q=[0.5 0.2 0.3] step=0.02684
10000: Loss=1.48548 q=[0.5 0.2 0.3] step=0.02147
11000: Loss=1.48548 q=[0.5 0.2 0.3] step=0.01718
12000: Loss=1.48548 q=[0.5 0.2 0.3] step=0.01374
13000: Loss=1.48548 q=[0.5 0.2 0.3] step=0.01100
14000: Loss=1.48548 q=[0.5 0.2 0.3] step=0.00880
15000: Loss=1.48548 q=[0.5 0.2 0.3] step=0.00704
16000: Loss=1.48548 q=[0.5 0.2 0.3] step=0.00563
17000: Loss=1.48548 q=[0.5 0.2 0.3] step=0.00450
18000: Loss=1.48548 q=[0.5 0.2 0.3] step=0.00360
19000: Loss=1.48548 q=[0.5 0.2 0.3] step=0.00288
20000: Loss=1.48548 q=[0.5 0.2 0.3] step=0.00231
------------------------------------------------------------
Final Result:
Optimized q : [0.5 0.2 0.3]
Target    p : [0.5 0.2 0.3]
Final Loss  : 1.48548
Diff (q - p): [ 3.80960678e-06 -2.78115244e-06 -1.02845435e-06]
```
