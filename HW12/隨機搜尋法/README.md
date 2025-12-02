程式碼:
---
```.py
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
```
執行結果:
---
```
PS C:\ccc\py2cs> & C:/Users/user/AppData/Local/Programs/Python/Python312/python.exe c:/ccc/py2cs/HW/HW12-Random.py
Random Search for CE(p,q) minimum
Target p: [0.5 0.2 0.3]

Initial q: [0.2201 0.0286 0.7512], Loss: 2.24072
00000: Loss=1.82764 q=[0.224  0.1514 0.6246]
01000: Loss=1.48574 q=[0.4969 0.2076 0.2955]
02000: Loss=1.48574 q=[0.4969 0.2076 0.2955]
03000: Loss=1.48574 q=[0.4969 0.2076 0.2955]
04000: Loss=1.48574 q=[0.4969 0.2076 0.2955]
05000: Loss=1.48574 q=[0.4969 0.2076 0.2955]
06000: Loss=1.48574 q=[0.4969 0.2076 0.2955]
07000: Loss=1.48574 q=[0.4969 0.2076 0.2955]
08000: Loss=1.48560 q=[0.4933 0.2028 0.3039]
09000: Loss=1.48560 q=[0.4933 0.2028 0.3039]
10000: Loss=1.48560 q=[0.4933 0.2028 0.3039]
11000: Loss=1.48560 q=[0.4933 0.2028 0.3039]
12000: Loss=1.48552 q=[0.4977 0.203  0.2993]
13000: Loss=1.48552 q=[0.4977 0.203  0.2993]
14000: Loss=1.48552 q=[0.4977 0.203  0.2993]
15000: Loss=1.48552 q=[0.4977 0.203  0.2993]
16000: Loss=1.48552 q=[0.4977 0.203  0.2993]
17000: Loss=1.48552 q=[0.4977 0.203  0.2993]
18000: Loss=1.48552 q=[0.4977 0.203  0.2993]
19000: Loss=1.48552 q=[0.4977 0.203  0.2993]
20000: Loss=1.48552 q=[0.4977 0.203  0.2993]
------------------------------------------------------------
Final Result:
Optimized q : [0.4977 0.203  0.2993]
Target    p : [0.5 0.2 0.3]
Final Loss  : 1.4855167230693143
Diff (q - p): [-0.00228049  0.00302062 -0.00074013]
```
