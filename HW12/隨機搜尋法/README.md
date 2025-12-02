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
PS C:\ccc\py2cs> & C:/Users/user/AppData/Local/Programs/Python/Python312/python.exe c:/ccc/py2cs/HW/HW12-Immediately.py
Random Search for CE(p,q) minimum
Target p: [0.5 0.2 0.3]

Initial q: [0.2228 0.2665 0.5107], Loss: 1.75538
00000: Loss=1.60575 q=[0.4275 0.098  0.4744]
01000: Loss=1.48627 q=[0.514  0.1884 0.2975]
02000: Loss=1.48564 q=[0.5069 0.1951 0.298 ]
03000: Loss=1.48557 q=[0.4967 0.2047 0.2986]
04000: Loss=1.48557 q=[0.4967 0.2047 0.2986]
05000: Loss=1.48557 q=[0.4967 0.2047 0.2986]
06000: Loss=1.48557 q=[0.4967 0.2047 0.2986]
07000: Loss=1.48557 q=[0.4967 0.2047 0.2986]
08000: Loss=1.48557 q=[0.4967 0.2047 0.2986]
09000: Loss=1.48557 q=[0.4967 0.2047 0.2986]
10000: Loss=1.48557 q=[0.4967 0.2047 0.2986]
11000: Loss=1.48548 q=[0.501 0.199 0.3  ]
12000: Loss=1.48548 q=[0.501 0.199 0.3  ]
13000: Loss=1.48548 q=[0.501 0.199 0.3  ]
14000: Loss=1.48548 q=[0.501 0.199 0.3  ]
15000: Loss=1.48548 q=[0.501 0.199 0.3  ]
16000: Loss=1.48548 q=[0.501 0.199 0.3  ]
17000: Loss=1.48548 q=[0.501 0.199 0.3  ]
18000: Loss=1.48548 q=[0.501 0.199 0.3  ]
19000: Loss=1.48548 q=[0.501 0.199 0.3  ]
20000: Loss=1.48548 q=[0.501 0.199 0.3  ]
------------------------------------------------------------
Final Result:
Optimized q : [0.501 0.199 0.3  ]
Target    p : [0.5 0.2 0.3]
Final Loss  : 1.4854802702786105
Diff (q - p): [ 1.01980311e-03 -9.79365210e-04 -4.04379015e-05]
```
