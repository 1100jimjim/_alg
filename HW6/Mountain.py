import random
import math

# 1. 建資料：y = 2x + 1 + noise
random.seed(42)
X = [i for i in range(20)]
Y = [2.0 * x + 1.0 + random.uniform(-1.0, 1.0) for x in X]
N = len(X)

# 2. 定義 cost 函數 (MSE)
def cost(w, b):
    s = 0.0
    for x, y in zip(X, Y):
        y_pred = w * x + b
        s += (y - y_pred) ** 2
    return s / N

# 3. 爬山演算法 (其實在 "往 cost 更低的方向走")
# 初始化
w = random.uniform(-1, 1)
b = random.uniform(-1, 1)
best_cost = cost(w, b)

step = 0.05       # 鄰居擾動的範圍
max_iter = 10000

for it in range(max_iter):
    # 產生鄰居解
    dw = random.uniform(-step, step)
    db = random.uniform(-step, step)
    w_new = w + dw
    b_new = b + db

    new_cost = cost(w_new, b_new)

    # 若比較好就接受
    if new_cost < best_cost:
        w = w_new
        b = b_new
        best_cost = new_cost

    # 也可以每隔一段時間縮小 step，讓搜尋更細緻
    if (it+1) % 2000 == 0:
        step *= 0.5  # 降低步長

print("找到的 w, b:", w, b)
print("最終 MSE:", best_cost)
