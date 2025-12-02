import random

# ====== 1. 造一組假資料： y = 2x + 1 + noise ======
random.seed(0)
X = [i for i in range(20)]
Y = [2.0 * x + 1.0 + random.uniform(-1.0, 1.0) for x in X]
N = len(X)

# ====== 2. 定義成本函數 (MSE) ======
def cost(w, b):
    s = 0.0
    for x, y in zip(X, Y):
        y_pred = w * x + b
        s += (y - y_pred) ** 2
    return s / N

# ====== 3. 非梯度下降的「貪婪搜尋」線性迴歸 ======

# 初始化
w = random.uniform(-1, 1)
b = random.uniform(-1, 1)
best_cost = cost(w, b)

step = 0.1          # 初始步長
max_iter = 10000
min_step = 1e-6     # 步長太小就停止

for it in range(max_iter):
    # 產生 4 個候選解
    candidates = [
        (w + step, b),
        (w - step, b),
        (w, b + step),
        (w, b - step),
    ]

    best_local_w = w
    best_local_b = b
    best_local_cost = best_cost

    # 在這一輪裡面貪婪地選 cost 最小的候選解
    for cw, cb in candidates:
        c = cost(cw, cb)
        if c < best_local_cost:
            best_local_cost = c
            best_local_w = cw
            best_local_b = cb

    if best_local_cost < best_cost:
        # 有改善 → 接受這個 move
        w, b, best_cost = best_local_w, best_local_b, best_local_cost
    else:
        # 沒改善 → 縮小步長
        step *= 0.5
        if step < min_step:
            break

print("最終找到的參數：")
print("w =", w)
print("b =", b)
print("最終 MSE =", best_cost)
