import random

# ===== 1. 產生一組假資料： y = 2x + 1 + noise =====
random.seed(0)
X = [i for i in range(20)]
Y = [2.0 * x + 1.0 + random.uniform(-1.0, 1.0) for x in X]
N = len(X)

# ===== 2. 定義成本函數 (MSE) =====
def cost(w, b):
    s = 0.0
    for x, y in zip(X, Y):
        y_pred = w * x + b
        s += (y - y_pred) ** 2
    return s / N

# ===== 3. 改良法 (local improvement) 解線性迴歸 =====

# 初始解（可以換成 0,0）
w = random.uniform(-1, 1)
b = random.uniform(-1, 1)

step_w = 0.1
step_b = 0.1

min_step = 1e-6
max_iter = 10000

best_cost = cost(w, b)

for it in range(max_iter):
    improved = False
    best_local_w = w
    best_local_b = b
    best_local_cost = best_cost

    # 掃描 3x3 鄰域：k, l in {-1, 0, 1}
    for k in [-1, 0, 1]:
        for l in [-1, 0, 1]:
            w_cand = w + k * step_w
            b_cand = b + l * step_b
            c = cost(w_cand, b_cand)

            if c < best_local_cost:
                best_local_cost = c
                best_local_w = w_cand
                best_local_b = b_cand
                improved = True

    if improved:
        # 有更好的鄰居 → 做改良
        w, b = best_local_w, best_local_b
        best_cost = best_local_cost
    else:
        # 鄰域內沒有更好的 → 縮小步長
        step_w *= 0.5
        step_b *= 0.5
        if step_w < min_step and step_b < min_step:
            # 步長太小，視為收斂
            break

print("改良法找到的參數：")
print("w =", w)
print("b =", b)
print("最終 MSE =", best_cost)
