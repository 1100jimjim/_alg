import random

# ===== 1. 產生假資料： y = 2x + 1 + noise =====
random.seed(0)
X = [i for i in range(20)]
Y = [2.0 * x + 1.0 + random.uniform(-1.0, 1.0) for x in X]
N = len(X)

# ===== 2. 定義成本函數 (MSE)，方便觀察收斂 =====
def cost(w, b):
    s = 0.0
    for x, y in zip(X, Y):
        y_pred = w * x + b
        s += (y - y_pred) ** 2
    return s / N

# ===== 3. 梯度下降法 =====

# 初始化參數
w = 0.0
b = 0.0

alpha = 0.0005     # 學習率，可以調整
max_iter = 20000

for it in range(max_iter):
    # 計算梯度 (對 w, b 的偏導)
    dJ_dw = 0.0
    dJ_db = 0.0

    for x, y in zip(X, Y):
        y_pred = w * x + b
        e = y - y_pred
        dJ_dw += -2 * x * e
        dJ_db += -2 * e

    dJ_dw /= N
    dJ_db /= N

    # 更新參數
    w = w - alpha * dJ_dw
    b = b - alpha * dJ_db

    # 每隔一段時間印一次 cost 看收斂狀況
    if (it + 1) % 2000 == 0:
        print(f"iter {it+1}, cost = {cost(w, b):.4f}, w = {w:.4f}, b = {b:.4f}")

print("\n最終結果：")
print("w =", w)
print("b =", b)
print("最終 MSE =", cost(w, b))
