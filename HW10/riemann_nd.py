import numpy as np
import math

def riemann_integral_1d(f, a, b, n=1000, method="midpoint"):
    """
    用黎曼和近似計算一維積分 ∫_a^b f(x) dx

    參數:
        f      : 被積函數 f(x)
        a, b   : 積分上下限 (a < b)
        n      : 切成幾等分 (越大越準)
        method : "left", "right", "midpoint", "trapezoid"

    回傳:
        近似的積分值 (float)
    """
    if n <= 0:
        raise ValueError("n 必須是正整數")
    if a == b:
        return 0.0
    if a > b:
        return -riemann_integral_1d(f, b, a, n=n, method=method)

    dx = (b - a) / n

    if method == "left":
        # 左端點
        x = a + dx * np.arange(0, n)
        fx = np.vectorize(f)(x)
        return np.sum(fx) * dx

    elif method == "right":
        # 右端點
        x = a + dx * np.arange(1, n + 1)
        fx = np.vectorize(f)(x)
        return np.sum(fx) * dx

    elif method == "midpoint":
        # 中點
        x = a + dx * (np.arange(0, n) + 0.5)
        fx = np.vectorize(f)(x)
        return np.sum(fx) * dx

    elif method == "trapezoid":
        # 梯形法
        x = a + dx * np.arange(0, n + 1)
        fx = np.vectorize(f)(x)
        return (fx[0] + fx[-1] + 2 * np.sum(fx[1:-1])) * dx / 2.0

    else:
        raise ValueError(f"未知的 method: {method}")


def riemann_integral_nd(f, bounds, divisions, method="midpoint"):
    """
    用規則格點的黎曼和近似計算 n 維積分：
        ∫_Ω f(x1,...,xn) d x

    其中 Ω 是 n 維超長方體：
        [a1, b1] × [a2, b2] × ... × [an, bn]

    參數:
        f : callable
            被積函數 f(x)，x 為 shape (..., n) 的 numpy array，
            或 f 接受一個長度為 n 的一維向量。
        bounds : list[tuple[float, float]]
            每一維的積分範圍 [(a1,b1), (a2,b2), ..., (an,bn)]。
        divisions : int 或 list[int]
            每一維切幾等分，如果給 int，代表每一維都切一樣多。
        method : str
            "midpoint"（目前實作），概念是 n 維中點法。
            （要改成 left/right 也可以，只是樣本點位置不同）

    回傳:
        近似的積分值 (float)
    """
    bounds = np.array(bounds, dtype=float)
    n_dim = bounds.shape[0]

    # divisions 可以是單一 int 或 list
    if isinstance(divisions, int):
        m_list = [divisions] * n_dim
    else:
        if len(divisions) != n_dim:
            raise ValueError("divisions 長度必須和維度數量相同")
        m_list = list(divisions)

    # 每一維的 dx
    a = bounds[:, 0]
    b = bounds[:, 1]
    lengths = b - a
    dx = lengths / np.array(m_list, dtype=float)

    # 建立每一維的格點（這裡用「中點法」）
    grids_1d = []
    for j in range(n_dim):
        mj = m_list[j]
        aj = a[j]
        dxj = dx[j]
        # 中點
        midpoints = aj + dxj * (np.arange(mj) + 0.5)
        grids_1d.append(midpoints)

    # 建立 n 維格點
    mesh = np.meshgrid(*grids_1d, indexing='ij')  # 每個 mesh[j] shape: (m1, m2, ..., mn)

    # 將格點組成最後一維是座標的陣列 points shape: (m1, m2, ..., mn, n_dim)
    stacked = np.stack(mesh, axis=-1)

    # 嘗試向量化呼叫 f
    try:
        values = f(stacked)       # 期望回傳 shape: (m1, m2, ..., mn)
        values = np.asarray(values)
        if values.shape != stacked.shape[:-1]:
            raise ValueError
    except Exception:
        # 若 f 不支援向量化，就逐點計算
        it = np.nditer(np.zeros(stacked.shape[:-1]), flags=['multi_index'])
        values = np.empty(stacked.shape[:-1], dtype=float)
        for _ in it:
            idx = it.multi_index
            point = stacked[idx]      # 長度為 n_dim 的向量
            values[idx] = f(point)

    # 每一小格的體積
    cell_volume = np.prod(dx)

    # n 維中點黎曼和 = Σ f(x_cell_center) * cell_volume
    integral = np.sum(values) * cell_volume
    return integral


def riemann_integral_nd_cube(f, n_dim, a=0.0, b=1.0, divisions=20, method="midpoint"):
    """
    方便用的封裝：在 n 維超立方體 [a,b]^n 上做 n 維黎曼積分。
    """
    bounds = [(a, b)] * n_dim
    return riemann_integral_nd(f, bounds, divisions, method=method)


# ================= 測試區 =================
if __name__ == "__main__":
    # 一維測試：∫_0^1 x^2 dx = 1/3
    def f1(x):
        return x**2

    exact_1d = 1.0 / 3.0
    est_1d = riemann_integral_1d(f1, 0.0, 1.0, n=1000, method="midpoint")
    print("1D test: ∫_0^1 x^2 dx")
    print("  estimate =", est_1d, ", exact =", exact_1d, ", error =", abs(est_1d - exact_1d))

    # 二維測試：∫_0^1 ∫_0^1 (x + y) dx dy
    # 理論：∫_0^1 ∫_0^1 x dx dy + ∫_0^1 ∫_0^1 y dx dy = 1/2 + 1/2 = 1
    def f2_xy(points):
        # points shape: (..., 2)
        x = points[..., 0]
        y = points[..., 1]
        return x + y

    exact_2d = 1.0
    est_2d = riemann_integral_nd_cube(f2_xy, n_dim=2, a=0.0, b=1.0, divisions=50)
    print("\n2D test: ∫_0^1∫_0^1 (x + y) dx dy")
    print("  estimate =", est_2d, ", exact =", exact_2d, ", error =", abs(est_2d - exact_2d))

    # 三維測試：∫_[0,1]^3 (x^2 + y^2 + z^2) dV
    # 每一維 E[x^2] = 1/3，所以 E[x^2 + y^2 + z^2] = 3*(1/3)=1，體積=1 => 積分=1
    def f3_xyz(points):
        # points shape: (..., 3)
        return np.sum(points**2, axis=-1)

    exact_3d = 1.0
    est_3d = riemann_integral_nd_cube(f3_xyz, n_dim=3, a=0.0, b=1.0, divisions=20)
    print("\n3D test: ∫_[0,1]^3 (x^2 + y^2 + z^2) dV")
    print("  estimate =", est_3d, ", exact =", exact_3d, ", error =", abs(est_3d - exact_3d))
