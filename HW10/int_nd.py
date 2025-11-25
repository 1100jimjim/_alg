import numpy as np

def integrate_nd_box(f, bounds, num_samples=100_000, rng=None):
    """
    使用蒙地卡羅法估計 n 維函數在超長方體上的定積分。

    參數：
        f : callable
            被積函數，接受形如 x (shape: (n,)) 的向量，回傳一個實數。
        bounds : list[tuple[float, float]]
            每一維的積分範圍 [(a1, b1), (a2, b2), ..., (an, bn)]。
        num_samples : int
            隨機取樣點數，越大越準但越慢。
        rng : np.random.Generator | None
            隨機數產生器（可傳入 np.random.default_rng(seed) 控制隨機種子）。

    回傳：
        estimate : float
            積分估計值。
        std_error : float
            估計值的標準誤差。
    """
    bounds = np.array(bounds, dtype=float)
    n = bounds.shape[0]  # 維度 n

    if rng is None:
        rng = np.random.default_rng()

    # 計算超長方體體積：prod (bi - ai)
    lengths = bounds[:, 1] - bounds[:, 0]
    volume = np.prod(lengths)

    # 在每一維 [ai, bi] 上做 uniform 取樣
    # samples shape: (num_samples, n)
    u = rng.random((num_samples, n))
    samples = bounds[:, 0] + u * lengths  # ai + u * (bi - ai)

    # 計算 f 在每個樣本點的值
    # 支援 f 一次吃一個向量 (num_samples, n)，或一個一個吃 (n,)
    try:
        values = f(samples)          # 嘗試 vectorized
        values = np.asarray(values)
        if values.shape != (num_samples,):
            raise ValueError
    except Exception:
        # 不是向量化版本，就一個一個算
        values = np.array([f(x) for x in samples])

    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1)  # 樣本標準差

    estimate = volume * mean_val
    std_error = volume * std_val / np.sqrt(num_samples)

    return estimate, std_error


def integrate_nd_cube(f, n, a=0.0, b=1.0, num_samples=100_000, rng=None):
    """
    在 n 維超立方體 [a, b]^n 上做積分的方便包裝函式。

    參數：
        f : callable
            被積函數，接受長度 n 的向量。
        n : int
            維度。
        a, b : float
            每一維的下界與上界（全部維度共用）。
        num_samples : int
            取樣點數。
        rng : np.random.Generator | None
            隨機數產生器。

    回傳：
        estimate, std_error
    """
    bounds = [(a, b)] * n
    return integrate_nd_box(f, bounds, num_samples=num_samples, rng=rng)


if __name__ == "__main__":
    # ======== 範例 1: f(x) = 1，積分結果應該是體積 (b-a)^n ========
    n = 5
    a, b = 0.0, 1.0

    def f_const(x):
        return 1.0

    est, err = integrate_nd_cube(f_const, n, a=a, b=b, num_samples=200_000)
    print(f"Example 1: ∫_[{a},{b}]^{n} 1 dx")
    print(f"  Monte Carlo 估計值 = {est}")
    print(f"  理論值 = {(b - a) ** n}")
    print(f"  標準誤差 ≈ {err}")
    print()

    # ======== 範例 2: f(x) = x1^2 + x2^2 + ... + xn^2, 在 [0,1]^n ========
    n = 3

    def f_sum_square(x):
        # x shape: (..., n)
        x = np.asarray(x)
        return np.sum(x**2, axis=-1)

    est, err = integrate_nd_cube(f_sum_square, n, a=0.0, b=1.0, num_samples=200_000)
    # 理論值：每一維 ∫_0^1 x^2 dx = 1/3
    # E[x1^2 + ... + xn^2] = n/3，所以積分 = n/3 * 體積(=1)
    exact = n / 3.0

    print(f"Example 2: ∫_[0,1]^{n} (x1^2 + ... + xn^2) dx")
    print(f"  Monte Carlo 估計值 = {est}")
    print(f"  理論值 = {exact}")
    print(f"  標準誤差 ≈ {err}")
