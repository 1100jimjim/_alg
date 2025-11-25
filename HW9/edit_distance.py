def edit_distance(s1: str, s2: str):
    """
    計算字串 s1 與 s2 的最小編輯距離 (Levenshtein Distance)
    允許的操作：插入、刪除、取代，每個操作成本為 1
    回傳：
      dist: 最小編輯距離
      dp:   動態規劃表 (2D list)，可用來觀察計算過程
    """
    m, n = len(s1), len(s2)

    # 建立 (m+1) x (n+1) 的 DP 表
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 初始化：s1 前 i 個字變成空字串，需要 i 次刪除
    for i in range(1, m + 1):
        dp[i][0] = i

    # 初始化：空字串變成 s2 前 j 個字，需要 j 次插入
    for j in range(1, n + 1):
        dp[0][j] = j

    # 填 DP 表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0   # 字元相同，不需取代
            else:
                cost = 1   # 字元不同，取代成本為 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,      # 刪除 s1[i-1]
                dp[i][j - 1] + 1,      # 插入 s2[j-1]
                dp[i - 1][j - 1] + cost  # 取代 s1[i-1] -> s2[j-1]
            )

    return dp[m][n], dp


def print_dp_table(s1: str, s2: str, dp):
    """
    把 DP 表格印得比較漂亮，方便學習與觀察。
    會在上方與左方標出字元。
    """
    m, n = len(s1), len(s2)

    # 先印第一列標頭
    print("    ", end="")
    print("  ".join([" "] + list(s2)))

    # 逐列印出 DP 值
    for i in range(m + 1):
        if i == 0:
            row_label = " "
        else:
            row_label = s1[i - 1]
        print(f"{row_label:>2} ", end="")

        for j in range(n + 1):
            print(f"{dp[i][j]:2}", end=" ")
        print()


if __name__ == "__main__":
    s1 = input("請輸入字串1：")
    s2 = input("請輸入字串2：")

    dist, dp = edit_distance(s1, s2)
    print(f"\n「{s1}」 變成 「{s2}」 的最小編輯距離為：{dist}\n")

    print("DP 表格如下（可用來觀察動態規劃過程）：")
    print_dp_table(s1, s2, dp)
