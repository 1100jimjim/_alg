# 方法 1
def power2n_1(n):
    return 2 ** n

# 方法 2a：用遞迴
def power2n_2a(n):
    if n == 0:
        return 1
    return power2n_2a(n-1) + power2n_2a(n-1)

# 方法 2b：用遞迴
def power2n_2b(n):
    if n == 0:
        return 1
    return 2 * power2n_2b(n-1)

# 方法 3：用遞迴 + 查表
p2Table = [-1] * 101
p2Table[0] = 1

def power2n_3(n):
    if p2Table[n] != -1:
        return p2Table[n]
    p2Table[n] = power2n_3(n-1) + power2n_3(n-1)
    return p2Table[n]

if __name__ == "__main__":
    for i in range(0, 11):
        print(i, power2n_1(i), power2n_2a(i), power2n_2b(i), power2n_3(i))
