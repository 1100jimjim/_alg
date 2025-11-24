from collections import deque

# 狀態: (M, W, G, C)  分別代表 人, 狼, 羊, 菜 在左岸(L)或右岸(R)
LEFT = 'L'
RIGHT = 'R'

def is_safe(state):
    """檢查該狀態是否安全（沒有人被吃）"""
    M, W, G, C = state

    if W == G and M != W:
        return False

    if G == C and M != G:
        return False

    return True


def move(state, passenger):
    """
    根據 passenger 產生下一個狀態:
    passenger 可以是 None, 'W', 'G', 'C'
    """
    M, W, G, C = state
    # 人移動方向：從 L 到 R 或 R 到 L
    new_side = RIGHT if M == LEFT else LEFT

    M2, W2, G2, C2 = M, W, G, C
    M2 = new_side

    if passenger == 'W':
        if W != M:   
            return None
        W2 = new_side
    elif passenger == 'G':
        if G != M:
            return None
        G2 = new_side
    elif passenger == 'C':
        if C != M:
            return None
        C2 = new_side
    elif passenger is None:
        pass
    else:
        return None 

    next_state = (M2, W2, G2, C2)
    
    if is_safe(next_state):
        return next_state
    else:
        return None


def get_neighbors(state):
    """從目前狀態產生所有合法的下一步狀態"""
    neighbors = []
    for passenger in [None, 'W', 'G', 'C']:
        ns = move(state, passenger)
        if ns is not None:
            neighbors.append((ns, passenger))
    return neighbors


def bfs(start, goal):
    """用 BFS 找出從 start 到 goal 的最短路徑"""
    queue = deque()
    queue.append(start)
    visited = set()
    visited.add(start)

    parent = {start: (None, None)}  # state: (prev_state, passenger)

    while queue:
        current = queue.popleft()
        if current == goal:
            
            path = []
            s = current
            while s is not None:
                prev, passenger = parent[s]
                path.append((s, passenger))
                s = prev
            path.reverse()
            return path

        for ns, passenger in get_neighbors(current):
            if ns not in visited:
                visited.add(ns)
                parent[ns] = (current, passenger)
                queue.append(ns)

    return None  


def print_solution(path):
    """把路徑用比較好讀的方式印出來"""
    def side_str(s):
        return "人:{} 狼:{} 羊:{} 菜:{}".format(*s)

    print("解的步驟：")
    for i, (state, passenger) in enumerate(path):
        if i == 0:
            print(f"步驟 {i}: 起點  -> {side_str(state)}")
        else:
            if passenger is None:
                action = "人單獨過河"
            else:
                action = f"人帶著「{passenger}」過河"
            print(f"步驟 {i}: {action:8s} -> {side_str(state)}")


if __name__ == "__main__":
    start_state = (LEFT, LEFT, LEFT, LEFT)
    goal_state = (RIGHT, RIGHT, RIGHT, RIGHT)

    path = bfs(start_state, goal_state)
    if path is None:
        print("沒有找到解")
    else:
        print_solution(path)
