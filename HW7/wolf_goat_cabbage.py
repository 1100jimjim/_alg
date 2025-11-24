from collections import deque

# 狀態: (M, W, G, C)  分別代表 人, 狼, 羊, 菜 在左岸(L)或右岸(R)
LEFT = 'L'
RIGHT = 'R'

def is_safe(state):
    
    M, W, G, C = state

    if W == G and M != W:
        return False

    if G == C and M != G:
        return False

    return True


def move(state, passenger):
    
    M, W, G, C = state
    
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
    
    neighbors = []
    for passenger in [None, 'W', 'G', 'C']:
        ns = move(state, passenger)
        if ns is not None:
            neighbors.append((ns, passenger))
    return neighbors


def bfs(start, goal):
    
    queue = deque()
    queue.append(start)
    visited = set()
    visited.add(start)

    parent = {start: (None, None)}  

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
