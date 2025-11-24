from typing import List, Tuple, Optional, Set

LEFT = 'L'
RIGHT = 'R'

State = Tuple[str, str, str, str]  # (M, W, G, C)


def is_safe(state: State) -> bool:
    
    M, W, G, C = state

    
    if W == G and M != W:
        return False

    
    if G == C and M != G:
        return False

    return True


def move(state: State, passenger: Optional[str]) -> Optional[State]:
    
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
    return next_state if is_safe(next_state) else None


def get_neighbors(state: State):
    neighbors = []
    for passenger in [None, 'W', 'G', 'C']:
        ns = move(state, passenger)
        if ns is not None:
            neighbors.append((ns, passenger))
    return neighbors


def dfs(current: State,
        goal: State,
        visited: Set[State],
        path: List[Tuple[State, Optional[str]]]) -> bool:
    
    if current == goal:
        return True  

    for ns, passenger in get_neighbors(current):
        if ns in visited:
            continue
        visited.add(ns)
        path.append((ns, passenger))
        if dfs(ns, goal, visited, path):
            return True  
        path.pop()

    return False


def print_solution(path: List[Tuple[State, Optional[str]]]):
    
    def side_str(s: State) -> str:
        return "人:{} 狼:{} 羊:{} 菜:{}".format(*s)

    print("DFS 找到的一組解：")
    for i, (state, passenger) in enumerate(path):
        if i == 0:
            print(f"步驟 {i}: 起點          -> {side_str(state)}")
        else:
            if passenger is None:
                action = "人單獨過河"
            else:
                name = {'W': '狼', 'G': '羊', 'C': '菜'}[passenger]
                action = f"人帶著「{name}」過河"
            print(f"步驟 {i}: {action:10s} -> {side_str(state)}")


if __name__ == "__main__":
    start_state: State = (LEFT, LEFT, LEFT, LEFT)
    goal_state: State = (RIGHT, RIGHT, RIGHT, RIGHT)

    visited = {start_state}
    
    path: List[Tuple[State, Optional[str]]] = [(start_state, None)]

    if dfs(start_state, goal_state, visited, path):
        print_solution(path)
    else:
        print("沒有找到解")
