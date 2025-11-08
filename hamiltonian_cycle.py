"""
Hamiltonian Cycle algorithm for Snake game.
Follows a cycle that visits every cell exactly once, guaranteeing the snake never traps itself.
"""

from typing import List, Tuple, Optional, Set, Dict


def generate_hamiltonian_cycle(grid_size: int) -> List[Tuple[int, int]]:
    """
    Generate a Hamiltonian cycle for the grid using a spiral pattern.
    This creates a cycle that visits every cell exactly once.
    
    Args:
        grid_size: Size of the grid
    
    Returns:
        List of positions forming a Hamiltonian cycle
    """
    cycle = []
    visited = [[False] * grid_size for _ in range(grid_size)]
    
    # Use a proper spiral pattern starting from top-left
    # Directions: right, down, left, up
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    dir_idx = 0
    
    x, y = 0, 0
    
    for _ in range(grid_size * grid_size):
        cycle.append((x, y))
        visited[x][y] = True
        
        # Calculate next position
        dx, dy = directions[dir_idx]
        next_x = x + dx
        next_y = y + dy
        
        # Check if we need to change direction
        # Change direction if:
        # 1. Next position is out of bounds, OR
        # 2. Next position is already visited
        if (next_x < 0 or next_x >= grid_size or 
            next_y < 0 or next_y >= grid_size or 
            visited[next_x][next_y]):
            # Change direction
            dir_idx = (dir_idx + 1) % 4
            dx, dy = directions[dir_idx]
            next_x = x + dx
            next_y = y + dy
        
        x, y = next_x, next_y
    
    return cycle


def find_cycle_position(cycle: List[Tuple[int, int]], position: Tuple[int, int]) -> int:
    """Find the index of a position in the cycle"""
    try:
        return cycle.index(position)
    except ValueError:
        return -1


def get_next_in_cycle(cycle: List[Tuple[int, int]], current_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """Get the next position in the cycle after current position"""
    idx = find_cycle_position(cycle, current_pos)
    if idx == -1:
        return None
    next_idx = (idx + 1) % len(cycle)
    return cycle[next_idx]


def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """Calculate Manhattan distance"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def get_neighbors(position: Tuple[int, int], grid_size: int, obstacles: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Get valid neighboring positions"""
    x, y = position
    neighbors = []
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    
    for dx, dy in directions:
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < grid_size and 0 <= new_y < grid_size:
            new_pos = (new_x, new_y)
            if new_pos not in obstacles:
                neighbors.append(new_pos)
    
    return neighbors


def find_path_to_goal(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    obstacles: Set[Tuple[int, int]],
    grid_size: int,
    max_depth: int = 10
) -> Optional[List[Tuple[int, int]]]:
    """
    Find a simple path from start to goal using BFS.
    Used to deviate from cycle to get apple.
    """
    from collections import deque
    
    if start == goal:
        return [start]
    
    queue = deque([(start, [start])])
    visited = {start}
    
    while queue:
        current, path = queue.popleft()
        
        if len(path) > max_depth:
            continue
        
        if current == goal:
            return path
        
        neighbors = get_neighbors(current, grid_size, obstacles)
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return None


def is_safe_to_deviate(
    snake_body: List[Tuple[int, int]],
    apple: Tuple[int, int],
    deviation_path: List[Tuple[int, int]],
    cycle: List[Tuple[int, int]],
    grid_size: int
) -> bool:
    """
    Check if it's safe to deviate from the cycle to get the apple.
    We need to ensure we can return to the cycle after getting the apple.
    """
    if not deviation_path or len(deviation_path) < 2:
        return False
    
    # Simulate snake following deviation path
    simulated_snake = snake_body.copy()
    
    for i in range(1, len(deviation_path)):
        next_pos = deviation_path[i]
        
        # Move snake
        simulated_snake.insert(0, next_pos)
        if next_pos != apple:
            if len(simulated_snake) > 1:
                simulated_snake.pop()
    
    # After getting apple, check if we can return to cycle
    end_pos = simulated_snake[0]
    next_cycle_pos = get_next_in_cycle(cycle, end_pos)
    
    if next_cycle_pos is None:
        return False
    
    # Check if next cycle position is reachable
    obstacles_after = set(simulated_snake[1:])
    neighbors = get_neighbors(end_pos, grid_size, obstacles_after)
    
    return next_cycle_pos in neighbors


def get_next_direction(
    snake_head: Tuple[int, int],
    apple: Tuple[int, int],
    snake_body: List[Tuple[int, int]],
    grid_size: int,
    current_direction: Tuple[int, int],
    cycle: Optional[List[Tuple[int, int]]] = None
) -> Tuple[Optional[Tuple[int, int]], Optional[List[Tuple[int, int]]]]:
    """
    Get the next direction using Hamiltonian Cycle approach.
    
    Args:
        snake_head: Current position of snake head
        apple: Position of apple
        snake_body: List of all snake body positions (including head)
        grid_size: Size of the grid
        current_direction: Current direction (for fallback)
        cycle: Pre-computed Hamiltonian cycle (will generate if None)
    
    Returns:
        Tuple of (next_direction, path)
    """
    # Generate or use provided cycle
    if cycle is None:
        cycle = generate_hamiltonian_cycle(grid_size)
    
    # Convert snake body to obstacles (excluding head and tail)
    obstacles = set(snake_body[1:-1]) if len(snake_body) > 2 else set(snake_body[1:])
    
    # Check if we're on the cycle
    head_idx = find_cycle_position(cycle, snake_head)
    
    # Get valid neighbors (adjacent cells only)
    valid_neighbors = get_neighbors(snake_head, grid_size, obstacles)
    
    if not valid_neighbors:
        # No valid moves, game over
        return current_direction, None
    
    # Check if apple is close and safe to get (only if adjacent)
    if snake_head == apple:
        # Already on apple, just move forward
        pass
    elif manhattan_distance(snake_head, apple) == 1 and apple in valid_neighbors:
        # Apple is adjacent, check if safe to get
        deviation_path = [snake_head, apple]
        if is_safe_to_deviate(snake_body, apple, deviation_path, cycle, grid_size):
            # Safe to get apple
            dx = apple[0] - snake_head[0]
            dy = apple[1] - snake_head[1]
            return (dx, dy), deviation_path
    
    # Try to follow the cycle
    if head_idx != -1:
        # We're on the cycle, try to follow it
        next_cycle_pos = get_next_in_cycle(cycle, snake_head)
        
        if next_cycle_pos and next_cycle_pos in valid_neighbors:
            # Can follow cycle directly
            dx = next_cycle_pos[0] - snake_head[0]
            dy = next_cycle_pos[1] - snake_head[1]
            
            # Build path showing cycle continuation
            cycle_path = [snake_head]
            current_idx = head_idx
            for _ in range(min(10, len(cycle))):  # Show next 10 steps of cycle
                current_idx = (current_idx + 1) % len(cycle)
                cycle_path.append(cycle[current_idx])
            
            return (dx, dy), cycle_path
        
        # Next cycle position is blocked, try to find nearest cycle position we can reach
        # Look ahead in the cycle to find the next reachable position
        for lookahead in range(1, min(10, len(cycle))):
            lookahead_idx = (head_idx + lookahead) % len(cycle)
            lookahead_pos = cycle[lookahead_idx]
            
            if lookahead_pos in valid_neighbors:
                # Found a reachable cycle position
                dx = lookahead_pos[0] - snake_head[0]
                dy = lookahead_pos[1] - snake_head[1]
                return (dx, dy), [snake_head, lookahead_pos]
    
    # Not on cycle or can't follow cycle, try to get back to cycle
    # Find the nearest cycle position that's adjacent
    best_cycle_pos = None
    min_dist = float('inf')
    
    for cycle_pos in cycle:
        if cycle_pos in valid_neighbors:
            dist = manhattan_distance(snake_head, cycle_pos)
            if dist < min_dist:
                min_dist = dist
                best_cycle_pos = cycle_pos
    
    if best_cycle_pos:
        # Move to nearest cycle position
        dx = best_cycle_pos[0] - snake_head[0]
        dy = best_cycle_pos[1] - snake_head[1]
        return (dx, dy), [snake_head, best_cycle_pos]
    
    # Fallback: move to any valid neighbor (prefer continuing in current direction)
    # Check if current direction is valid
    dx, dy = current_direction
    next_pos = (snake_head[0] + dx, snake_head[1] + dy)
    if next_pos in valid_neighbors:
        return (dx, dy), [snake_head, next_pos]
    
    # Use first valid neighbor
    next_pos = valid_neighbors[0]
    dx = next_pos[0] - snake_head[0]
    dy = next_pos[1] - snake_head[1]
    return (dx, dy), [snake_head, next_pos]


# Global cycle cache to avoid regenerating
_cycle_cache: Dict[int, List[Tuple[int, int]]] = {}


def get_hamiltonian_cycle(grid_size: int) -> List[Tuple[int, int]]:
    """Get Hamiltonian cycle with caching"""
    if grid_size not in _cycle_cache:
        _cycle_cache[grid_size] = generate_hamiltonian_cycle(grid_size)
    return _cycle_cache[grid_size]


# Wrapper function that uses cached cycle
def get_next_direction_cached(
    snake_head: Tuple[int, int],
    apple: Tuple[int, int],
    snake_body: List[Tuple[int, int]],
    grid_size: int,
    current_direction: Tuple[int, int]
) -> Tuple[Optional[Tuple[int, int]], Optional[List[Tuple[int, int]]]]:
    """Wrapper that uses cached cycle"""
    cycle = get_hamiltonian_cycle(grid_size)
    return get_next_direction(snake_head, apple, snake_body, grid_size, current_direction, cycle)

