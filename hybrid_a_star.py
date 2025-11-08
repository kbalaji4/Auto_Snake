"""
Hybrid A* pathfinding algorithm for Snake game.
Finds optimal path from snake head to apple while avoiding obstacles (snake body).
"""

import heapq
from typing import List, Tuple, Optional, Set

# Optional RL integration
try:
    from rl_agent import get_rl_penalty, get_rl_neighbor_penalty
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    def get_rl_penalty(*args, **kwargs):
        return 0.0
    def get_rl_neighbor_penalty(neighbor_count: int, base_weight: float = 1.5) -> float:
        max_neighbors = 4
        return (max_neighbors - neighbor_count) * base_weight


class Node:
    """Node for A* algorithm"""
    def __init__(self, position: Tuple[int, int], g_cost: float = 0, h_cost: float = 0, parent: Optional['Node'] = None):
        self.position = position
        self.g_cost = g_cost  # Cost from start
        self.h_cost = h_cost  # Heuristic cost to goal
        self.f_cost = g_cost + h_cost  # Total cost
        self.parent = parent
    
    def __lt__(self, other):
        """For priority queue comparison"""
        return self.f_cost < other.f_cost or (self.f_cost == other.f_cost and self.g_cost < other.g_cost)
    
    def __eq__(self, other):
        """For equality comparison"""
        return isinstance(other, Node) and self.position == other.position
    
    def __hash__(self):
        """For set operations"""
        return hash(self.position)


def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """Calculate Manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def distance_to_nearest_obstacle(position: Tuple[int, int], obstacles: Set[Tuple[int, int]], grid_size: int) -> float:
    """
    Calculate the Manhattan distance to the nearest obstacle.
    Returns a large value if no obstacles are nearby (within reasonable search distance).
    
    Args:
        position: Current position (x, y)
        obstacles: Set of obstacle positions
        grid_size: Size of the grid
    
    Returns:
        Distance to nearest obstacle, or a large value if no obstacles nearby
    """
    if not obstacles:
        return grid_size  # No obstacles, return max distance
    
    min_distance = float('inf')
    x, y = position
    
    # Search within a reasonable radius (e.g., up to 5 cells away)
    search_radius = min(5, grid_size)
    
    for obs_x, obs_y in obstacles:
        distance = abs(x - obs_x) + abs(y - obs_y)
        if distance < min_distance:
            min_distance = distance
            if min_distance == 0:  # Adjacent to obstacle
                return 0
    
    # If no obstacle found within search radius, return max search radius
    if min_distance == float('inf'):
        return search_radius
    
    return min_distance


def count_reachable_cells(
    start: Tuple[int, int],
    obstacles: Set[Tuple[int, int]],
    grid_size: int,
    max_search: int = 50
) -> int:
    """
    Count how many cells are reachable from a given position using BFS.
    This helps determine if the snake will be trapped.
    
    Args:
        start: Starting position
        obstacles: Set of obstacle positions
        grid_size: Size of the grid
        max_search: Maximum number of cells to search (for performance)
    
    Returns:
        Number of reachable cells (capped at max_search)
    """
    if start in obstacles:
        return 0
    
    from collections import deque
    
    queue = deque([start])
    visited = {start}
    
    while queue and len(visited) < max_search:
        current = queue.popleft()
        neighbors = get_neighbors(current, grid_size, obstacles)
        
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return len(visited)


def check_reachability_after_path(
    path: List[Tuple[int, int]],
    snake_body: List[Tuple[int, int]],
    apple: Tuple[int, int],
    grid_size: int,
    cache: Optional[dict] = None
) -> float:
    """
    Check how many free cells are reachable after following a path, considering tail movement.
    This helps avoid paths that lead to trapped positions.
    
    Args:
        path: Path to follow (starting from current head)
        snake_body: Current snake body positions (including head)
        apple: Position of apple
        grid_size: Size of the grid
    
    Returns:
        Ratio of free cells that are reachable (0.0 to 1.0)
    """
    if not path or len(path) < 2:
        return 0.0
    
    # Simulate snake moving along the path
    simulated_snake = snake_body.copy()
    
    # Follow the path step by step
    for i in range(1, len(path)):
        next_pos = path[i]
        
        # Move head to next position
        simulated_snake.insert(0, next_pos)
        
        # If we eat the apple, snake grows (tail doesn't move)
        if next_pos != apple:
            # Snake doesn't grow, tail moves forward (remove last element)
            if len(simulated_snake) > 1:
                simulated_snake.pop()
    
    # Now check reachability from the end position
    end_position = simulated_snake[0]
    obstacles_after_path = set(simulated_snake[1:])  # Exclude head
    
    # Count all free cells
    free_cells = []
    for x in range(grid_size):
        for y in range(grid_size):
            pos = (x, y)
            if pos not in obstacles_after_path and pos != end_position:
                free_cells.append(pos)
    
    if len(free_cells) == 0:
        return 0.0
    
    # Count how many free cells are reachable using BFS
    from collections import deque
    queue = deque([end_position])
    visited = {end_position}
    
    while queue:
        current = queue.popleft()
        neighbors = get_neighbors(current, grid_size, obstacles_after_path)
        
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    # Count how many free cells were reached
    reachable_count = sum(1 for cell in free_cells if cell in visited)
    reachability_ratio = reachable_count / len(free_cells)
    
    return reachability_ratio


def check_reachability_to_test_apple(
    position: Tuple[int, int],
    obstacles: Set[Tuple[int, int]],
    grid_size: int,
    num_tests: int = 3,
    cache: Optional[dict] = None
) -> float:
    """
    Check if the snake can reach randomly placed test apples from a given position.
    This helps avoid paths that lead to trapped positions.
    
    Args:
        position: Position to check from
        obstacles: Set of obstacle positions
        grid_size: Size of the grid
        num_tests: Number of random test apples to check
        cache: Optional cache dictionary to store results
    
    Returns:
        Ratio of test apples that can be reached (0.0 to 1.0)
    """
    if position in obstacles:
        return 0.0
    
    # Check cache first (obstacles are constant within a single A* search)
    if cache is not None:
        if position in cache:
            return cache[position]
    
    # Generate random test apple positions
    free_cells = []
    for x in range(grid_size):
        for y in range(grid_size):
            pos = (x, y)
            if pos not in obstacles and pos != position:
                free_cells.append(pos)
    
    if len(free_cells) < num_tests:
        num_tests = len(free_cells)
    
    if num_tests == 0:
        result = 0.0
        if cache is not None:
            cache[position] = result
        return result
    
    import random
    test_apples = random.sample(free_cells, min(num_tests, len(free_cells)))
    
    # Check how many test apples are reachable
    reachable_count = 0
    for test_apple in test_apples:
        # Quick BFS to check if test apple is reachable
        from collections import deque
        queue = deque([position])
        visited = {position}
        found = False
        
        while queue and not found:
            current = queue.popleft()
            if current == test_apple:
                found = True
                break
            
            neighbors = get_neighbors(current, grid_size, obstacles)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    # Limit search to avoid performance issues
                    if len(visited) > 100:
                        break
        
        if found:
            reachable_count += 1
    
    result = reachable_count / num_tests
    # Cache result (obstacles are constant within a single A* search)
    if cache is not None:
        cache[position] = result
    return result


def count_free_neighbors(
    position: Tuple[int, int],
    obstacles: Set[Tuple[int, int]],
    grid_size: int
) -> int:
    """
    Count the number of free neighbors (up, down, left, right) from a position.
    More neighbors = more options = less likely to get trapped.
    
    Args:
        position: Position to check
        obstacles: Set of obstacle positions
        grid_size: Size of the grid
    
    Returns:
        Number of free neighbors (0-4)
    """
    neighbors = get_neighbors(position, grid_size, obstacles)
    return len(neighbors)


def enhanced_heuristic(
    position: Tuple[int, int],
    goal: Tuple[int, int],
    obstacles: Set[Tuple[int, int]],
    grid_size: int,
    reachability_weight: float = 2.0,
    neighbor_weight: float = 1.5,
    cache: Optional[dict] = None,
    snake_body: Optional[List[Tuple[int, int]]] = None,
    action: Optional[Tuple[int, int]] = None,
    rl_weight: float = 1.0
) -> float:
    """
    Enhanced heuristic that combines distance to goal with neighbor count and reachability.
    Prefers paths that lead to positions with more free neighbors (more exploration options).
    
    Args:
        position: Current position (x, y)
        goal: Goal position (x, y)
        obstacles: Set of obstacle positions
        grid_size: Size of the grid
        reachability_weight: Weight for reachability penalty (higher = more avoidance of traps)
        neighbor_weight: Weight for neighbor count bonus (higher = more preference for open spaces)
        cache: Optional cache for reachability checks
    
    Returns:
        Heuristic value (lower is better)
    """
    # Base heuristic: Manhattan distance to goal
    distance_to_goal = manhattan_distance(position, goal)
    
    # Count free neighbors - more neighbors = more options = better
    num_neighbors = count_free_neighbors(position, obstacles, grid_size)
    
    # Use RL-learned neighbor penalty if available, otherwise use fixed weight
    if RL_AVAILABLE:
        neighbor_penalty = get_rl_neighbor_penalty(num_neighbors, base_weight=neighbor_weight)
    else:
        max_neighbors = 4  # Maximum possible neighbors in a grid
        # Convert to penalty: fewer neighbors = higher penalty
        # If we have 4 neighbors, penalty is 0. If we have 0 neighbors, penalty is max
        neighbor_penalty = (max_neighbors - num_neighbors) * neighbor_weight
    
    # Check reachability - can we reach test apples from this position?
    # Lower reachability means higher risk of getting trapped
    # Only check reachability for positions that might be problematic (near obstacles)
    # to save computation
    distance_to_obstacle = distance_to_nearest_obstacle(position, obstacles, grid_size)
    
    # Only do expensive reachability check if we're close to obstacles
    # or if we're far from the goal (might be going into a trap)
    if distance_to_obstacle <= 2 or distance_to_goal > grid_size // 2:
        reachability = check_reachability_to_test_apple(position, obstacles, grid_size, num_tests=3, cache=cache)
        # Convert reachability to penalty: low reachability = high penalty
        trap_penalty = (1.0 - reachability) * reachability_weight
    else:
        # Assume good reachability if we're far from obstacles
        trap_penalty = 0.0
    
    # RL-based penalty (if available and action provided)
    rl_penalty = 0.0
    if RL_AVAILABLE and snake_body is not None and action is not None:
        try:
            rl_penalty = get_rl_penalty(position, goal, snake_body, grid_size, action) * rl_weight
        except:
            pass  # Ignore RL errors
    
    # Combine: distance to goal + penalty for few neighbors + penalty for low reachability + RL penalty
    heuristic = distance_to_goal + neighbor_penalty + trap_penalty + rl_penalty
    
    return heuristic


def get_neighbors(position: Tuple[int, int], grid_size: int, obstacles: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Get valid neighboring positions (up, down, left, right).
    
    Args:
        position: Current position (x, y)
        grid_size: Size of the grid
        obstacles: Set of obstacle positions to avoid
    
    Returns:
        List of valid neighbor positions
    """
    x, y = position
    neighbors = []
    
    # Four directions: up, down, left, right
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    
    for dx, dy in directions:
        new_x, new_y = x + dx, y + dy
        
        # Check boundaries
        if 0 <= new_x < grid_size and 0 <= new_y < grid_size:
            new_pos = (new_x, new_y)
            # Check if not an obstacle
            if new_pos not in obstacles:
                neighbors.append(new_pos)
    
    return neighbors


def reconstruct_path(node: Node) -> List[Tuple[int, int]]:
    """Reconstruct path from goal node to start node"""
    path = []
    current = node
    while current is not None:
        path.append(current.position)
        current = current.parent
    return path[::-1]  # Reverse to get path from start to goal


def hybrid_a_star(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    obstacles: Set[Tuple[int, int]],
    grid_size: int,
    snake_body: Optional[List[Tuple[int, int]]] = None
) -> Tuple[Optional[List[Tuple[int, int]]], Optional[List[Tuple[int, int]]]]:
    """
    Hybrid A* algorithm to find path from start to goal.
    Also tracks the longest path found during exploration.
    
    Args:
        start: Starting position (snake head)
        goal: Goal position (apple)
        obstacles: Set of obstacle positions (snake body cells)
        grid_size: Size of the grid
    
    Returns:
        Tuple of (path_to_goal, longest_path):
        - path_to_goal: List of positions forming the path from start to goal, or None if no path exists
        - longest_path: List of positions forming the longest path found during exploration, or None
    """
    # If start or goal is in obstacles, no path exists
    if start in obstacles or goal in obstacles:
        return None, None
    
    # Cache for reachability checks to avoid redundant calculations
    reachability_cache = {}
    
    # Initialize open and closed sets
    open_set = []
    closed_set: Set[Tuple[int, int]] = set()
    
    # Create start node with enhanced heuristic
    start_node = Node(
        position=start,
        g_cost=0,
        h_cost=enhanced_heuristic(start, goal, obstacles, grid_size, 
                                 cache=reachability_cache, 
                                 snake_body=snake_body)
    )
    
    heapq.heappush(open_set, start_node)
    
    # Dictionary to track best g_cost for each position
    g_costs = {start: 0}
    
    # Track longest path found during exploration
    max_g_cost = 0
    longest_path_node = start_node
    
    while open_set:
        # Get node with lowest f_cost
        current_node = heapq.heappop(open_set)
        
        # Skip if already processed with better cost
        if current_node.position in closed_set:
            continue
        
        # Add to closed set
        closed_set.add(current_node.position)
        
        # Track longest path (node with maximum g_cost)
        if current_node.g_cost > max_g_cost:
            max_g_cost = current_node.g_cost
            longest_path_node = current_node
        
        # Check if goal reached
        if current_node.position == goal:
            path_to_goal = reconstruct_path(current_node)
            longest_path = reconstruct_path(longest_path_node)
            return path_to_goal, longest_path
        
        # Explore neighbors
        neighbors = get_neighbors(current_node.position, grid_size, obstacles)
        
        for neighbor_pos in neighbors:
            if neighbor_pos in closed_set:
                continue
            
            # Calculate costs
            tentative_g_cost = current_node.g_cost + 1  # Each step costs 1
            
            # Check if this is a better path to this neighbor
            if neighbor_pos not in g_costs or tentative_g_cost < g_costs[neighbor_pos]:
                g_costs[neighbor_pos] = tentative_g_cost
                # Calculate action (direction) from current to neighbor
                action = (neighbor_pos[0] - current_node.position[0], 
                         neighbor_pos[1] - current_node.position[1])
                h_cost = enhanced_heuristic(neighbor_pos, goal, obstacles, grid_size, 
                                          cache=reachability_cache,
                                          snake_body=snake_body,
                                          action=action)
                
                neighbor_node = Node(
                    position=neighbor_pos,
                    g_cost=tentative_g_cost,
                    h_cost=h_cost,
                    parent=current_node
                )
                
                heapq.heappush(open_set, neighbor_node)
    
    # No path to goal found, return longest path found during exploration
    if max_g_cost > 0:
        longest_path = reconstruct_path(longest_path_node)
        return None, longest_path
    
    # No path found at all
    return None, None


def validate_path_with_moving_tail(
    path: List[Tuple[int, int]],
    snake_body: List[Tuple[int, int]],
    apple: Tuple[int, int],
    grid_size: int
) -> bool:
    """
    Validate a path considering that the snake's tail moves forward each step.
    As the snake moves along the path, the tail position changes, so obstacles
    that exist now might not exist in future steps.
    
    Args:
        path: List of positions forming the path (starting from current head)
        snake_body: Current snake body positions (including head)
        apple: Position of apple
        grid_size: Size of the grid
    
    Returns:
        True if path is valid considering moving tail, False otherwise
    """
    if not path or len(path) < 2:
        return False
    
    # Simulate the snake moving along the path
    simulated_snake = snake_body.copy()
    
    # Check each step in the path (skip first position as it's the current head)
    for i in range(1, len(path)):
        next_pos = path[i]
        
        # Check boundaries
        if (next_pos[0] < 0 or next_pos[0] >= grid_size or 
            next_pos[1] < 0 or next_pos[1] >= grid_size):
            return False
        
        # Check collision with current snake body (excluding tail which will move)
        # The tail (last element) will move forward, so we exclude it from collision check
        obstacles = set(simulated_snake[:-1]) if len(simulated_snake) > 1 else set()
        
        if next_pos in obstacles:
            return False
        
        # Simulate snake movement: move head to next position
        simulated_snake.insert(0, next_pos)
        
        # If we eat the apple, snake grows (tail doesn't move)
        if next_pos == apple:
            # Snake grows, so we keep the tail
            pass
        else:
            # Snake doesn't grow, tail moves forward (remove last element)
            if len(simulated_snake) > 1:
                simulated_snake.pop()
    
    return True


def get_next_direction(
    snake_head: Tuple[int, int],
    apple: Tuple[int, int],
    snake_body: List[Tuple[int, int]],
    grid_size: int,
    current_direction: Tuple[int, int]
) -> Tuple[Optional[Tuple[int, int]], Optional[List[Tuple[int, int]]]]:
    """
    Get the next direction for the snake to move towards the apple.
    
    Args:
        snake_head: Current position of snake head (x, y)
        apple: Position of apple (x, y)
        snake_body: List of all snake body positions (including head)
        grid_size: Size of the grid
        current_direction: Current direction the snake is moving (for fallback)
    
    Returns:
        Tuple of (next_direction, path):
        - next_direction: Next direction tuple (dx, dy), or None if no valid path exists
        - path: The path being followed (for visualization), or None
    """
    # Convert snake body to set of obstacles (excluding head for pathfinding)
    # Note: We exclude the tail as well since it will move forward each step
    obstacles = set(snake_body[1:-1]) if len(snake_body) > 2 else set(snake_body[1:])
    
    # Find path using Hybrid A* (returns both path to goal and longest path)
    path_to_goal, longest_path = hybrid_a_star(snake_head, apple, obstacles, grid_size, snake_body=snake_body)
    
    # Prioritize reachability over goal-seeking to avoid traps
    # Check reachability for all candidate paths
    best_path = None
    best_reachability = -1.0
    
    # Check path to goal
    if path_to_goal is not None and len(path_to_goal) >= 2:
        # Validate path considering moving tail
        if validate_path_with_moving_tail(path_to_goal, snake_body, apple, grid_size):
            # Check reachability after following this path
            reachability = check_reachability_after_path(path_to_goal, snake_body, apple, grid_size)
            # Only consider path to goal if it maintains very good reachability (>= 0.8)
            # Otherwise, we risk trapping ourselves
            if reachability >= 0.8 and reachability > best_reachability:
                best_path = path_to_goal
                best_reachability = reachability
    
    # Check longest path
    if longest_path is not None and len(longest_path) >= 2:
        # Validate longest path considering moving tail
        if validate_path_with_moving_tail(longest_path, snake_body, apple, grid_size):
            # Check reachability after following this path
            reachability = check_reachability_after_path(longest_path, snake_body, apple, grid_size)
            # Prefer longest path if it has better reachability
            if reachability > best_reachability:
                best_path = longest_path
                best_reachability = reachability
    
    # If we found a good path, use it
    if best_path is not None and len(best_path) >= 2:
        next_pos = best_path[1]
        dx = next_pos[0] - snake_head[0]
        dy = next_pos[1] - snake_head[1]
        return (dx, dy), best_path
    
    # Fallback: Check all immediate neighbors for best reachability
    # This helps when both paths have poor reachability
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
    best_neighbor = None
    best_neighbor_reachability = -1.0
    
    for dx, dy in directions:
        next_pos = (snake_head[0] + dx, snake_head[1] + dy)
        # Check boundaries
        if 0 <= next_pos[0] < grid_size and 0 <= next_pos[1] < grid_size:
            # Check if not an obstacle (excluding tail)
            if next_pos not in obstacles:
                # Create a simple path with just this move
                simple_path = [snake_head, next_pos]
                # Check reachability after this move
                if validate_path_with_moving_tail(simple_path, snake_body, apple, grid_size):
                    reachability = check_reachability_after_path(simple_path, snake_body, apple, grid_size)
                    if reachability > best_neighbor_reachability:
                        best_neighbor = (dx, dy)
                        best_neighbor_reachability = reachability
    
    # Use the neighbor with best reachability
    if best_neighbor is not None:
        return best_neighbor, [snake_head, (snake_head[0] + best_neighbor[0], snake_head[1] + best_neighbor[1])]
    
    # Final fallback: try any safe move that doesn't cause immediate collision
    for dx, dy in directions:
        next_pos = (snake_head[0] + dx, snake_head[1] + dy)
        # Check boundaries
        if 0 <= next_pos[0] < grid_size and 0 <= next_pos[1] < grid_size:
            # Check if not an obstacle
            if next_pos not in obstacles:
                # Return a simple path with just the next position
                return (dx, dy), [snake_head, next_pos]
    
    # If no safe move, return current direction (will likely cause game over)
    return current_direction, None

