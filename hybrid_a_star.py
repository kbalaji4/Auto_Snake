"""
Hybrid A* pathfinding algorithm for Snake game.
Finds optimal path from snake head to apple while avoiding obstacles (snake body).
"""

import heapq
from typing import List, Tuple, Optional, Set


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


def enhanced_heuristic(
    position: Tuple[int, int],
    goal: Tuple[int, int],
    obstacles: Set[Tuple[int, int]],
    grid_size: int,
    obstacle_penalty_weight: float = 0.5
) -> float:
    """
    Enhanced heuristic that combines distance to goal with obstacle avoidance.
    Prefers paths that stay away from obstacles.
    
    Args:
        position: Current position (x, y)
        goal: Goal position (x, y)
        obstacles: Set of obstacle positions
        grid_size: Size of the grid
        obstacle_penalty_weight: Weight for obstacle penalty (higher = more avoidance)
    
    Returns:
        Heuristic value (lower is better)
    """
    # Base heuristic: Manhattan distance to goal
    distance_to_goal = manhattan_distance(position, goal)
    
    # Obstacle penalty: prefer positions farther from obstacles
    distance_to_obstacle = distance_to_nearest_obstacle(position, obstacles, grid_size)
    
    # Convert distance to obstacle into a penalty
    # Closer to obstacle = higher penalty
    # We want to maximize distance_to_obstacle, so we subtract it from a max value
    max_obstacle_distance = min(5, grid_size)  # Maximum reasonable distance to check
    obstacle_penalty = max_obstacle_distance - distance_to_obstacle
    
    # Combine: distance to goal + penalty for being near obstacles
    # The penalty is weighted so it doesn't override the goal-seeking behavior
    heuristic = distance_to_goal + (obstacle_penalty * obstacle_penalty_weight)
    
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
    grid_size: int
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
    
    # Initialize open and closed sets
    open_set = []
    closed_set: Set[Tuple[int, int]] = set()
    
    # Create start node with enhanced heuristic
    start_node = Node(
        position=start,
        g_cost=0,
        h_cost=enhanced_heuristic(start, goal, obstacles, grid_size)
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
                h_cost = enhanced_heuristic(neighbor_pos, goal, obstacles, grid_size)
                
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
    path_to_goal, longest_path = hybrid_a_star(snake_head, apple, obstacles, grid_size)
    
    # Validate and use path to goal if available
    if path_to_goal is not None and len(path_to_goal) >= 2:
        # Validate path considering moving tail
        if validate_path_with_moving_tail(path_to_goal, snake_body, apple, grid_size):
            next_pos = path_to_goal[1]  # path_to_goal[0] is current head, path_to_goal[1] is next position
            dx = next_pos[0] - snake_head[0]
            dy = next_pos[1] - snake_head[1]
            return (dx, dy), path_to_goal
    
    # No valid path to apple found, try longest path
    if longest_path is not None and len(longest_path) >= 2:
        # Validate longest path considering moving tail
        if validate_path_with_moving_tail(longest_path, snake_body, apple, grid_size):
            next_pos = longest_path[1]  # longest_path[0] is current head, longest_path[1] is next position
            dx = next_pos[0] - snake_head[0]
            dy = next_pos[1] - snake_head[1]
            return (dx, dy), longest_path
    
    # Fallback: try to find any safe move that doesn't cause immediate collision
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
    
    # Try to move in a direction that doesn't hit obstacles
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

