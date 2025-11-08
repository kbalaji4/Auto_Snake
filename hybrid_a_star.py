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
    
    # Create start node
    start_node = Node(
        position=start,
        g_cost=0,
        h_cost=manhattan_distance(start, goal)
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
                h_cost = manhattan_distance(neighbor_pos, goal)
                
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


def get_next_direction(
    snake_head: Tuple[int, int],
    apple: Tuple[int, int],
    snake_body: List[Tuple[int, int]],
    grid_size: int,
    current_direction: Tuple[int, int]
) -> Optional[Tuple[int, int]]:
    """
    Get the next direction for the snake to move towards the apple.
    
    Args:
        snake_head: Current position of snake head (x, y)
        apple: Position of apple (x, y)
        snake_body: List of all snake body positions (including head)
        grid_size: Size of the grid
        current_direction: Current direction the snake is moving (for fallback)
    
    Returns:
        Next direction tuple (dx, dy), or None if no valid path exists
    """
    # Convert snake body to set of obstacles (excluding head for pathfinding)
    obstacles = set(snake_body[1:])  # Exclude head, include rest of body
    
    # Find path using Hybrid A* (returns both path to goal and longest path)
    path_to_goal, longest_path = hybrid_a_star(snake_head, apple, obstacles, grid_size)
    
    # Use path to goal if available
    if path_to_goal is not None and len(path_to_goal) >= 2:
        next_pos = path_to_goal[1]  # path_to_goal[0] is current head, path_to_goal[1] is next position
        dx = next_pos[0] - snake_head[0]
        dy = next_pos[1] - snake_head[1]
        return (dx, dy)
    
    # No path to apple found, use the longest path tracked during A* search
    if longest_path is not None and len(longest_path) >= 2:
        next_pos = longest_path[1]  # longest_path[0] is current head, longest_path[1] is next position
        dx = next_pos[0] - snake_head[0]
        dy = next_pos[1] - snake_head[1]
        return (dx, dy)
    
    # Fallback: try to find any safe move that doesn't cause immediate collision
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
    
    # Try to move in a direction that doesn't hit obstacles
    for dx, dy in directions:
        next_pos = (snake_head[0] + dx, snake_head[1] + dy)
        # Check boundaries
        if 0 <= next_pos[0] < grid_size and 0 <= next_pos[1] < grid_size:
            # Check if not an obstacle
            if next_pos not in obstacles:
                return (dx, dy)
    
    # If no safe move, return current direction (will likely cause game over)
    return current_direction

