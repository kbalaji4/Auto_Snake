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
) -> Optional[List[Tuple[int, int]]]:
    """
    Hybrid A* algorithm to find path from start to goal.
    
    Args:
        start: Starting position (snake head)
        goal: Goal position (apple)
        obstacles: Set of obstacle positions (snake body cells)
        grid_size: Size of the grid
    
    Returns:
        List of positions forming the path from start to goal, or None if no path exists
    """
    # If start or goal is in obstacles, no path exists
    if start in obstacles or goal in obstacles:
        return None
    
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
    
    while open_set:
        # Get node with lowest f_cost
        current_node = heapq.heappop(open_set)
        
        # Skip if already processed with better cost
        if current_node.position in closed_set:
            continue
        
        # Add to closed set
        closed_set.add(current_node.position)
        
        # Check if goal reached
        if current_node.position == goal:
            return reconstruct_path(current_node)
        
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
    
    # No path found
    return None


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
    
    # Find path using Hybrid A*
    path = hybrid_a_star(snake_head, apple, obstacles, grid_size)
    
    if path is None or len(path) < 2:
        # No path found, try to find a safe move that doesn't cause immediate collision
        # This is a fallback strategy
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
    
    # Path found, return direction to first step
    next_pos = path[1]  # path[0] is current head, path[1] is next position
    dx = next_pos[0] - snake_head[0]
    dy = next_pos[1] - snake_head[1]
    
    return (dx, dy)

