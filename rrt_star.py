"""
RRT* (Rapidly-exploring Random Tree Star) pathfinding algorithm for Snake game.
A sampling-based algorithm that explores the space randomly and optimizes paths.
"""

import random
import math
from typing import List, Tuple, Optional, Set


class RRTNode:
    """Node for RRT* tree"""
    def __init__(self, position: Tuple[int, int], parent: Optional['RRTNode'] = None, cost: float = 0.0):
        self.position = position
        self.parent = parent
        self.cost = cost  # Cost from root to this node
        self.children = []
    
    def __eq__(self, other):
        return isinstance(other, RRTNode) and self.position == other.position
    
    def __hash__(self):
        return hash(self.position)


def euclidean_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """Calculate Euclidean distance between two positions"""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """Calculate Manhattan distance between two positions"""
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


def is_valid_position(position: Tuple[int, int], grid_size: int, obstacles: Set[Tuple[int, int]]) -> bool:
    """Check if a position is valid (within bounds and not an obstacle)"""
    x, y = position
    return (0 <= x < grid_size and 0 <= y < grid_size and position not in obstacles)


def steer(from_pos: Tuple[int, int], to_pos: Tuple[int, int], step_size: float = 1.0) -> Tuple[int, int]:
    """
    Steer from one position towards another with a maximum step size.
    For grid-based movement, we move one cell at a time.
    """
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    dist = math.sqrt(dx**2 + dy**2)
    
    if dist <= step_size:
        return to_pos
    
    # Normalize and scale
    new_x = int(from_pos[0] + (dx / dist) * step_size)
    new_y = int(from_pos[1] + (dy / dist) * step_size)
    
    return (new_x, new_y)


def find_nearest_node(tree: List[RRTNode], position: Tuple[int, int]) -> RRTNode:
    """Find the nearest node in the tree to a given position"""
    if not tree:
        return None
    
    nearest = tree[0]
    min_dist = manhattan_distance(nearest.position, position)
    
    for node in tree:
        dist = manhattan_distance(node.position, position)
        if dist < min_dist:
            min_dist = dist
            nearest = node
    
    return nearest


def find_near_nodes(tree: List[RRTNode], position: Tuple[int, int], radius: float) -> List[RRTNode]:
    """Find all nodes within a certain radius of a position"""
    near_nodes = []
    for node in tree:
        if manhattan_distance(node.position, position) <= radius:
            near_nodes.append(node)
    return near_nodes


def reconstruct_path(node: RRTNode) -> List[Tuple[int, int]]:
    """Reconstruct path from root to node"""
    path = []
    current = node
    while current is not None:
        path.append(current.position)
        current = current.parent
    return path[::-1]


def rrt_star(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    obstacles: Set[Tuple[int, int]],
    grid_size: int,
    max_iterations: int = 1000,
    goal_sample_rate: float = 0.1,
    step_size: float = 1.0,
    search_radius: float = 5.0
) -> Tuple[Optional[List[Tuple[int, int]]], Optional[List[Tuple[int, int]]]]:
    """
    RRT* algorithm to find path from start to goal.
    
    Args:
        start: Starting position (snake head)
        goal: Goal position (apple)
        obstacles: Set of obstacle positions (snake body cells)
        grid_size: Size of the grid
        max_iterations: Maximum number of iterations
        goal_sample_rate: Probability of sampling the goal (0.0 to 1.0)
        step_size: Step size for steering (1.0 for grid-based)
        search_radius: Radius for finding near nodes for rewiring
    
    Returns:
        Tuple of (path_to_goal, longest_path):
        - path_to_goal: Path from start to goal, or None if not found
        - longest_path: Longest path found during exploration, or None
    """
    if start in obstacles or goal in obstacles:
        return None, None
    
    # Initialize tree with start node
    root = RRTNode(start, None, 0.0)
    tree = [root]
    
    path_to_goal = None
    longest_path_node = root
    max_cost = 0.0
    
    # Get all free cells for random sampling
    free_cells = []
    for x in range(grid_size):
        for y in range(grid_size):
            pos = (x, y)
            if pos not in obstacles:
                free_cells.append(pos)
    
    if not free_cells:
        return None, None
    
    for iteration in range(max_iterations):
        # Sample random position (with goal bias)
        if random.random() < goal_sample_rate:
            random_pos = goal
        else:
            random_pos = random.choice(free_cells)
        
        # Find nearest node
        nearest = find_nearest_node(tree, random_pos)
        if nearest is None:
            continue
        
        # Steer towards random position
        new_pos = steer(nearest.position, random_pos, step_size)
        
        # Check if new position is valid
        if not is_valid_position(new_pos, grid_size, obstacles):
            continue
        
        # Skip if too close to nearest (already explored)
        if manhattan_distance(nearest.position, new_pos) < 0.5:
            continue
        
        # Find near nodes for potential rewiring
        near_nodes = find_near_nodes(tree, new_pos, search_radius)
        
        # Find best parent (node with minimum cost to reach new_pos)
        best_parent = nearest
        best_cost = nearest.cost + manhattan_distance(nearest.position, new_pos)
        
        for near_node in near_nodes:
            # Check if path from near_node to new_pos is collision-free
            # For grid-based, we assume direct path is valid if both positions are valid
            cost = near_node.cost + manhattan_distance(near_node.position, new_pos)
            if cost < best_cost:
                best_parent = near_node
                best_cost = cost
        
        # Create new node
        new_node = RRTNode(new_pos, best_parent, best_cost)
        best_parent.children.append(new_node)
        tree.append(new_node)
        
        # Rewire: check if we can improve paths to near nodes
        for near_node in near_nodes:
            if near_node == best_parent:
                continue
            
            # Check if path through new_node is better
            new_cost = new_node.cost + manhattan_distance(new_node.position, near_node.position)
            if new_cost < near_node.cost:
                # Rewire: change parent of near_node
                if near_node.parent:
                    # Safely remove from old parent's children list
                    if near_node in near_node.parent.children:
                        near_node.parent.children.remove(near_node)
                near_node.parent = new_node
                near_node.cost = new_cost
                # Only add if not already a child (shouldn't happen, but safety check)
                if near_node not in new_node.children:
                    new_node.children.append(near_node)
                
                # Update costs of all descendants
                def update_descendants(node):
                    for child in node.children:
                        child.cost = node.cost + manhattan_distance(node.position, child.position)
                        update_descendants(child)
                update_descendants(near_node)
        
        # Check if we reached the goal
        if path_to_goal is None and manhattan_distance(new_pos, goal) < 1.5:
            # Check if we can actually reach goal from here
            if is_valid_position(goal, grid_size, obstacles):
                goal_node = RRTNode(goal, new_node, new_node.cost + manhattan_distance(new_pos, goal))
                tree.append(goal_node)
                path_to_goal = reconstruct_path(goal_node)
        
        # Track longest path
        if new_node.cost > max_cost:
            max_cost = new_node.cost
            longest_path_node = new_node
    
    # Reconstruct longest path
    longest_path = reconstruct_path(longest_path_node) if longest_path_node else None
    
    return path_to_goal, longest_path


def get_next_direction(
    snake_head: Tuple[int, int],
    apple: Tuple[int, int],
    snake_body: List[Tuple[int, int]],
    grid_size: int,
    current_direction: Tuple[int, int]
) -> Tuple[Optional[Tuple[int, int]], Optional[List[Tuple[int, int]]]]:
    """
    Get the next direction for the snake using RRT*.
    
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
    # Convert snake body to set of obstacles (excluding head and tail)
    obstacles = set(snake_body[1:-1]) if len(snake_body) > 2 else set(snake_body[1:])
    
    # Find path using RRT*
    path_to_goal, longest_path = rrt_star(
        snake_head, 
        apple, 
        obstacles, 
        grid_size,
        max_iterations=500,  # Adjust based on performance needs
        goal_sample_rate=0.2
    )
    
    # Use path to goal if available
    if path_to_goal is not None and len(path_to_goal) >= 2:
        next_pos = path_to_goal[1]
        dx = next_pos[0] - snake_head[0]
        dy = next_pos[1] - snake_head[1]
        return (dx, dy), path_to_goal
    
    # Use longest path if available
    if longest_path is not None and len(longest_path) >= 2:
        next_pos = longest_path[1]
        dx = next_pos[0] - snake_head[0]
        dy = next_pos[1] - snake_head[1]
        return (dx, dy), longest_path
    
    # Fallback: try any safe move
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    for dx, dy in directions:
        next_pos = (snake_head[0] + dx, snake_head[1] + dy)
        if 0 <= next_pos[0] < grid_size and 0 <= next_pos[1] < grid_size:
            if next_pos not in obstacles:
                return (dx, dy), [snake_head, next_pos]
    
    return current_direction, None

