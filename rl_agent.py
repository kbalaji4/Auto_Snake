"""
Reinforcement Learning agent to help Hybrid A* avoid traps.
Uses Q-learning to learn which positions/actions lead to game over.
"""

import pickle
import os
from typing import Dict, Tuple, Optional, Set, List
from collections import defaultdict
import random
import threading


class RLAgent:
    """
    Q-learning agent that learns to avoid traps.
    Learns from game outcomes to adjust path preferences.
    """
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9, 
                 epsilon: float = 0.1, q_table_file: str = "q_table.pkl"):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table_file = q_table_file
        
        # Q-table: maps (state, action) -> Q-value
        # State: tuple of (head_pos, apple_pos, snake_length, free_space_ratio)
        # Action: direction tuple (dx, dy)
        self.q_table: Dict[Tuple, float] = defaultdict(float)
        
        # Track current episode for learning
        self.current_episode = []
        self.games_played = 0
        
        # Track neighbor count statistics: maps neighbor_count -> (death_count, total_count)
        # This helps learn which neighbor counts lead to death
        self.neighbor_stats: Dict[int, Tuple[int, int]] = defaultdict(lambda: (0, 0))
        
        # Thread safety lock for concurrent access
        self.lock = threading.Lock()
        
        self.load_q_table()
    
    def load_q_table(self):
        """Load Q-table from file if it exists"""
        if os.path.exists(self.q_table_file):
            try:
                with open(self.q_table_file, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, dict):
                        # Check if it's the new format with metadata
                        if 'q_table' in data and 'games_played' in data:
                            self.q_table = defaultdict(float, data['q_table'])
                            self.games_played = data.get('games_played', 0)
                            self.neighbor_stats = defaultdict(lambda: (0, 0), data.get('neighbor_stats', {}))
                        else:
                            # Old format - just Q-table
                            self.q_table = defaultdict(float, data)
                            self.games_played = len(self.q_table) // 10  # Rough estimate
                            self.neighbor_stats = defaultdict(lambda: (0, 0))
                    else:
                        self.q_table = defaultdict(float)
            except:
                self.q_table = defaultdict(float)
    
    def save_q_table(self):
        """Save Q-table to file"""
        try:
            with open(self.q_table_file, 'wb') as f:
                # Save with metadata
                data = {
                    'q_table': dict(self.q_table),
                    'games_played': self.games_played,
                    'neighbor_stats': dict(self.neighbor_stats)
                }
                pickle.dump(data, f)
        except:
            pass
    
    def get_state_key(self, snake_head: Tuple[int, int], apple: Tuple[int, int],
                     snake_body: List[Tuple[int, int]], grid_size: int) -> Tuple:
        """
        Create a state representation for Q-learning.
        Uses a simplified state space to make learning feasible.
        """
        # Calculate free space ratio (how much of grid is free)
        obstacles = set(snake_body)
        free_cells = grid_size * grid_size - len(obstacles)
        free_ratio = free_cells / (grid_size * grid_size)
        
        # Discretize positions to reduce state space
        # Use relative position to apple
        dx_to_apple = apple[0] - snake_head[0]
        dy_to_apple = apple[1] - snake_head[1]
        
        # Discretize free ratio
        free_ratio_bucket = int(free_ratio * 10)  # 0-10 buckets
        
        # Snake length bucket
        length_bucket = min(len(snake_body) // 5, 10)  # 0-10 buckets
        
        # Create state key
        state = (
            dx_to_apple,  # Relative x to apple
            dy_to_apple,  # Relative y to apple
            free_ratio_bucket,
            length_bucket
        )
        
        return state
    
    def count_neighbors(self, position: Tuple[int, int], obstacles: Set[Tuple[int, int]], 
                       grid_size: int) -> int:
        """Count free neighbors for a position"""
        x, y = position
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                if (nx, ny) not in obstacles:
                    neighbors.append((nx, ny))
        return len(neighbors)
    
    def get_action_key(self, direction: Tuple[int, int]) -> Tuple[int, int]:
        """Normalize action to key"""
        return direction
    
    def get_q_value(self, state: Tuple, action: Tuple[int, int]) -> float:
        """Get Q-value for state-action pair (thread-safe)"""
        action_key = self.get_action_key(action)
        with self.lock:
            return self.q_table[(state, action_key)]
    
    def update_q_value(self, state: Tuple, action: Tuple[int, int], 
                      reward: float, next_state: Optional[Tuple] = None):
        """Update Q-value using Q-learning update rule (thread-safe)"""
        action_key = self.get_action_key(action)
        state_action = (state, action_key)
        
        with self.lock:
            current_q = self.q_table[state_action]
        
        if next_state is not None:
            # Q-learning update: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
            max_next_q = max([
                self.get_q_value(next_state, (0, -1)),
                self.get_q_value(next_state, (0, 1)),
                self.get_q_value(next_state, (-1, 0)),
                self.get_q_value(next_state, (1, 0))
            ], default=0.0)
            
            target_q = reward + self.discount_factor * max_next_q
        else:
            # Terminal state
            target_q = reward
        
        # Update Q-value (thread-safe)
        with self.lock:
            self.q_table[state_action] = current_q + self.learning_rate * (target_q - current_q)
    
    def get_reward(self, snake_head: Tuple[int, int], apple: Tuple[int, int],
                  snake_body: List[Tuple[int, int]], grid_size: int,
                  game_over: bool, ate_apple: bool, fast_mode: bool = False) -> float:
        """
        Calculate reward for current state.
        Positive for good outcomes, negative for traps.
        Now includes reachability-based rewards.
        
        Args:
            fast_mode: If True, skip expensive reachability checks for faster training
        """
        if game_over:
            return -100.0  # Large negative reward for game over
        
        if ate_apple:
            return 10.0  # Positive reward for eating apple
        
        # Reward based on distance to apple (closer is better)
        distance = abs(apple[0] - snake_head[0]) + abs(apple[1] - snake_head[1])
        distance_reward = -distance * 0.1
        
        # Reward for having more free space (less trapped)
        obstacles = set(snake_body)
        free_cells = grid_size * grid_size - len(obstacles)
        free_ratio = free_cells / (grid_size * grid_size)
        space_reward = free_ratio * 5.0
        
        # Reward based on reachability - how many cells can be reached from current position
        # Higher reachability = less trapped = better
        # Skip in fast_mode for speed (use neighbor count as proxy)
        if fast_mode:
            # Fast approximation: use neighbor count as proxy for reachability
            neighbor_count = self.count_neighbors(snake_head, obstacles, grid_size)
            max_neighbors = 4
            # Approximate reachability based on neighbors (more neighbors = better reachability)
            reachability_ratio = neighbor_count / max_neighbors
            reachability_reward = reachability_ratio * 5.0  # Reduced reward in fast mode
        else:
            try:
                from hybrid_a_star import count_reachable_cells
                reachable_cells = count_reachable_cells(snake_head, obstacles, grid_size, max_search=50)
                total_free = free_cells
                if total_free > 0:
                    reachability_ratio = reachable_cells / total_free
                    # Reward for high reachability (0.0 to 1.0, scaled to meaningful reward)
                    # 100% reachable = +10.0 reward, 0% reachable = 0.0 reward
                    reachability_reward = reachability_ratio * 10.0
                else:
                    reachability_reward = 0.0
            except:
                # Fallback if reachability check fails
                reachability_reward = 0.0
        
        return distance_reward + space_reward + reachability_reward
    
    def get_action_penalty(self, state: Tuple, action: Tuple[int, int]) -> float:
        """
        Get penalty/bonus for an action based on learned Q-values.
        Returns a value to add to heuristic (negative = penalty, positive = bonus).
        """
        q_value = self.get_q_value(state, action)
        # Convert Q-value to penalty (negative Q = high penalty)
        # Scale it appropriately
        penalty = -q_value * 0.5  # Scale factor
        return penalty
    
    def get_neighbor_penalty(self, neighbor_count: int, base_weight: float = 1.5) -> float:
        """
        Get learned penalty for a neighbor count based on death statistics.
        Returns a penalty value to use in heuristic (higher = more dangerous).
        (thread-safe read)
        
        Args:
            neighbor_count: Number of free neighbors (0-4)
            base_weight: Base weight to use if no statistics available
        
        Returns:
            Penalty value (higher for more dangerous neighbor counts)
        """
        with self.lock:
            if neighbor_count not in self.neighbor_stats:
                # No data yet, use default penalty
                max_neighbors = 4
                return (max_neighbors - neighbor_count) * base_weight
            
            death_count, total_count = self.neighbor_stats[neighbor_count]
        
        if total_count == 0:
            # No data, use default
            max_neighbors = 4
            return (max_neighbors - neighbor_count) * base_weight
        
        # Calculate death probability for this neighbor count
        death_probability = death_count / total_count
        
        # Convert to penalty: higher death probability = higher penalty
        # Scale based on how dangerous this neighbor count is
        # If death_probability is 0.5, we want penalty similar to base_weight
        # If death_probability is 1.0, we want much higher penalty
        # If death_probability is 0.0, we want lower penalty
        
        # Adaptive weight: base_weight * (1 + death_probability * 2)
        # This means:
        # - 0% death rate -> base_weight (normal penalty)
        # - 50% death rate -> base_weight * 2 (double penalty)
        # - 100% death rate -> base_weight * 3 (triple penalty)
        adaptive_weight = base_weight * (1.0 + death_probability * 2.0)
        
        # Apply penalty based on how many neighbors are missing
        max_neighbors = 4
        penalty = (max_neighbors - neighbor_count) * adaptive_weight
        
        return penalty
    
    def record_step(self, snake_head: Tuple[int, int], apple: Tuple[int, int],
                   snake_body: List[Tuple[int, int]], grid_size: int,
                   action: Tuple[int, int], reward: float):
        """Record a step in the current episode (thread-safe)"""
        state = self.get_state_key(snake_head, apple, snake_body, grid_size)
        # Also track neighbor count for this position
        obstacles = set(snake_body)
        neighbor_count = self.count_neighbors(snake_head, obstacles, grid_size)
        with self.lock:
            self.current_episode.append((state, action, reward, neighbor_count))
    
    def learn_from_episode(self, final_reward: float, game_over: bool = False, final_score: int = 0):
        """
        Learn from the completed episode using Q-learning.
        Updates Q-values based on episode experience.
        Also updates neighbor count statistics.
        
        Args:
            final_reward: Final reward for the episode
            game_over: Whether the game ended in game over
            final_score: Final score achieved (for display purposes)
        """
        if not self.current_episode:
            return
        
        # Update neighbor statistics based on episode outcome (thread-safe)
        with self.lock:
            episode_copy = list(self.current_episode)  # Copy to avoid holding lock during processing
        
        if game_over:
            # If game over, all neighbor counts in episode contributed to death
            for step in episode_copy:
                if len(step) >= 4:  # Has neighbor_count
                    _, _, _, neighbor_count = step
                    with self.lock:
                        death_count, total_count = self.neighbor_stats[neighbor_count]
                        self.neighbor_stats[neighbor_count] = (death_count + 1, total_count + 1)
        else:
            # If survived, these neighbor counts didn't lead to death
            for step in episode_copy:
                if len(step) >= 4:  # Has neighbor_count
                    _, _, _, neighbor_count = step
                    with self.lock:
                        death_count, total_count = self.neighbor_stats[neighbor_count]
                        self.neighbor_stats[neighbor_count] = (death_count, total_count + 1)
        
        # Update Q-values backwards through episode
        for i in range(len(episode_copy) - 1, -1, -1):
            step = episode_copy[i]
            if len(step) >= 4:
                state, action, reward, _ = step
            else:
                state, action, reward = step
            
            if i == len(episode_copy) - 1:
                # Last step - use final reward
                self.update_q_value(state, action, final_reward, None)
            else:
                # Use next state
                next_step = episode_copy[i + 1]
                if len(next_step) >= 4:
                    next_state, _, _, _ = next_step
                else:
                    next_state, _, _ = next_step
                self.update_q_value(state, action, reward, next_state)
        
        # Clear episode (thread-safe)
        with self.lock:
            self.current_episode = []
            self.games_played += 1
        
        # Print learning progress
        print(f"RL Learning: Game #{self.games_played} | Score: {final_score} | Reward: {final_reward:.1f} | Q-table: {len(self.q_table)}")
        
        # Print neighbor statistics every 10 games
        if self.games_played % 10 == 0 and self.neighbor_stats:
            print("Neighbor Statistics (death rate):")
            for neighbor_count in sorted(self.neighbor_stats.keys()):
                death_count, total_count = self.neighbor_stats[neighbor_count]
                if total_count > 0:
                    death_rate = death_count / total_count
                    print(f"  {neighbor_count} neighbors: {death_rate:.1%} ({death_count}/{total_count})")
        
        # Periodically save Q-table
        if random.random() < 0.1:  # 10% chance to save
            self.save_q_table()
    
    def get_stats(self) -> Dict[str, any]:
        """Get statistics about the RL agent (thread-safe)"""
        with self.lock:
            # Calculate neighbor statistics summary
            neighbor_summary = {}
            for neighbor_count in range(5):  # 0-4 neighbors
                if neighbor_count in self.neighbor_stats:
                    death_count, total_count = self.neighbor_stats[neighbor_count]
                    if total_count > 0:
                        death_rate = death_count / total_count
                        neighbor_summary[f'{neighbor_count}_neighbors'] = {
                            'death_rate': death_rate,
                            'total': total_count
                        }
            
            return {
                'q_table_size': len(self.q_table),
                'games_played': self.games_played,
                'is_active': True,
                'neighbor_stats': neighbor_summary
            }


# Global RL agent instance
_rl_agent: Optional[RLAgent] = None
_agent_lock = threading.Lock()


def get_rl_agent() -> RLAgent:
    """Get or create global RL agent instance (thread-safe)"""
    global _rl_agent
    if _rl_agent is None:
        with _agent_lock:
            # Double-check pattern to avoid race condition
            if _rl_agent is None:
                _rl_agent = RLAgent()
    return _rl_agent


def get_rl_penalty(snake_head: Tuple[int, int], apple: Tuple[int, int],
                  snake_body: List[Tuple[int, int]], grid_size: int,
                  action: Tuple[int, int]) -> float:
    """
    Get RL-based penalty for an action.
    Can be added to heuristic to guide pathfinding away from traps.
    """
    agent = get_rl_agent()
    state = agent.get_state_key(snake_head, apple, snake_body, grid_size)
    return agent.get_action_penalty(state, action)


def record_game_step(snake_head: Tuple[int, int], apple: Tuple[int, int],
                    snake_body: List[Tuple[int, int]], grid_size: int,
                    action: Tuple[int, int], reward: float):
    """Record a game step for RL learning"""
    agent = get_rl_agent()
    agent.record_step(snake_head, apple, snake_body, grid_size, action, reward)


def load_high_score() -> int:
    """Load high score from file"""
    high_score_file = "high_score.txt"
    if os.path.exists(high_score_file):
        try:
            with open(high_score_file, 'r') as f:
                return int(f.read().strip())
        except:
            return 0
    return 0


def learn_from_game(game_over: bool, final_score: int):
    """
    Learn from completed game.
    Rewards are normalized based on current high score to encourage improvement:
    - Base penalty for game over: -100.0
    - Score bonus is calculated relative to high score
    - Only scores >= 80% of high score get positive final rewards
    - Scores below 80% of high score remain negative
    
    Reward formula:
    - If high_score == 0: use score * 10.0 (no normalization yet)
    - If score_ratio < 0.8: score_bonus = score_ratio * 125.0 (0 to 100)
    - If score_ratio >= 0.8: score_bonus = (score_ratio - 0.8) * 500.0 + 100.0
    - Bonus: if score >= high_score, add extra 100.0
    
    Examples (assuming high_score = 64):
    - Score 0 (0%): -100.0 + 0.0 = -100.0 (worst)
    - Score 32 (50%): -100.0 + 40.0 = -60.0 (negative)
    - Score 51 (80%): -100.0 + 100.0 = 0.0 (break even)
    - Score 64 (100%): -100.0 + 200.0 + 100.0 = +200.0 (matches record)
    - Score 80 (125%): -100.0 + 325.0 + 100.0 = +325.0 (new record!)
    """
    agent = get_rl_agent()
    
    # Load current high score
    high_score = load_high_score()
    
    # Base penalty for game over (always happens when game ends)
    base_penalty = -100.0
    
    # Calculate score bonus relative to high score
    if high_score == 0:
        # No high score yet, use simple multiplier
        score_bonus = final_score * 10.0
    else:
        # Normalize based on high score
        # Score as percentage of high score
        score_ratio = final_score / high_score
        
        # Only give positive bonus if score is at least 80% of high score
        # Below 80% should still be negative overall
        if score_ratio >= 0.8:
            # Good performance: scale from 80% to 100%+
            # At 80%: score_bonus = 100.0 (breaks even: -100 + 100 = 0)
            # At 100%: score_bonus = 200.0 (good: -100 + 200 = +100)
            # Above 100%: continues scaling
            score_bonus = (score_ratio - 0.8) * 500.0 + 100.0  # Scale from 100 to 200+ at 100%
            
            # Extra bonus for beating or matching the high score
            if final_score >= high_score:
                score_bonus += 100.0  # Significant bonus for new records
        else:
            # Poor performance: below 80% of high score
            # Scale from 0% to 80%: penalty reduction from -100 to 0
            # At 0%: score_bonus = 0 (full penalty: -100)
            # At 80%: score_bonus = 100.0 (breaks even: -100 + 100 = 0)
            score_bonus = score_ratio * 125.0  # Scale from 0 to 100
    
    # Final reward combines both: higher scores (relative to high score) get better rewards
    final_reward = base_penalty + score_bonus
    
    agent.learn_from_episode(final_reward, game_over=game_over, final_score=final_score)
    agent.save_q_table()  # Save after each game


def get_rl_neighbor_penalty(neighbor_count: int, base_weight: float = 1.5) -> float:
    """
    Get RL-learned penalty for a neighbor count.
    This penalty is based on learned statistics about which neighbor counts lead to death.
    """
    try:
        agent = get_rl_agent()
        return agent.get_neighbor_penalty(neighbor_count, base_weight)
    except:
        # Fallback to default if RL not available
        max_neighbors = 4
        return (max_neighbors - neighbor_count) * base_weight


def get_rl_stats() -> Optional[Dict[str, any]]:
    """Get RL agent statistics"""
    try:
        agent = get_rl_agent()
        return agent.get_stats()
    except:
        return None

