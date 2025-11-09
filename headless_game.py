"""
Headless Snake game runner for concurrent RL training.
No Pygame rendering - just game logic for fast training.
"""

import random
import os
from hybrid_a_star import get_next_direction as hybrid_a_star_get_next_direction

# Optional RL integration
try:
    from rl_agent import record_game_step, learn_from_game
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    def record_game_step(*args, **kwargs):
        pass
    def learn_from_game(*args, **kwargs):
        pass

# Constants (must match snake_game.py)
GRID_SIZE = 15
HIGH_SCORE_FILE = "high_score.txt"

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# Algorithm types
ALGORITHM_HYBRID_A_STAR = 0


class HeadlessSnakeGame:
    """Headless version of Snake game for fast training without rendering"""
    
    def __init__(self):
        self.high_score = self.load_high_score()
        self.reset_game()
        self.algorithm = ALGORITHM_HYBRID_A_STAR
        self.ate_apple_this_step = False
    
    def load_high_score(self):
        """Load high score from file"""
        if os.path.exists(HIGH_SCORE_FILE):
            try:
                with open(HIGH_SCORE_FILE, 'r') as f:
                    return int(f.read().strip())
            except:
                return 0
        return 0
    
    def save_high_score(self):
        """Save high score to file (thread-safe with file locking on Unix)"""
        try:
            # Try to use file locking on Unix systems
            try:
                import fcntl
                lock_file = HIGH_SCORE_FILE + ".lock"
                # Try to acquire lock (non-blocking)
                try:
                    with open(lock_file, 'w') as lock:
                        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        # Read current high score to ensure we don't overwrite a higher one
                        current_high = self.load_high_score()
                        if self.high_score > current_high:
                            with open(HIGH_SCORE_FILE, 'w') as f:
                                f.write(str(self.high_score))
                        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                    return
                except (IOError, OSError):
                    # Lock is held by another thread, skip this save
                    # The other thread will save it
                    pass
            except ImportError:
                # fcntl not available (Windows), use fallback
                pass
        except:
            pass
        
        # Fallback: simple write with double-check (may have race conditions but better than nothing)
        try:
            current_high = self.load_high_score()
            if self.high_score > current_high:
                with open(HIGH_SCORE_FILE, 'w') as f:
                    f.write(str(self.high_score))
        except:
            pass
    
    def reset_game(self):
        """Reset game to initial state"""
        # Reload high score in case another thread updated it
        self.high_score = self.load_high_score()
        center = GRID_SIZE // 2
        self.snake = [(center, center), (center - 1, center)]
        self.direction = RIGHT
        self.next_direction = RIGHT
        self.apple = self.generate_apple()
        self.score = 0
        self.game_over = False
        self.steps = 0
        self.max_steps = GRID_SIZE * GRID_SIZE * 10  # Prevent infinite games
    
    def generate_apple(self):
        """Generate apple at a random position that's not occupied by the snake"""
        while True:
            apple = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            if apple not in self.snake:
                return apple
    
    def update(self):
        """Update game state - returns True if game continues, False if game over"""
        if self.game_over:
            return False
        
        self.steps += 1
        if self.steps > self.max_steps:
            # Game too long, consider it a success but end it
            self.game_over = True
            return False
        
        # Track if we ate apple this step (for RL)
        self.ate_apple_this_step = False
        
        # Auto mode: use Hybrid A* to determine next direction
        next_dir, _ = hybrid_a_star_get_next_direction(
            snake_head=self.snake[0],
            apple=self.apple,
            snake_body=self.snake,
            grid_size=GRID_SIZE,
            current_direction=self.direction
        )
        
        if next_dir:
            # Only update if the new direction is valid (not opposite of current)
            if (next_dir[0] * -1, next_dir[1] * -1) != self.direction:
                self.next_direction = next_dir
        
        # Update direction
        self.direction = self.next_direction
        
        # Record step for RL learning (BEFORE moving)
        if RL_AVAILABLE:
            try:
                from rl_agent import get_rl_agent
                agent = get_rl_agent()
                # Calculate reward for current state (before move)
                # Use fast_mode=True to skip expensive reachability checks during training
                reward = agent.get_reward(
                    self.snake[0], self.apple, self.snake, GRID_SIZE,
                    False, False, fast_mode=True  # Not game over, not ate apple yet, fast mode
                )
                record_game_step(
                    self.snake[0], self.apple, self.snake, GRID_SIZE,
                    self.direction, reward
                )
            except:
                pass  # Ignore RL errors
        
        # Calculate new head position
        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])
        
        # Check wall collision
        if (new_head[0] < 0 or new_head[0] >= GRID_SIZE or 
            new_head[1] < 0 or new_head[1] >= GRID_SIZE):
            self.game_over = True
            # Update high score if needed
            if self.score > self.high_score:
                self.high_score = self.score
                self.save_high_score()
            # Learn from game for RL
            if RL_AVAILABLE:
                try:
                    learn_from_game(True, self.score)
                except:
                    pass
            return False
        
        # Check self collision (exclude tail since it will move forward)
        body_without_tail = self.snake[:-1] if len(self.snake) > 1 else self.snake
        if new_head in body_without_tail:
            self.game_over = True
            # Update high score if needed
            if self.score > self.high_score:
                self.high_score = self.score
                self.save_high_score()
            # Learn from game for RL
            if RL_AVAILABLE:
                try:
                    learn_from_game(True, self.score)
                except:
                    pass
            return False
        
        # Move snake
        self.snake.insert(0, new_head)
        
        # Check if apple is eaten
        if new_head == self.apple:
            self.score += 1
            self.ate_apple_this_step = True
            self.apple = self.generate_apple()
            # Record positive reward for eating apple (RL)
            if RL_AVAILABLE:
                try:
                    from rl_agent import get_rl_agent
                    agent = get_rl_agent()
                    # Use fast_mode=True to skip expensive reachability checks during training
                    reward = agent.get_reward(
                        new_head, self.apple, self.snake, GRID_SIZE,
                        False, True, fast_mode=True  # Not game over, ate apple, fast mode
                    )
                    record_game_step(
                        new_head, self.apple, self.snake, GRID_SIZE,
                        self.direction, reward
                    )
                except:
                    pass
        else:
            # Remove tail if apple not eaten
            self.snake.pop()
        
        return True
    
    def run_game(self):
        """Run a single game to completion, returns final score"""
        while self.update():
            pass
        return self.score


def train_worker(worker_id: int, num_games: int):
    """
    Worker function for training - runs multiple games.
    This will be called in separate threads (thread-safe RL agent).
    """
    scores = []
    for game_num in range(num_games):
        game = HeadlessSnakeGame()
        score = game.run_game()
        scores.append(score)
        
        if (game_num + 1) % 10 == 0:
            avg_score = sum(scores[-10:]) / 10
            print(f"Worker {worker_id}: Game {game_num + 1}/{num_games} | Last 10 avg: {avg_score:.1f}")
    
    return scores


if __name__ == "__main__":
    # Test single game
    game = HeadlessSnakeGame()
    score = game.run_game()
    print(f"Game finished with score: {score}")

