import pygame
import random
import sys
import os
from hybrid_a_star import get_next_direction as hybrid_a_star_get_next_direction, set_use_rl, get_use_rl
from hamiltonian_cycle import get_next_direction_cached as hamiltonian_cycle_get_next_direction

# Optional RL integration
try:
    from rl_agent import record_game_step, learn_from_game, get_rl_agent, get_rl_stats
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    def record_game_step(*args, **kwargs):
        pass
    def learn_from_game(*args, **kwargs):
        pass
    def get_rl_agent():
        return None
    def get_rl_stats():
        return None

# Initialize Pygame
pygame.init()

# Constants
GRID_SIZE = 15
CELL_SIZE = 30
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
HIGH_SCORE_FILE = "high_score.txt"

# Speed settings
SPEED_SLOW = 6
SPEED_MEDIUM = 10
SPEED_FAST = 15

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
DARK_GREEN = (0, 200, 0)
GRAY = (128, 128, 128)
YELLOW = (255, 255, 0)
BLUE = (0, 100, 255)

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# Game states
STATE_START = 0
STATE_PLAYING = 1
STATE_AUTO = 2
STATE_GAME_OVER = 3

# Algorithm types
ALGORITHM_HYBRID_A_STAR = 0
ALGORITHM_HAMILTONIAN_CYCLE = 1

class SnakeGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + 50))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.title_font = pygame.font.Font(None, 48)
        self.small_font = pygame.font.Font(None, 28)
        
        # Game state
        self.state = STATE_START
        self.game_speed = SPEED_MEDIUM  # Default to medium
        self.high_score = self.load_high_score()
        self.auto_mode = False
        self.current_path = None  # Path for visualization in auto mode
        self.algorithm = ALGORITHM_HAMILTONIAN_CYCLE  # Default algorithm (for start screen)
        self.ate_apple_this_step = False  # Track if apple was eaten this step
        self.auto_restart = False  # Auto-restart after death in auto mode
        self.restart_delay = 0  # Delay counter for auto-restart
        
        self.reset_game()
    
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
        """Save high score to file"""
        try:
            with open(HIGH_SCORE_FILE, 'w') as f:
                f.write(str(self.high_score))
        except:
            pass
    
    def reset_game(self):
        # Snake starts at 2 cells long in the center
        center = GRID_SIZE // 2
        self.snake = [(center, center), (center - 1, center)]
        self.direction = RIGHT
        self.next_direction = RIGHT
        self.apple = self.generate_apple()
        self.score = 0
        self.game_over = False
    
    def generate_apple(self):
        """Generate apple at a random position that's not occupied by the snake"""
        while True:
            apple = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            if apple not in self.snake:
                return apple
    
    def handle_input(self):
        """Handle keyboard input for both arrow keys and WASD"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                if self.state == STATE_START:
                    # Speed selection
                    if event.key == pygame.K_1 or event.key == pygame.K_KP1:
                        self.game_speed = SPEED_SLOW
                        self.auto_mode = False
                        self.state = STATE_PLAYING
                        self.reset_game()
                    elif event.key == pygame.K_2 or event.key == pygame.K_KP2:
                        self.game_speed = SPEED_MEDIUM
                        self.auto_mode = False
                        self.state = STATE_PLAYING
                        self.reset_game()
                    elif event.key == pygame.K_3 or event.key == pygame.K_KP3:
                        self.game_speed = SPEED_FAST
                        self.auto_mode = False
                        self.state = STATE_PLAYING
                        self.reset_game()
                    elif event.key == pygame.K_a:
                        # Auto mode - default to Hybrid A* for RL learning
                        self.game_speed = SPEED_MEDIUM
                        self.auto_mode = True
                        self.algorithm = ALGORITHM_HYBRID_A_STAR  # Default to Hybrid A* in auto mode
                        self.state = STATE_AUTO
                        self.reset_game()
                    elif event.key == pygame.K_t:
                        # Toggle algorithm
                        self.algorithm = ALGORITHM_HAMILTONIAN_CYCLE if self.algorithm == ALGORITHM_HYBRID_A_STAR else ALGORITHM_HYBRID_A_STAR
                    elif event.key == pygame.K_ESCAPE:
                        return False
                
                elif self.state == STATE_AUTO:
                    # Allow switching algorithm during auto mode
                    if event.key == pygame.K_t:
                        self.algorithm = ALGORITHM_HAMILTONIAN_CYCLE if self.algorithm == ALGORITHM_HYBRID_A_STAR else ALGORITHM_HYBRID_A_STAR
                    # Toggle RL on/off for Hybrid A* (press 'Q' key for Q-learning)
                    elif event.key == pygame.K_q and self.algorithm == ALGORITHM_HYBRID_A_STAR:
                        current_rl_state = get_use_rl()
                        set_use_rl(not current_rl_state)
                        print(f"RL in Hybrid A*: {'ENABLED' if not current_rl_state else 'DISABLED'}")
                
                elif self.state == STATE_GAME_OVER:
                    if event.key == pygame.K_r:
                        # Restart with current settings
                        self.reset_game()
                        # Restore the previous state (auto or playing)
                        if self.auto_mode:
                            self.state = STATE_AUTO
                        else:
                            self.state = STATE_PLAYING
                    elif event.key == pygame.K_SPACE:
                        self.state = STATE_START
                        self.auto_mode = False
                    elif event.key == pygame.K_ESCAPE:
                        return False
                
                elif self.state == STATE_PLAYING:
                    # Manual control only in playing mode
                    # Arrow keys
                    if event.key == pygame.K_UP and self.direction != DOWN:
                        self.next_direction = UP
                    elif event.key == pygame.K_DOWN and self.direction != UP:
                        self.next_direction = DOWN
                    elif event.key == pygame.K_LEFT and self.direction != RIGHT:
                        self.next_direction = LEFT
                    elif event.key == pygame.K_RIGHT and self.direction != LEFT:
                        self.next_direction = RIGHT
                    
                    # WASD keys
                    elif event.key == pygame.K_w and self.direction != DOWN:
                        self.next_direction = UP
                    elif event.key == pygame.K_s and self.direction != UP:
                        self.next_direction = DOWN
                    elif event.key == pygame.K_a and self.direction != RIGHT:
                        self.next_direction = LEFT
                    elif event.key == pygame.K_d and self.direction != LEFT:
                        self.next_direction = RIGHT
        
        return True
    
    def update(self):
        """Update game state"""
        # Handle auto-restart delay
        if self.auto_restart and self.restart_delay > 0:
            self.restart_delay -= 1
            if self.restart_delay <= 0:
                # Auto-restart the game
                self.reset_game()
                self.auto_restart = False
                self.state = STATE_AUTO
            return
        
        if (self.state != STATE_PLAYING and self.state != STATE_AUTO) or self.game_over:
            return
        
        # Track if we ate apple this step (for RL)
        self.ate_apple_this_step = False
        
        # Auto mode: use selected algorithm to determine next direction
        if self.state == STATE_AUTO:
            # Select algorithm based on current setting
            if self.algorithm == ALGORITHM_HYBRID_A_STAR:
                get_next_direction = hybrid_a_star_get_next_direction
            else:  # ALGORITHM_HAMILTONIAN_CYCLE
                get_next_direction = hamiltonian_cycle_get_next_direction
            
            next_dir, path = get_next_direction(
                snake_head=self.snake[0],
                apple=self.apple,
                snake_body=self.snake,
                grid_size=GRID_SIZE,
                current_direction=self.direction
            )
            # Store path for visualization
            self.current_path = path
            if next_dir:
                # Only update if the new direction is valid (not opposite of current)
                if (next_dir[0] * -1, next_dir[1] * -1) != self.direction:
                    self.next_direction = next_dir
        else:
            # Clear path when not in auto mode
            self.current_path = None
        
        # Update direction (for both manual and auto mode)
        self.direction = self.next_direction
        
        # Calculate new head position
        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])
        
        # Record step for RL learning (only for Hybrid A* in auto mode)
        # Record BEFORE moving so we capture the state-action pair
        if RL_AVAILABLE and self.state == STATE_AUTO and self.algorithm == ALGORITHM_HYBRID_A_STAR:
            try:
                from rl_agent import get_rl_agent
                agent = get_rl_agent()
                # Calculate reward for current state (before move)
                reward = agent.get_reward(
                    self.snake[0], self.apple, self.snake, GRID_SIZE,
                    False, False  # Not game over, not ate apple yet
                )
                record_game_step(
                    self.snake[0], self.apple, self.snake, GRID_SIZE,
                    self.direction, reward
                )
            except:
                pass  # Ignore RL errors
        
        # Check wall collision
        if (new_head[0] < 0 or new_head[0] >= GRID_SIZE or 
            new_head[1] < 0 or new_head[1] >= GRID_SIZE):
            self.game_over = True
            self.state = STATE_GAME_OVER
            if self.score > self.high_score:
                self.high_score = self.score
                self.save_high_score()
            # Learn from game for RL
            if RL_AVAILABLE and self.algorithm == ALGORITHM_HYBRID_A_STAR:
                try:
                    learn_from_game(True, self.score)
                except:
                    pass
            # Auto-restart in auto mode
            if self.auto_mode:
                self.auto_restart = True
                self.restart_delay = 60  # Wait 60 frames (~1 second at 60 FPS)
            return
        
        # Check self collision (exclude tail since it will move forward)
        # The tail will move out of the way when we pop it, so we only check collision
        # with the body excluding the tail
        body_without_tail = self.snake[:-1] if len(self.snake) > 1 else self.snake
        if new_head in body_without_tail:
            self.game_over = True
            self.state = STATE_GAME_OVER
            if self.score > self.high_score:
                self.high_score = self.score
                self.save_high_score()
            # Learn from game for RL
            if RL_AVAILABLE and self.algorithm == ALGORITHM_HYBRID_A_STAR:
                try:
                    learn_from_game(True, self.score)
                except:
                    pass
            # Auto-restart in auto mode
            if self.auto_mode:
                self.auto_restart = True
                self.restart_delay = 60  # Wait 60 frames (~1 second at 60 FPS)
            return
        
        # Move snake
        self.snake.insert(0, new_head)
        
        # Check if apple is eaten
        if new_head == self.apple:
            self.score += 1
            self.ate_apple_this_step = True
            self.apple = self.generate_apple()
            # Record positive reward for eating apple (RL)
            if RL_AVAILABLE and self.state == STATE_AUTO and self.algorithm == ALGORITHM_HYBRID_A_STAR:
                try:
                    from rl_agent import get_rl_agent
                    agent = get_rl_agent()
                    reward = agent.get_reward(
                        new_head, self.apple, self.snake, GRID_SIZE,
                        False, True  # Not game over, ate apple
                    )
                    record_game_step(
                        new_head, self.apple, self.snake, GRID_SIZE,
                        self.direction, reward
                    )
                except:
                    pass
        else:
            # Remove tail if no apple eaten
            self.snake.pop()
    
    def draw(self):
        """Draw everything on the screen"""
        # Clear screen
        self.screen.fill(BLACK)
        
        if self.state == STATE_START:
            self.draw_start_screen()
        elif self.state == STATE_PLAYING or self.state == STATE_AUTO:
            self.draw_game()
        elif self.state == STATE_GAME_OVER:
            self.draw_game()
            # Only show game over screen if not auto-restarting
            if not self.auto_restart:
                self.draw_game_over()
        
        pygame.display.flip()
    
    def draw_start_screen(self):
        """Draw the start screen with speed selection"""
        # Title
        title_text = self.title_font.render("SNAKE GAME", True, GREEN)
        title_rect = title_text.get_rect(center=(WINDOW_SIZE // 2, 80))
        self.screen.blit(title_text, title_rect)
        
        # High score
        high_score_text = self.font.render(f"High Score: {self.high_score}", True, YELLOW)
        high_score_rect = high_score_text.get_rect(center=(WINDOW_SIZE // 2, 140))
        self.screen.blit(high_score_text, high_score_rect)
        
        # Speed selection
        speed_title = self.font.render("Select Speed:", True, WHITE)
        speed_title_rect = speed_title.get_rect(center=(WINDOW_SIZE // 2, 220))
        self.screen.blit(speed_title, speed_title_rect)
        
        # Speed options
        y_offset = 280
        speeds = [
            ("1 - Slow", SPEED_SLOW, self.game_speed == SPEED_SLOW and not self.auto_mode),
            ("2 - Medium", SPEED_MEDIUM, self.game_speed == SPEED_MEDIUM and not self.auto_mode),
            ("3 - Fast", SPEED_FAST, self.game_speed == SPEED_FAST and not self.auto_mode)
        ]
        
        for i, (label, speed, is_selected) in enumerate(speeds):
            color = YELLOW if is_selected else WHITE
            speed_text = self.font.render(label, True, color)
            speed_rect = speed_text.get_rect(center=(WINDOW_SIZE // 2, y_offset + i * 50))
            self.screen.blit(speed_text, speed_rect)
            
            # Draw indicator
            if is_selected:
                indicator_x = WINDOW_SIZE // 2 - 100
                pygame.draw.circle(self.screen, YELLOW, (indicator_x, y_offset + i * 50), 5)
        
        # Auto mode option
        auto_y = y_offset + 3 * 50
        auto_color = YELLOW if self.auto_mode else WHITE
        auto_text = self.font.render("A - Auto Mode (AI)", True, auto_color)
        auto_rect = auto_text.get_rect(center=(WINDOW_SIZE // 2, auto_y))
        self.screen.blit(auto_text, auto_rect)
        
        if self.auto_mode:
            indicator_x = WINDOW_SIZE // 2 - 100
            pygame.draw.circle(self.screen, YELLOW, (indicator_x, auto_y), 5)
        
        # Algorithm selection (only show if in auto mode or about to enter auto mode)
        if self.auto_mode:
            algo_y = auto_y + 50
            algo_name = "Hamiltonian Cycle" if self.algorithm == ALGORITHM_HAMILTONIAN_CYCLE else "Hybrid A*"
            algo_text = self.font.render(f"T - Algorithm: {algo_name}", True, BLUE)
            algo_rect = algo_text.get_rect(center=(WINDOW_SIZE // 2, algo_y))
            self.screen.blit(algo_text, algo_rect)
        
        # Instructions
        instruction_text = self.small_font.render("Press 1, 2, 3 to select speed or A for auto mode", True, GRAY)
        instruction_rect = instruction_text.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE - 60))
        self.screen.blit(instruction_text, instruction_rect)
        
        if self.auto_mode:
            algo_instruction = self.small_font.render("Press T to toggle algorithm", True, GRAY)
            algo_instruction_rect = algo_instruction.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE - 30))
            self.screen.blit(algo_instruction, algo_instruction_rect)
        
        esc_text = self.small_font.render("Press ESC to quit", True, GRAY)
        esc_rect = esc_text.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE - 10))
        self.screen.blit(esc_text, esc_rect)
    
    def draw_game(self):
        """Draw the game screen"""
        # Draw grid lines
        for i in range(GRID_SIZE + 1):
            pygame.draw.line(self.screen, GRAY, 
                           (i * CELL_SIZE, 0), 
                           (i * CELL_SIZE, WINDOW_SIZE), 1)
            pygame.draw.line(self.screen, GRAY, 
                           (0, i * CELL_SIZE), 
                           (WINDOW_SIZE, i * CELL_SIZE), 1)
        
        # Draw path visualization in auto mode (before snake so it appears behind)
        if self.state == STATE_AUTO and self.current_path is not None:
            # Draw path cells in blue (skip the first cell which is the snake head)
            for i, (x, y) in enumerate(self.current_path[1:], 1):
                # Don't draw if it's the apple position (will be drawn in red)
                if (x, y) != self.apple:
                    # Use semi-transparent blue for path cells
                    path_surface = pygame.Surface((CELL_SIZE - 2, CELL_SIZE - 2))
                    path_surface.set_alpha(150)  # Semi-transparent
                    path_surface.fill(BLUE)
                    self.screen.blit(path_surface, (x * CELL_SIZE + 1, y * CELL_SIZE + 1))
        
        # Draw snake
        for i, (x, y) in enumerate(self.snake):
            color = GREEN if i == 0 else DARK_GREEN  # Head is brighter
            pygame.draw.rect(self.screen, color, 
                           (x * CELL_SIZE + 1, y * CELL_SIZE + 1, 
                            CELL_SIZE - 2, CELL_SIZE - 2))
        
        # Draw apple
        apple_x, apple_y = self.apple
        pygame.draw.rect(self.screen, RED, 
                       (apple_x * CELL_SIZE + 1, apple_y * CELL_SIZE + 1, 
                        CELL_SIZE - 2, CELL_SIZE - 2))
        
        # Draw score
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, WINDOW_SIZE + 10))
        
        # Draw mode indicator and algorithm
        if self.state == STATE_AUTO:
            mode_text = self.small_font.render("AUTO MODE", True, BLUE)
            self.screen.blit(mode_text, (WINDOW_SIZE // 2 - 50, WINDOW_SIZE + 10))
            algo_name = "Hamiltonian" if self.algorithm == ALGORITHM_HAMILTONIAN_CYCLE else "Hybrid A*"
            algo_text = self.small_font.render(f"Algo: {algo_name} (Press T to switch)", True, BLUE)
            self.screen.blit(algo_text, (WINDOW_SIZE // 2 - 100, WINDOW_SIZE + 30))
            
            # Show RL status if using Hybrid A*
            if self.algorithm == ALGORITHM_HYBRID_A_STAR and RL_AVAILABLE:
                try:
                    rl_enabled = get_use_rl()
                    rl_stats = get_rl_stats()
                    if rl_stats:
                        rl_state = "Active" if rl_enabled else "Disabled"
                        rl_status = f"RL: {rl_state} | Q-table: {rl_stats['q_table_size']} | Games: {rl_stats['games_played']} (Press Q to toggle)"
                        rl_text = self.small_font.render(rl_status, True, YELLOW if rl_enabled else GRAY)
                        self.screen.blit(rl_text, (10, WINDOW_SIZE + 35))
                except:
                    pass
        
        # Draw high score
        high_score_text = self.small_font.render(f"High Score: {self.high_score}", True, YELLOW)
        self.screen.blit(high_score_text, (WINDOW_SIZE - 150, WINDOW_SIZE + 15))
    
    def draw_game_over(self):
        """Draw game over overlay"""
        # Semi-transparent overlay
        overlay = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
        overlay.set_alpha(180)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        # Game over text
        game_over_text = self.title_font.render("Game Over!", True, RED)
        game_over_rect = game_over_text.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE // 2 - 60))
        self.screen.blit(game_over_text, game_over_rect)
        
        # Final score
        final_score_text = self.font.render(f"Final Score: {self.score}", True, WHITE)
        final_score_rect = final_score_text.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE // 2 - 10))
        self.screen.blit(final_score_text, final_score_rect)
        
        # High score update
        if self.score == self.high_score and self.score > 0:
            new_high_text = self.font.render("New High Score!", True, YELLOW)
            new_high_rect = new_high_text.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE // 2 + 30))
            self.screen.blit(new_high_text, new_high_rect)
        
        # Instructions
        restart_text = self.small_font.render("Press R to restart with current settings", True, WHITE)
        restart_rect = restart_text.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE // 2 + 60))
        self.screen.blit(restart_text, restart_rect)
        
        menu_text = self.small_font.render("Press SPACE to return to menu", True, WHITE)
        menu_rect = menu_text.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE // 2 + 90))
        self.screen.blit(menu_text, menu_rect)
        
        esc_text = self.small_font.render("Press ESC to quit", True, GRAY)
        esc_rect = esc_text.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE // 2 + 120))
        self.screen.blit(esc_text, esc_rect)
    
    def run(self):
        """Main game loop"""
        running = True
        while running:
            running = self.handle_input()
            self.update()
            self.draw()
            # Use dynamic FPS based on selected speed
            self.clock.tick(self.game_speed)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = SnakeGame()
    game.run()

