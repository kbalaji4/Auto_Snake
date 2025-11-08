import pygame
import random
import sys

# Initialize Pygame
pygame.init()

# Constants
GRID_SIZE = 15
CELL_SIZE = 30
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
FPS = 10

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
DARK_GREEN = (0, 200, 0)
GRAY = (128, 128, 128)

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

class SnakeGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + 50))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.reset_game()
    
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
                if self.game_over:
                    if event.key == pygame.K_SPACE:
                        self.reset_game()
                    elif event.key == pygame.K_ESCAPE:
                        return False
                else:
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
        if self.game_over:
            return
        
        # Update direction
        self.direction = self.next_direction
        
        # Calculate new head position
        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])
        
        # Check wall collision
        if (new_head[0] < 0 or new_head[0] >= GRID_SIZE or 
            new_head[1] < 0 or new_head[1] >= GRID_SIZE):
            self.game_over = True
            return
        
        # Check self collision
        if new_head in self.snake:
            self.game_over = True
            return
        
        # Move snake
        self.snake.insert(0, new_head)
        
        # Check if apple is eaten
        if new_head == self.apple:
            self.score += 1
            self.apple = self.generate_apple()
        else:
            # Remove tail if no apple eaten
            self.snake.pop()
    
    def draw(self):
        """Draw everything on the screen"""
        # Clear screen
        self.screen.fill(BLACK)
        
        if not self.game_over:
            # Draw grid lines
            for i in range(GRID_SIZE + 1):
                pygame.draw.line(self.screen, GRAY, 
                               (i * CELL_SIZE, 0), 
                               (i * CELL_SIZE, WINDOW_SIZE), 1)
                pygame.draw.line(self.screen, GRAY, 
                               (0, i * CELL_SIZE), 
                               (WINDOW_SIZE, i * CELL_SIZE), 1)
            
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
        
        # Draw game over message
        if self.game_over:
            game_over_text = self.font.render("Game Over!", True, WHITE)
            restart_text = self.font.render("Press SPACE to restart or ESC to quit", True, WHITE)
            text_rect = game_over_text.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE // 2 - 20))
            restart_rect = restart_text.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE // 2 + 20))
            self.screen.blit(game_over_text, text_rect)
            self.screen.blit(restart_text, restart_rect)
        
        pygame.display.flip()
    
    def run(self):
        """Main game loop"""
        running = True
        while running:
            running = self.handle_input()
            self.update()
            self.draw()
            self.clock.tick(FPS)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = SnakeGame()
    game.run()

