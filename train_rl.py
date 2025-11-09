"""
Concurrent RL training script for Snake game.
Runs multiple game instances in parallel threads to speed up learning.
Uses threading for shared RL agent state (thread-safe with locks).
"""

import threading
import time
from headless_game import train_worker
from rl_agent import get_rl_agent


def train_concurrent(num_workers: int = 4, games_per_worker: int = 100):
    """
    Train RL agent using multiple concurrent worker threads.
    
    Args:
        num_workers: Number of parallel worker threads
        games_per_worker: Number of games each worker should play
    """
    print(f"Starting concurrent RL training with {num_workers} worker threads")
    print(f"Each worker will play {games_per_worker} games")
    print(f"Total games: {num_workers * games_per_worker}")
    print("-" * 50)
    
    start_time = time.time()
    
    # Create worker threads
    threads = []
    for worker_id in range(num_workers):
        t = threading.Thread(
            target=train_worker,
            args=(worker_id, games_per_worker)
        )
        t.start()
        threads.append(t)
    
    # Wait for all workers to complete
    for t in threads:
        t.join()
    
    elapsed_time = time.time() - start_time
    total_games = num_workers * games_per_worker
    
    print("-" * 50)
    print(f"Training completed!")
    print(f"Total games: {total_games}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Games per second: {total_games / elapsed_time:.2f}")
    print(f"Average time per game: {elapsed_time / total_games:.3f} seconds")
    
    # Print final statistics
    try:
        agent = get_rl_agent()
        stats = agent.get_stats()
        print(f"\nFinal RL Statistics:")
        print(f"  Q-table size: {stats['q_table_size']}")
        print(f"  Total games played: {stats['games_played']}")
        if stats.get('neighbor_stats'):
            print(f"  Neighbor statistics:")
            for key, value in stats['neighbor_stats'].items():
                print(f"    {key}: {value['death_rate']:.1%} death rate ({value['total']} samples)")
        
        # Save final Q-table
        agent.save_q_table()
        print(f"\nQ-table saved to {agent.q_table_file}")
    except Exception as e:
        print(f"Error getting stats: {e}")


if __name__ == "__main__":
    import sys
    
    # Default values
    num_workers = 4
    games_per_worker = 100
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        num_workers = int(sys.argv[1])
    if len(sys.argv) > 2:
        games_per_worker = int(sys.argv[2])
    
    print("Concurrent RL Training for Snake Game")
    print("=" * 50)
    print(f"Usage: python train_rl.py [num_workers] [games_per_worker]")
    print(f"Current: {num_workers} workers, {games_per_worker} games each")
    print("=" * 50)
    print()
    
    train_concurrent(num_workers, games_per_worker)

