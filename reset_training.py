"""
Reset RL training data - clears high score and Q-table.
Use this when you want to start training from scratch after policy changes.
"""

import os

def reset_training():
    """Reset all training data"""
    files_to_remove = [
        "high_score.txt",
        "q_table.pkl",
        "high_score.txt.lock"  # Lock file if it exists
    ]
    
    removed = []
    for filename in files_to_remove:
        if os.path.exists(filename):
            try:
                os.remove(filename)
                removed.append(filename)
                print(f"✓ Removed {filename}")
            except Exception as e:
                print(f"✗ Failed to remove {filename}: {e}")
        else:
            print(f"- {filename} not found (already clean)")
    
    if removed:
        print(f"\n✓ Training data reset! Removed {len(removed)} file(s).")
        print("You can now start fresh training.")
    else:
        print("\n✓ No training data found - already clean!")


if __name__ == "__main__":
    print("Reset RL Training Data")
    print("=" * 50)
    print("This will delete:")
    print("  - high_score.txt")
    print("  - q_table.pkl")
    print("=" * 50)
    
    response = input("\nAre you sure you want to reset? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        reset_training()
    else:
        print("Reset cancelled.")

