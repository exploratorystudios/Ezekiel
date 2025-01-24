import pygame
import sys
import random
import collections
from collections import deque
from tetris_agent import TetrisAgent  # Import the agent and neural network
import torch.optim as optim
import torch.nn as nn
import torch
import torch.multiprocessing as mp  
from torch.multiprocessing import Value, Lock 
import numpy as np
import os
import re
import time
import warnings
import pickle
import copy
import logging
import traceback

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

logging.basicConfig(
    filename='tetris_player.log',
    filemode='a',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 1) Define an ACTION_MAP from BFS tokens to agent action indices

BFS_ACTION_MAP = {
    'CW': 0,       # Rotate clockwise
    'CCW': 1,      # Rotate counterclockwise
    'L': 2,        # Move left
    'R': 3,        # Move right
    'D': 4,        # Soft drop
    'SPACE': 5,    # Hard drop
    # 'NOP': 6,    # No operation (if included)
}

def scale_metrics(score, lines_cleared, pieces_placed, level):
    """
    Scale game metrics to a normalized range.
    Adjust the scaling factors based on your game's maximum expected values.
    """
    score_scaled = score / 10000.0
    lines_cleared_scaled = lines_cleared / 100.0
    pieces_placed_scaled = pieces_placed / 1000.0
    level_scaled = level / 100.0
    return [score_scaled, lines_cleared_scaled, pieces_placed_scaled, level_scaled]

def grid_to_input(grid):
    """Convert the 2D grid into a flattened list with binary values."""
    flattened = []
    for row in grid:
        for cell in row:
            flattened.append(0 if cell == 0 else 1)  # Assuming 0 is empty
    return flattened

def construct_state_tensor(grid, score, lines_cleared, pieces_placed, level, device):
    """
    Constructs a 5-channel state tensor from the grid and game metrics.

    Parameters:
    - grid (list of lists): Current Tetris grid state.
    - score (int): Current score.
    - lines_cleared (int): Lines cleared in the last move.
    - pieces_placed (int): Total pieces placed.
    - level (int): Current level.
    - device (torch.device): Device to place the tensor on.

    Returns:
    - torch.Tensor: Combined 5-channel state tensor of shape (5, 20, 10).
    """
    # Convert grid to binary (1 for occupied, 0 for empty)
    input_grid = grid_to_input(grid)  # 1 channel

    # Scale game metrics
    scaled_metrics = scale_metrics(score, lines_cleared, pieces_placed, level)  # 4 channels

    # Convert grid to tensor without the batch dimension
    grid_arr = np.array(input_grid, dtype=np.float32).reshape((1, 20, 10))  # Shape: (1, 20, 10)
    grid_tensor = torch.tensor(grid_arr, device=device)  # Shape: (1, 20, 10)

    # Convert metrics to tensor and expand to match grid dimensions
    metrics_tensor = torch.tensor(scaled_metrics, dtype=torch.float32, device=device)  # Shape: (4,)
    metrics_expanded = metrics_tensor.view(4, 1, 1).expand(4, 20, 10)  # Shape: (4, 20, 10)

    # Concatenate grid and metrics to form a 5-channel tensor
    combined_input = torch.cat((grid_tensor, metrics_expanded), dim=0)  # Shape: (5, 20, 10)

    return combined_input

def inspect_bfs_data_details(bfs_data, sample_size=5):
    print(f"\n--- Inspecting First {sample_size} BFS Records ---")
    for idx, sample in enumerate(bfs_data[:sample_size]):
        if len(sample) != 8:
            print(f"Record {idx}: Invalid length {len(sample)}. Expected 8 elements.")
            print(f"  Sample Content: {sample}")
            logging.warning(f"Record {idx}: Invalid length {len(sample)}. Expected 8 elements.")
            continue
        try:
            input_grid, best_path, final_x, final_rot, score, lines_cleared, pieces_placed, level = sample
        except ValueError as e:
            print(f"Record {idx}: Error unpacking sample - {e}")
            print(f"  Sample Content: {sample}")
            logging.error(f"Record {idx}: Error unpacking sample - {e}")
            continue

        # Check if final_rot is defined and valid
        if final_rot is None:
            print(f"Record {idx}: final_rot is None.")
            logging.warning(f"Record {idx}: final_rot is None.")
            continue

        print(f"Record {idx}:")
        print(f"  Input Grid Length: {len(input_grid)}")
        print(f"  Best Path: {best_path}")
        print(f"  Final X: {final_x}")
        print(f"  Final Rotation: {final_rot}")
        print(f"  Score: {score}")
        print(f"  Lines Cleared: {lines_cleared}")
        print(f"  Pieces Placed: {pieces_placed}")
        print(f"  Level: {level}\n")

        if isinstance(best_path, list):
            print(f"  Best Path Type: List with {len(best_path)} tokens")
        elif isinstance(best_path, str):
            print(f"  Best Path Type: String - {best_path}")
        else:
            print(f"  Best Path Type: {type(best_path)}")

        print("-" * 50)

def inspect_bfs_tokens(bfs_data, sample_size=100):
    """
    Inspects and prints unique action tokens in the BFS data.
    """
    unique_tokens = set()
    for idx, sample in enumerate(bfs_data[:sample_size]):
        if len(sample) != 8:
            continue  # Skip invalid records
        _, best_path, _, _, _, _, _, _ = sample
        
        if isinstance(best_path, list):
            tokens = best_path
        elif isinstance(best_path, str):
            tokens = best_path.split()
        else:
            logging.warning(f"Record {idx}: best_path is neither list nor string.")
            continue
        
        for token in tokens:
            normalized_token = token.upper().strip()
            unique_tokens.add(normalized_token)
    
    print(f"Unique action tokens in BFS data (first {sample_size} records): {unique_tokens}")
    logging.info(f"Unique action tokens in BFS data (first {sample_size} records): {unique_tokens}")

def supervised_train_from_bfs(
    agent, 
    bfs_data_path="training_data.pkl", 
    epochs=5, 
    batch_size=32
):
    """
    Loads BFS data and trains the agent's model via supervised learning.
    """
    # --- Load BFS Data ---
    try:
        with open(bfs_data_path, 'rb') as f:
            bfs_data = pickle.load(f)
        print(f"Successfully loaded BFS data. Records: {len(bfs_data)}")
        logging.info(f"Successfully loaded BFS data. Records: {len(bfs_data)}")
    except Exception as e:
        print(f"Failed to load BFS data from {bfs_data_path}: {e}")
        logging.error(f"Failed to load BFS data from {bfs_data_path}: {e}")
        return

    # --- Inspect Action Tokens ---
    inspect_bfs_tokens(bfs_data, sample_size=100)  # Inspect first 100 records
    device = agent.device
    action_size = agent.action_size

    X_list = []
    y_list = []

    # --- Build training pairs (state -> action) from BFS data ---
    for idx, sample in enumerate(bfs_data):
        # Each sample must match the structure: (input_grid, best_path, final_x, final_rot, score, lines_cleared, pieces_placed, level)
        if len(sample) != 8:
            print(f"Record {idx} missing elements (expected 8). Skipping.")
            logging.warning(f"Record {idx} missing elements (expected 8). Skipping.")
            continue

        input_grid, best_path, final_x, final_rot, score, lines_cleared, pieces_placed, level = sample

        # Convert the flattened grid to a 1-channel tensor of shape (1,1,20,10)
        try:
            arr = np.array(input_grid, dtype=np.float32).reshape((1, 1, 20, 10))
            grid_tensor = torch.tensor(arr, device=device)  # shape: (1,1,20,10)
        except Exception as e:
            print(f"Record {idx}: Error reshaping input_grid: {e}. Skipping.")
            logging.error(f"Record {idx}: Error reshaping input_grid: {e}. Skipping.")
            continue

        # Scale game metrics and expand them into 4 additional channels
        scaled_metrics = scale_metrics(score, lines_cleared, pieces_placed, level)
        metrics_tensor = torch.tensor(scaled_metrics, dtype=torch.float32, device=device)  # shape: (4,)
        metrics_expanded = metrics_tensor.view(1, -1, 1, 1).expand(1, 4, 20, 10)           # shape: (1,4,20,10)

        # Combine grid + metrics into a single 5-channel tensor: (1,5,20,10)
        try:
            combined_input = torch.cat((grid_tensor, metrics_expanded), dim=1)
        except Exception as e:
            print(f"Record {idx}: Error concatenating metrics: {e}. Skipping.")
            logging.error(f"Record {idx}: Error concatenating metrics: {e}. Skipping.")
            continue

        # For each action token in the BFS path, create a separate (state, action) example
        for token in best_path:
            normalized_token = token.upper()
            if normalized_token not in BFS_ACTION_MAP:
                print(f"Record {idx}: Unknown BFS token '{token}'. Skipping this action.")
                logging.warning(f"Record {idx}: Unknown BFS token '{token}'. Skipping this action.")
                continue
            action_idx = BFS_ACTION_MAP[normalized_token]
            if action_idx >= action_size:
                print(f"Record {idx}: Action index {action_idx} out of range. Skipping.")
                logging.warning(f"Record {idx}: Action index {action_idx} out of range. Skipping.")
                continue
            X_list.append(combined_input)
            y_list.append(action_idx)
            logging.debug(f"Record {idx}: Mapped token '{token}' to action {action_idx}")

    # --- Inspect Action Tokens ---
    inspect_bfs_tokens(bfs_data, sample_size=100)  # Inspect first 100 records

    # Log first few best_paths
    inspect_bfs_data_details(bfs_data, sample_size=5)
    if not X_list:
        print("No valid BFS records found. Supervised training aborted.")
        logging.error("No valid BFS records found. Supervised training aborted.")
        return
    else:
        print(f"Valid BFS records for training: {len(X_list)}")
        logging.info(f"Valid BFS records for training: {len(X_list)}")

    # --- Convert to tensors for training ---
    try:
        X = torch.cat(X_list, dim=0)  # shape: (N,5,20,10)
        y = torch.tensor(y_list, dtype=torch.long, device=device)  # shape: (N,)
        print(f"Prepared BFS training data. X shape: {X.shape}, y shape: {y.shape}")
        logging.info(f"Prepared BFS training data. X shape: {X.shape}, y shape: {y.shape}")
    except Exception as e:
        print(f"Error creating training tensors: {e}")
        logging.error(f"Error creating training tensors: {e}")
        return

    # --- Define a simple training loop (CrossEntropy) ---
    model = agent.gpu_model
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    dataset_size = len(X)
    num_batches = (dataset_size + batch_size - 1) // batch_size
    print(f"Supervised training: {epochs} epochs, ~{num_batches} batches/epoch")
    logging.info(f"Supervised training: {epochs} epochs, ~{num_batches} batches/epoch")

    for epoch in range(epochs):
        permutation = torch.randperm(dataset_size)
        epoch_loss = 0.0

        for b_idx in range(num_batches):
            start_idx = b_idx * batch_size
            end_idx = min(start_idx + batch_size, dataset_size)
            indices = permutation[start_idx:end_idx]

            batch_X = X[indices]    # shape: (B,5,20,10)
            batch_y = y[indices]    # shape: (B,)

            optimizer.zero_grad()
            outputs = model(batch_X)  # shape: (B, action_size)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / num_batches
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")
        logging.info(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    print("BFS supervised training complete.\n")
    logging.info("BFS supervised training complete.")

    # --- Save the trained model ---
    model_save_path = "trained_bfs_model.pth"
    agent.save_model(model_save_path)
    print(f"Model parameters saved to {model_save_path}")
    logging.info(f"Model parameters saved to {model_save_path}")

# -------------------- END SUPERVISED BFS TRAINING CHANGES --------------------

def save_training_data(data, file_path="training_data.pkl"):
    """
    Saves the training data to a pickle file.

    Parameters:
    - data (list): List of training samples.
    - file_path (str): Path to the pickle file.
    """
    try:
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        logging.info(f"Training data saved to {file_path}. Total samples: {len(data)}")
    except Exception as e:
        logging.error(f"Failed to save training data to {file_path}: {e}")

def extract_path(base_save_path):
    if isinstance(base_save_path, dict):
        # Get the actual path from DictProxy
        return base_save_path.get('path', None)
    return base_save_path

def save_training_state(agent, optimizer, generation, base_save_path, save_lock=None, force_save=False):
    unique_id = f"{time.time()}_{os.getpid()}"
    temp_save_path = f"{base_save_path}.{unique_id}.tmp"
    
    # Prepare the state to be saved, including the model state, optimizer state, generation, and memory
    save_state = {
        'model_state_dict': agent.gpu_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'generation': generation,
        'memory': list(agent.memory)  # Convert deque to list for saving
    }
    
    print(f"[DEBUG] Preparing to save checkpoint to {temp_save_path}. Save state keys: {save_state.keys()}")
    logging.debug(f"Preparing to save checkpoint to {temp_save_path}. Save state keys: {save_state.keys()}")

    # Save the state to a temporary file first
    try:
        torch.save(save_state, temp_save_path)
        logging.info(f"Temporary checkpoint saved at {temp_save_path}. Now finalizing save to {base_save_path}.")
        print(f"[DEBUG] Temporary checkpoint saved at {temp_save_path}. Now finalizing save to {base_save_path}.")
    except Exception as e:
        print(f"[DEBUG] Failed to save temporary checkpoint: {e}")
        logging.error(f"Failed to save temporary checkpoint: {e}")
        return

    def finalize_save():
        try:
            if not os.path.exists(base_save_path):
                os.rename(temp_save_path, base_save_path)
                print(f"[DEBUG] Final checkpoint saved at {base_save_path}.")
                logging.info(f"Final checkpoint saved at {base_save_path}.")
            else:
                existing_state = torch.load(base_save_path)
                existing_generation = existing_state.get('generation', 0)
                if force_save or generation > existing_generation:
                    os.remove(base_save_path)
                    os.rename(temp_save_path, base_save_path)
                    print(f"[DEBUG] Overwritten old checkpoint with new generation {generation} at {base_save_path}.")
                    logging.info(f"Overwritten old checkpoint with new generation {generation} at {base_save_path}.")
                else:
                    print(f"[DEBUG] Did not overwrite checkpoint at {base_save_path}, as existing generation {existing_generation} is newer.")
                    logging.info(f"Did not overwrite checkpoint at {base_save_path}, existing generation {existing_generation} is newer.")
                    os.remove(temp_save_path)
        except Exception as e:
            print(f"[DEBUG] Error during finalize_save: {e}")
            logging.error(f"Error during finalize_save: {e}")
            if os.path.exists(temp_save_path):
                os.remove(temp_save_path)

    try:
        if save_lock:
            with save_lock:
                finalize_save()
        else:
            finalize_save()
    except Exception as e:
        print(f"[DEBUG] Error during save by process {os.getpid()}: {str(e)}")
        logging.error(f"Error during save by process {os.getpid()}: {str(e)}")
        if os.path.exists(temp_save_path):
            os.remove(temp_save_path)

def load_training_state(agent, base_save_path, optimizer, max_retries=5, retry_delay=1.0):
    base_save_path = extract_path(base_save_path)  # Ensure it's a string path
    print(f"[DEBUG] Attempting to load checkpoint from: {base_save_path}")
    logging.debug(f"Attempting to load checkpoint from: {base_save_path}")

    for attempt in range(max_retries):
        try:
            if os.path.exists(base_save_path):
                checkpoint = torch.load(base_save_path)
                print(f"[DEBUG] Loaded checkpoint with keys: {checkpoint.keys()} from {base_save_path}")
                logging.debug(f"Loaded checkpoint with keys: {checkpoint.keys()} from {base_save_path}")

                # Ensure the checkpoint contains all required keys
                if all(key in checkpoint for key in ['model_state_dict', 'optimizer_state_dict', 'generation', 'memory']):
                    # **Convert dict transitions to tuples if necessary**
                    if len(checkpoint['memory']) > 0 and isinstance(checkpoint['memory'][0], dict):
                        logging.info("Converting memory transitions from dicts to tuples.")
                        print("[DEBUG] Converting memory transitions from dicts to tuples.")
                        converted_memory = []
                        for transition in checkpoint['memory']:
                            if isinstance(transition, dict):
                                try:
                                    converted_transition = (
                                        transition['state'],
                                        transition['action'],
                                        transition['reward'],
                                        transition['next_state'],
                                        transition['done']
                                    )
                                    converted_memory.append(converted_transition)
                                except KeyError as e:
                                    logging.error(f"Missing key in transition during conversion: {e}")
                                    print(f"[DEBUG] Missing key in transition during conversion: {e}")
                            elif isinstance(transition, tuple):
                                # Additional type checks for tuple elements
                                if len(transition) != 5:
                                    logging.error(f"Transition tuple length mismatch: {len(transition)}")
                                    print(f"[DEBUG] Transition tuple length mismatch: {len(transition)}")
                                    continue
                                state, action, reward, next_state, done = transition
                                if not (isinstance(state, torch.Tensor) and isinstance(action, int) and 
                                        (isinstance(reward, float) or isinstance(reward, int)) and 
                                        isinstance(next_state, torch.Tensor) and isinstance(done, bool)):
                                    logging.error(f"Transition tuple has incorrect types: {transition}")
                                    print(f"[DEBUG] Transition tuple has incorrect types: {transition}")
                                    continue
                                converted_memory.append(transition)
                            else:
                                logging.error(f"Unknown transition type: {type(transition)}. Skipping.")
                                print(f"[DEBUG] Unknown transition type: {type(transition)}. Skipping.")
                        checkpoint['memory'] = converted_memory
                        logging.info("Conversion of memory transitions complete.")
                        print("[DEBUG] Conversion of memory transitions complete.")
                    
                    agent.gpu_model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    agent.memory = deque(checkpoint['memory'], maxlen=2000)  # Load the memory buffer
                    print(f"[DEBUG] Loaded training state from generation {checkpoint['generation']} from {base_save_path}")
                    logging.info(f"Loaded training state from generation {checkpoint['generation']} from {base_save_path}")
                    return checkpoint['generation']
                else:
                    print(f"[DEBUG] Checkpoint {base_save_path} is missing required keys: {checkpoint.keys()}")
                    logging.warning(f"Checkpoint {base_save_path} is missing required keys: {checkpoint.keys()}")
                    return 0
            else:
                print(f"[DEBUG] No checkpoint found at {base_save_path}. Starting from scratch.")
                logging.info(f"No checkpoint found at {base_save_path}. Starting from scratch.")
                return 0
        except Exception as e:
            print(f"[DEBUG] Error loading checkpoint: {e}. Retrying in {retry_delay} seconds...")
            logging.error(f"Error loading checkpoint: {e}. Retrying in {retry_delay} seconds...", exc_info=True)
            time.sleep(retry_delay)
    
    raise RuntimeError(f"[DEBUG] Failed to load training state after {max_retries} attempts.")

def get_state_feature_vector(state, device):
    try:
        grid = np.array(state['board'])  
    except Exception as e:
        raise e

    grid = np.expand_dims(grid, axis=0)  
    return torch.tensor(grid, dtype=torch.float32).to(device)

def draw_grid(grid, surface):
    reverse_color_mapping = {v: k for k, v in color_mapping.items()}  # Reverse the mapping

    for i, row in enumerate(grid):
        for j, block in enumerate(row):
            if block:
                color = reverse_color_mapping.get(block, WHITE)  # Map back to color
                pygame.draw.rect(surface, color, (j * BLOCK_SIZE, i * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(surface, WHITE, (j * BLOCK_SIZE, i * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 1)

def draw_score(surface, score):
    font = pygame.font.SysFont(None, 36)
    score_text = font.render(f'Score: {score}', True, BLACK)
    surface.blit(score_text, (10, 10))

def draw_level(surface, level):
    font = pygame.font.SysFont(None, 36)
    level_text = font.render(f'Level: {level}', True, BLACK)
    surface.blit(level_text, (10, 40))

def draw_next_tetrimino(surface, next_tetrimino_key):
    tetrimino_info = TETRIMINOS[next_tetrimino_key]  # Access the shape and color from the dictionary
    draw_x = GRID_WIDTH * BLOCK_SIZE - 2 * BLOCK_SIZE  # Position to draw next Tetrimino
    draw_y = 0  # Adjust as needed for spacing

    draw_tetrimino(surface, tetrimino_info['shape'], draw_x, draw_y, tetrimino_info['color'], scale=0.5, is_ghost=False)

def get_game_state(grid):
    """
    Returns the current state of the game.
    This function will return the grid as the game state.

    Args:
    - grid: The current grid of the game.

    Returns:
    - state: A dictionary representing the current state of the game.
    """
    state = {
        'board': grid
    }
    return state

BFS_WEIGHTS = {
    'complete_lines': 300,
    'aggregate_height': -30,
    'holes': -100,
    'bumpiness': -15,
    'flat_bonus': 5
}

def get_center_penalty(x):
    """Calculate penalty based on the x-position to encourage central stacking."""
    center = GRID_WIDTH / 2
    distance_from_center = abs(x - center)
    penalty = distance_from_center * 2  # Adjust multiplier as needed
    return penalty

def get_reward_and_next_state(
    agent,
    grid,
    lines_cleared,
    total_lines_cleared,
    game_over,
    rotations=0,
    moved_horizontally=False,
    moved_down=False,
    locked=False,
    tetrimino_shape=None,
    x=0,
    y=0,
    score=0,
    pieces_placed=0,
    level=1,
    device=None
):
    """
    Enhanced Tetris reward function aligned with BFS heuristic and improved for better performance.
    """
    # Initialize reward
    reward = -1  # Step penalty

    # 1) LINE CLEAR REWARD
    line_clear_reward_map = {1: 1000, 2: 3000, 3: 7000, 4: 14000}
    line_clear_points = line_clear_reward_map.get(lines_cleared, 0)
    reward += line_clear_points

    if lines_cleared > 0:
        print(f"[DEBUG] Lines cleared: {lines_cleared} -> +{line_clear_points}")

    # 2) ROTATION PENALTY
    if rotations > 3:
        rotation_penalty = (rotations - 3) * 5
        reward -= rotation_penalty
        print(f"[DEBUG] Rotation penalty => -{rotation_penalty}")

    # After state updates
    after_holes = count_holes(grid)
    after_max_height, after_col_heights = get_max_height_and_column_heights(grid)
    after_bumpiness = get_bumpiness(after_col_heights)

    if locked:
        # Before locking, remove the locked piece to measure before state
        grid_without = [row[:] for row in grid]
        for i, row_block in enumerate(tetrimino_shape):
            for j, block in enumerate(row_block):
                if block and 0 <= (y + i) < GRID_HEIGHT and 0 <= (x + j) < GRID_WIDTH:
                    grid_without[y + i][x + j] = 0

        before_holes = count_holes(grid_without)
        before_max_height, before_col_heights = get_max_height_and_column_heights(grid_without)
        before_bumpiness = get_bumpiness(before_col_heights)

        # 3) APPLY BFS HEURISTIC WEIGHTS
        reward += (
            BFS_WEIGHTS['complete_lines'] * lines_cleared
            + BFS_WEIGHTS['aggregate_height'] * after_max_height
            + BFS_WEIGHTS['holes'] * after_holes
            + BFS_WEIGHTS['bumpiness'] * after_bumpiness
        )

        # 4) ADD FLATNESS BONUS
        flat_bonus = (GRID_HEIGHT - after_max_height) * BFS_WEIGHTS['flat_bonus']
        reward += flat_bonus
        if flat_bonus > 0:
            print(f"[DEBUG] Flatness bonus: +{flat_bonus}")

        # 5) CENTRALIZATION PENALTY
        central_penalty = 0
        for row in grid:
            for x_pos, cell in enumerate(row):
                if cell != 0:
                    central_penalty += get_center_penalty(x_pos)
        reward -= central_penalty
        if central_penalty > 0:
            print(f"[DEBUG] Centralization penalty: -{central_penalty}")

        # 6) HARSH PENALTY IF ANY COLUMN ABOVE 75%
        if after_max_height >= 0.75 * GRID_HEIGHT:
            reward -= 500
            print("[DEBUG] Tall column penalty: -500")

        # 7) GAME OVER PENALTY
        if game_over:
            reward -= 3000  # even bigger than previous
            print("[DEBUG] Game over penalty => -3000")

    # Build next state with 5 channels
    next_state = construct_state_tensor(grid, score, lines_cleared, pieces_placed, level, device)

    return reward, next_state

# -------------- Helper Functions ---------------

def count_holes(board):
    """Count the number of holes in the board."""
    holes = 0
    for col in range(GRID_WIDTH):
        for row in range(1, GRID_HEIGHT):
            if board[row][col] == 0 and board[row-1][col] != 0:
                holes += 1
    return holes

def get_max_height_and_column_heights(board):
    """Return max column height and array of each column's height."""
    col_heights = []
    for col in range(GRID_WIDTH):
        h = 0
        for row in range(GRID_HEIGHT):
            if board[row][col] != 0:
                h = GRID_HEIGHT - row
                break
        col_heights.append(h)
    return max(col_heights) if col_heights else 0, col_heights

def get_bumpiness(col_heights):
    """Sum of absolute differences between adjacent columns."""
    bumpiness = 0
    for i in range(len(col_heights)-1):
        bumpiness += abs(col_heights[i] - col_heights[i+1])
    return bumpiness

# Initialize Pygame
pygame.init()
pygame.mixer.init()

# Constants
pygame.display.set_caption("Ezekiel")  # Set window title to "Ezekiel"
WINDOW_WIDTH = 300
WINDOW_HEIGHT = 600
BLOCK_SIZE = 30

GRID_WIDTH = WINDOW_WIDTH // BLOCK_SIZE
GRID_HEIGHT = WINDOW_HEIGHT // BLOCK_SIZE

LEVEL_UP_SCORE = 1000  # Score needed to level up
MIN_FALL_SPEED = 100   # Minimum fall speed in milliseconds
FALL_SPEED_DECREMENT = 50  # Speed increase per level

SCORING_RULES = {
    1: 100,
    2: 300,
    3: 600,
    4: 1000,
}

SCORE_INCREMENT = 100  # Score increase for each cleared line

# Colors
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

# Sounds
try:
    rotate_sound = pygame.mixer.Sound("rotate.wav")
    lock_sound = pygame.mixer.Sound("lock.wav")
    hold_sound = pygame.mixer.Sound("hold.wav")
    clear_sound = pygame.mixer.Sound("clear.wav")
    renis_clear_sound = pygame.mixer.Sound("renisclear.wav")
except pygame.error as e:
    print(f"Error loading sound files: {e}")
    # Assign default sounds or handle accordingly
    rotate_sound = pygame.mixer.Sound(None)
    lock_sound = pygame.mixer.Sound(None)
    hold_sound = pygame.mixer.Sound(None)
    clear_sound = pygame.mixer.Sound(None)
    renis_clear_sound = pygame.mixer.Sound(None)

# Tetrimino shapes
TETRIMINOS = {
    'O': {'shape': [[1, 1],
                    [1, 1]], 'color': (255, 255, 0)},  # Yellow
    'I': {'shape': [[1, 1, 1, 1]], 'color': (0, 255, 255)},  # Cyan
    'S': {'shape': [[0, 1, 1],
                    [1, 1, 0]], 'color': (0, 255, 0)},  # Green
    'Z': {'shape': [[1, 1, 0],
                    [0, 1, 1]], 'color': (255, 0, 0)},  # Red
    'L': {'shape': [[1, 1, 1],
                    [1, 0, 0]], 'color': (255, 165, 0)},  # Orange
    'J': {'shape': [[1, 1, 1],
                    [0, 0, 1]], 'color': (0, 0, 255)},  # Blue
    'T': {'shape': [[1, 1, 1],
                    [0, 1, 0]], 'color': (128, 0, 128)},  # Purple
}
color_mapping = {
    (0, 255, 255): 1,    # Cyan (I)
    (255, 255, 0): 2,    # Yellow (O)
    (0, 255, 0): 3,      # Green (S)
    (255, 0, 0): 4,      # Red (Z)
    (255, 165, 0): 5,    # Orange (L)
    (0, 0, 255): 6,      # Blue (J)
    (128, 0, 128): 7,    # Purple (T)
}

# To keep track of the last three generated pieces
last_three_pieces = collections.deque(maxlen=3)

# Initialize window
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

def generate_tetrimino():
    while True:
        tetrimino_key = random.choice(list(TETRIMINOS.keys()))
        # Check if the new key is not repeating more than 3 times
        if last_three_pieces.count(tetrimino_key) < 3:
            break
    last_three_pieces.append(tetrimino_key)  # Add the new key to the queue
    
    tetrimino_shape = TETRIMINOS[tetrimino_key]['shape']
    x_position = random.randint(0, GRID_WIDTH - len(tetrimino_shape[0]))
    return {'shape': tetrimino_shape, 'position': (x_position, 0), 'color': TETRIMINOS[tetrimino_key]['color'], 'key': tetrimino_key}

def update_tetriminos(tetriminos):
    for tetrimino in tetriminos:
        x, y = tetrimino['position']
        tetrimino['position'] = (x, y + 1)
    tetriminos[:] = [t for t in tetriminos if t['position'][1] < GRID_HEIGHT]

def draw_tetrimino(surface, shape, x, y, color, is_ghost=False, scale=1):
    for i, row in enumerate(shape):
        for j, block in enumerate(row):
            if block:
                scaled_block_size = int(BLOCK_SIZE * scale)
                if is_ghost:
                    # Draw semi-transparent ghost piece
                    ghost_color = (255, 255, 255, 100)  # White with transparency
                    ghost_surface = pygame.Surface((scaled_block_size, scaled_block_size), pygame.SRCALPHA)
                    ghost_surface.fill((*color, 100))
                    surface.blit(ghost_surface, (x + j * scaled_block_size, y + i * scaled_block_size))
                else:
                    pygame.draw.rect(surface, color, (x + j * BLOCK_SIZE, y + i * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
                    pygame.draw.rect(surface, WHITE, (x + j * BLOCK_SIZE, y + i * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 1)  # Always draw white border

def is_valid_move(grid, shape, x, y):
    for i, row in enumerate(shape):
        for j, cell in enumerate(row):
            if cell:
                if y + i >= GRID_HEIGHT or x + j < 0 or x + j >= GRID_WIDTH:
                    return False
                if grid[y + i][x + j]:
                    return False
    return True

def rotate_tetrimino(tetrimino, grid, x, y):
    shape = tetrimino['shape']
    
    # Rotate clockwise
    rotated_shape = [
        [shape[y][x] for y in reversed(range(len(shape)))]
        for x in range(len(shape[0]))
    ]
    
    # Check rotation validity
    if is_valid_move(grid, rotated_shape, x, y):
        tetrimino['shape'] = rotated_shape
        #rotate_sound.play()
    return tetrimino

def rotate_tetrimino_ccw(tetrimino, grid, x, y):
    shape = tetrimino['shape']
    
    # Rotate counter-clockwise
    rotated_shape = [
        [shape[y][x] for y in range(len(shape))]
        for x in reversed(range(len(shape[0])))
    ]
    
    # Check rotation validity
    if is_valid_move(grid, rotated_shape, x, y):
        tetrimino['shape'] = rotated_shape
        #rotate_sound.play()
    return tetrimino

def hard_drop(tetrimino, x, y, grid):
    while is_valid_move(grid, tetrimino['shape'], x, y + 1):
        y += 1
    tetrimino['position'] = (x, y)
    #lock_sound.play()
    return y

def merge_tetrimino(grid, tetrimino, x, y):
    cleared_lines = 0
    for i, row in enumerate(tetrimino['shape']):
        for j, block in enumerate(row):
            if block:
                grid[y + i][x + j] = color_mapping[tetrimino['color']]
    
    # Check for line clears
    lines_to_clear = []
    for i, row in enumerate(grid):
        if all(cell != 0 for cell in row):
            lines_to_clear.append(i)
    
    # Clear lines
    for i in lines_to_clear:
        del grid[i]
        grid.insert(0, [0 for _ in range(GRID_WIDTH)])
        cleared_lines += 1
    
    # Play sound effects
    if cleared_lines == 4:
        #renis_clear_sound.play()
    elif cleared_lines > 0:
        #clear_sound.play()
    
    return cleared_lines

def main_game(agent, base_save_path, save_lock=None, player_actions=False):
    """
    Runs the main game loop where the AI agent plays Tetris.
    After each game session, it automatically restarts for continuous training.
    """
    # Ensure Pygame is initialized in this process
    pygame.init()

    # Load existing training state (if any)
    optimizer = optim.Adam(agent.gpu_model.parameters())
    generation = load_training_state(agent, base_save_path, optimizer)

    # Create/render window
    window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Ezekiel")

    # Map from key presses to action indices (unused since AI is controlling)
    key_to_action = {
        pygame.K_z: 0,      # Rotate clockwise
        pygame.K_x: 1,      # Rotate counterclockwise
        pygame.K_LEFT: 2,   # Move left
        pygame.K_RIGHT: 3,  # Move right
        pygame.K_DOWN: 4,   # Soft drop
        pygame.K_SPACE: 5,  # Hard drop
    }

    # Valid actions include a "no-op" (index 6)
    valid_actions = list(key_to_action.values()) + [6]

    print(f"\n=== Starting Generation {generation} ===")
    logging.info(f"=== Starting Generation {generation} ===")
    print(f"Memory at start of Gen {generation}: {len(agent.memory)} experiences stored.")
    logging.info(f"Memory at start of Gen {generation}: {len(agent.memory)} experiences stored.")

    # ----------------------------------------------------------------
    # Outer loop: Each iteration is ONE entire game. Once the game ends,
    # we save results, then auto-restart a new game (next generation).
    # ----------------------------------------------------------------
    while True:
        # Reset the board/game state
        grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        next_tetriminos = [generate_tetrimino() for _ in range(3)]
        current_tetrimino = next_tetriminos.pop(0)
        next_tetriminos.append(generate_tetrimino())
        x, y = current_tetrimino['position']
        tetrimino_shape = current_tetrimino['shape']
        color = current_tetrimino['color']
        key = current_tetrimino['key']

        clock = pygame.time.Clock()

        # Basic game parameters
        level = 1
        fall_speed = 500  # ms
        score = 0
        lock_timer = 0
        fall_time = 0
        lines_cleared = 0
        total_lines_cleared = 0
        rotations = 0
        action_count = 0
        game_over = False
        hard_drop_executed = False
        pieces_placed = 0  # Initialize pieces_placed to 0
        logging.debug(f"Initialized pieces_placed: {pieces_placed}")

        # If initial spawn invalid => immediate game over
        if not is_valid_move(grid, tetrimino_shape, x, y):
            print("Game Over: Invalid initial position")
            logging.warning("Game Over: Invalid initial position")
            game_over = True

        # Construct initial 5-channel state
        state = construct_state_tensor(
            grid, score, lines_cleared, pieces_placed, level, agent.device
        )

        # --------------- MAIN GAME LOOP ---------------
        while not game_over:
            dt = clock.tick(60)
            fall_time += dt

            # Handle events (e.g., quit)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # 1) AI selects action based on current state
            try:
                action = agent.select_action(state)
            except Exception as e:
                print(f"Error selecting action: {e}")
                logging.error(f"Error selecting action: {e}", exc_info=True)
                pygame.quit()
                sys.exit()

            # 2) Apply chosen action
            moved_horizontally = False
            moved_down = False
            locked = False

            if action != 6:  # Not a no-op
                if action == 0:  # Rotate CW
                    rotated_tetrimino = copy.deepcopy(current_tetrimino)
                    rotate_tetrimino(rotated_tetrimino, grid, x, y)
                    if is_valid_move(grid, rotated_tetrimino['shape'], x, y):
                        current_tetrimino['shape'] = rotated_tetrimino['shape']
                        rotations += 1
                        lock_timer = 0
                        logging.debug(f"Action {action}: Rotated CW. Rotations: {rotations}")
                elif action == 1:  # Rotate CCW
                    rotated_tetrimino = copy.deepcopy(current_tetrimino)
                    rotate_tetrimino_ccw(rotated_tetrimino, grid, x, y)
                    if is_valid_move(grid, rotated_tetrimino['shape'], x, y):
                        current_tetrimino['shape'] = rotated_tetrimino['shape']
                        rotations += 1
                        lock_timer = 0
                        logging.debug(f"Action {action}: Rotated CCW. Rotations: {rotations}")
                elif action == 2:  # Left
                    if is_valid_move(grid, current_tetrimino['shape'], x - 1, y):
                        current_tetrimino['position'] = (x - 1, y)
                        x -= 1
                        lock_timer = 0
                        moved_horizontally = True
                        logging.debug(f"Action {action}: Moved Left to position ({x}, {y})")
                elif action == 3:  # Right
                    if is_valid_move(grid, current_tetrimino['shape'], x + 1, y):
                        current_tetrimino['position'] = (x + 1, y)
                        x += 1
                        lock_timer = 0
                        moved_horizontally = True
                        logging.debug(f"Action {action}: Moved Right to position ({x}, {y})")
                elif action == 4:  # Soft drop
                    if is_valid_move(grid, current_tetrimino['shape'], x, y + 1):
                        current_tetrimino['position'] = (x, y + 1)
                        y += 1
                        fall_time = 0
                        lock_timer = 0
                        moved_down = True
                        logging.debug(f"Action {action}: Soft Dropped to position ({x}, {y})")
                elif action == 5:  # Hard drop
                    try:
                        y = hard_drop(current_tetrimino, x, y, grid)
                        locked = True
                        lock_timer = 500  # Start lock timer
                        hard_drop_executed = True
                        logging.debug(f"Action {action}: Hard Dropped to position ({x}, {y})")
                    except Exception as e:
                        print(f"Error during hard drop: {e}")
                        logging.error(f"Error during hard drop: {e}", exc_info=True)
                        pygame.quit()
                        sys.exit()

            # 3) Handle locking vs gravity
            if hard_drop_executed or not is_valid_move(grid, current_tetrimino['shape'], x, y + 1):
                lock_timer += dt
                if lock_timer >= 500:
                    try:
                        lines_cleared = merge_tetrimino(grid, current_tetrimino, x, y)
                        total_lines_cleared += lines_cleared
                        score += SCORING_RULES.get(lines_cleared, lines_cleared * SCORE_INCREMENT)
                        pieces_placed += 1  # Increment pieces_placed by 1
                        logging.debug(f"Placed piece #{pieces_placed}. Lines cleared: {lines_cleared}, Total lines: {total_lines_cleared}, Score: {score}")

                        if score >= level * LEVEL_UP_SCORE:
                            level += 1
                            fall_speed = max(MIN_FALL_SPEED, fall_speed - FALL_SPEED_DECREMENT)
                            logging.debug(f"Leveled up to Level {level}. New fall speed: {fall_speed} ms")

                        # Spawn next tetrimino
                        current_tetrimino = next_tetriminos.pop(0)
                        next_tetriminos.append(generate_tetrimino())
                        x, y = current_tetrimino['position']
                        tetrimino_shape = current_tetrimino['shape']
                        color = current_tetrimino['color']
                        key = current_tetrimino['key']

                        # Check for game over
                        if not is_valid_move(grid, tetrimino_shape, x, y):
                            game_over = True
                            logging.warning("Game Over: New tetrimino spawn position invalid.")

                        hard_drop_executed = False
                        lock_timer = 0
                        rotations = 0
                    except Exception as e:
                        print(f"Error during locking: {e}")
                        logging.error(f"Error during locking: {e}", exc_info=True)
                        pygame.quit()
                        sys.exit()
            else:
                fall_time += dt
                if fall_time > fall_speed:
                    if is_valid_move(grid, current_tetrimino['shape'], x, y + 1):
                        current_tetrimino['position'] = (x, y + 1)
                        y += 1
                    fall_time = 0

            # 4) Build temp grid for rendering
            temp_grid = [row[:] for row in grid]
            for i, row_block in enumerate(current_tetrimino['shape']):
                for j, block in enumerate(row_block):
                    if block:
                        if 0 <= y + i < GRID_HEIGHT and 0 <= x + j < GRID_WIDTH:
                            temp_grid[y + i][x + j] = color_mapping.get(current_tetrimino['color'], 0)

            # 5) Agent updates (if AI is controlling)
            try:
                reward, next_state = get_reward_and_next_state(
                    agent,
                    temp_grid,
                    lines_cleared,
                    total_lines_cleared,
                    game_over,
                    rotations,
                    moved_horizontally,
                    moved_down,
                    locked,
                    tetrimino_shape=current_tetrimino['shape'],
                    x=x,
                    y=y,
                    score=score,
                    pieces_placed=pieces_placed,
                    level=level,
                    device=agent.device
                )
                next_state = next_state.to(agent.device)
            except Exception as e:
                print(f"Error getting reward and next state: {e}")
                logging.error(f"Error getting reward and next state: {e}", exc_info=True)
                pygame.quit()
                sys.exit()

            # Debugging: Ensure next_state is a tensor
            if not isinstance(next_state, torch.Tensor):
                logging.error(f"Next state is not a torch.Tensor: {type(next_state)}")
                raise TypeError(f"Next state is not a torch.Tensor: {type(next_state)}")

            # Store transition and train if not player-controlled
            if not player_actions:
                try:
                    agent.store_transition(state, action, reward, next_state, game_over)
                    agent.train_model()
                except Exception as e:
                    print(f"Error storing transition or training model: {e}")
                    logging.error(f"Error storing transition or training model: {e}", exc_info=True)
                    pygame.quit()
                    sys.exit()

            # Update state
            state = next_state
            lines_cleared = 0  # Reset lines cleared

            # 6) Render
            try:
                window.fill(WHITE)
                draw_grid(temp_grid, window)
                draw_score(window, score)
                draw_level(window, level)
                draw_next_tetrimino(window, next_tetriminos[0]['key'])
                pygame.display.flip()
            except Exception as e:
                print(f"Error during rendering: {e}")
                logging.error(f"Error during rendering: {e}", exc_info=True)
                pygame.quit()
                sys.exit()

            action_count += 1
            if action_count % 50 == 0:
                print(f"Action {action_count} | Score={score} | Reward={reward} | Lines={total_lines_cleared}")
                logging.info(f"Action {action_count} | Score={score} | Reward={reward} | Lines={total_lines_cleared}")

        # -------------- GAME OVER --------------
        print(f"Game Over - Score {score}, Lines Cleared {total_lines_cleared}, Pieces Placed {pieces_placed}")
        logging.info(f"Game Over - Score {score}, Lines Cleared {total_lines_cleared}, Pieces Placed {pieces_placed}")

        # Save training state
        try:
            if save_lock:
                save_training_state(agent, optimizer, generation, base_save_path, save_lock, force_save=True)
            else:
                save_training_state(agent, optimizer, generation, base_save_path, force_save=True)
        except Exception as e:
            print(f"Error saving training state: {e}")
            logging.error(f"Error saving training state: {e}", exc_info=True)

        # snapshot every 10 generations
        if generation % 10 == 0:
            try:
                gen_ckpt = f"{base_save_path.split('.pth')[0]}_gen{generation}.pth"
                save_training_state(agent, optimizer, generation, gen_ckpt, save_lock)
                print(f"Snapshot saved at {gen_ckpt}")
                logging.info(f"Snapshot saved at {gen_ckpt}")
            except Exception as e:
                print(f"Error saving snapshot: {e}")
                logging.error(f"Error saving snapshot: {e}", exc_info=True)

        # Auto-restart next generation
        generation += 1
        agent.decay_epsilon()  # Decay epsilon after each generation
        print(f"=== Auto-restarting for Generation {generation} ===\n")
        logging.info(f"=== Auto-restarting for Generation {generation} ===\n")

def find_latest_valid_checkpoint(base_save_path):
    checkpoint_dir = os.path.dirname(base_save_path)
    if checkpoint_dir == "":
        checkpoint_dir = "."

    checkpoint_pattern = re.compile(r'_gen(\d+)\.pth$')
    checkpoints = []
    
    for filename in os.listdir(checkpoint_dir):
        match = checkpoint_pattern.search(filename)
        if match:
            generation = int(match.group(1))
            checkpoint_path = os.path.join(checkpoint_dir, filename)
            result = check_checkpoint_integrity(checkpoint_path)
            if result:
                checkpoints.append(result)
    
    if checkpoints:
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        latest_generation, latest_checkpoint = checkpoints[0]
        print(f"Loading the latest valid checkpoint from generation {latest_generation}: {latest_checkpoint}")
        logging.info(f"Loading the latest valid checkpoint from generation {latest_generation}: {latest_checkpoint}")
        return latest_checkpoint
    else:
        print("No valid checkpoints found. Starting from scratch.")
        logging.info("No valid checkpoints found. Starting from scratch.")
        return None

def check_checkpoint_integrity(checkpoint_path):
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path)
            required_keys = ['model_state_dict', 'optimizer_state_dict', 'generation', 'memory']
            if all(key in checkpoint for key in required_keys):
                print(f"Checkpoint {checkpoint_path} is valid with generation {checkpoint['generation']}.")
                logging.info(f"Checkpoint {checkpoint_path} is valid with generation {checkpoint['generation']}.")
                return checkpoint['generation'], checkpoint_path
            else:
                print(f"Checkpoint {checkpoint_path} is missing required keys.")
                logging.warning(f"Checkpoint {checkpoint_path} is missing required keys.")
                return None
        except Exception as e:
            print(f"Failed to load checkpoint {checkpoint_path}: {e}")
            logging.error(f"Failed to load checkpoint {checkpoint_path}: {e}", exc_info=True)
            return None
    else:
        print(f"Checkpoint {checkpoint_path} does not exist.")
        logging.warning(f"Checkpoint {checkpoint_path} does not exist.")
        return None

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize agent with correct parameters
    agent = TetrisAgent(input_height=20, input_width=10, action_size=6, sync_frequency=4)
    agent.device = device
    agent.gpu_model.to(device)

    # Define checkpoint path
    base_save_path = "trained_bfs_model.pth"

    # Attempt to load the latest valid checkpoint
    latest_checkpoint = find_latest_valid_checkpoint(base_save_path)
    if latest_checkpoint:
        try:
            # Re-initialize optimizer to match loaded state
            optimizer = optim.Adam(agent.gpu_model.parameters())
            generation = load_training_state(agent, latest_checkpoint[1], optimizer)
        except Exception as e:
            print(f"Error loading training state: {e}")
            logging.error(f"Error loading training state: {e}", exc_info=True)
            generation = 0
    else:
        generation = 0  # Start from scratch if no checkpoint is found
        optimizer = optim.Adam(agent.gpu_model.parameters())  # Initialize optimizer

    # Run supervised BFS training if needed
    supervised_train_from_bfs(
     agent,
     bfs_data_path="training_data.pkl",  # Path to your BFS-collected data
     epochs=5,
     batch_size=32
    )

    # Start the main game loop with continuous training
    try:
        main_game(agent, base_save_path)
    except Exception as e:
        print(f"An error occurred during the game loop: {e}")
        traceback.print_exc()  # Print full traceback
        logging.error(f"An error occurred during the game loop: {e}", exc_info=True)
        pygame.quit()
        sys.exit()

    # After game over, save the model
    try:
        agent.save_model(base_save_path)
        print(f"Model parameters saved to {base_save_path}")
        logging.info(f"Model parameters saved to {base_save_path}")
    except Exception as e:
        print(f"Error saving model after gameplay: {e}")
        logging.error(f"Error saving model after gameplay: {e}", exc_info=True)