import pygame
import sys
import random
import collections
from collections import deque
from tetris_agent import TetrisAgent, TetrisCNN  # Import the agent and neural network
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
import glob
import matplotlib.pyplot as plt

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
        
def extract_path(base_save_path):
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

def load_training_state(agent, checkpoint_path):
    generation = agent.load_model(checkpoint_path, agent.device)
    if generation > 0:
        logging.info(f"Successfully loaded training state from generation {generation}.")
    else:
        logging.info("Starting training from scratch.")
    return generation
    
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
    'complete_lines': 300,      # Positive reward for lines cleared
    'aggregate_height': -40,    # Negative reward for higher aggregate height
    'holes': -150,              # Negative reward for holes
    'bumpiness': -15,           # Negative reward for bumpiness
    'flat_bonus': 10            # Positive reward for flatness
}

def get_center_penalty(x):
    """Calculate penalty based on the x-position to encourage central stacking."""
    center = GRID_WIDTH / 2
    distance_from_center = abs(x - center)
    penalty = distance_from_center * 2  # Adjust multiplier as needed
    return penalty

def get_height_bonus(tetrimino_shape, y):
    """Calculate bonus based on how low the piece is placed."""
    piece_height = len(tetrimino_shape)
    piece_bottom_y = y + piece_height
    height_from_bottom = (GRID_HEIGHT - piece_bottom_y) / GRID_HEIGHT
    height_bonus = (1 - height_from_bottom) * 2000
    print(f"Height bonus: {height_bonus} for y={y}, bottom_y={piece_bottom_y}")
    return height_bonus

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
    reward = 0
    
    if rotations > 3:
        penalty = (rotations - 3) * 1000
        reward -= penalty
        #print(f"[DEBUG] Rotation penalty => -{penalty}")
        
    # Small penalty each time we move or rotate
    if moved_horizontally or moved_down or locked or rotations > 0:
        reward -= 0.1

    if locked:
        # -----------------------------------------------------------
        # 1) Make a copy of the grid, remove the newly locked piece
        #    so we don't incorrectly count holes "above" it.
        # -----------------------------------------------------------
        temp_grid = [row[:] for row in grid]
        if tetrimino_shape:
            for i, row_data in enumerate(tetrimino_shape):
                for j, block in enumerate(row_data):
                    if block:
                        board_r = y + i
                        board_c = x + j
                        if 0 <= board_r < GRID_HEIGHT and 0 <= board_c < GRID_WIDTH:
                            # Remove this piece block from the copy
                            temp_grid[board_r][board_c] = 0

        # -----------------------------------------------------------
        # 2) Compute hole-related stats on the temp_grid (no piece)
        # -----------------------------------------------------------
        holes = count_holes(temp_grid)
        before_max_height, before_aggregate_height, before_col_heights = get_max_height_and_aggregate_height(temp_grid)
        bumpiness = get_bumpiness(before_col_heights)

        # Height bonus for how low we placed the piece
        height_bonus = get_height_bonus(tetrimino_shape, y)
        reward += height_bonus

        contact_points = count_contact_points(grid, tetrimino_shape, x, y)
        side_touches = count_side_touches(grid, tetrimino_shape, x, y)
        reward += contact_points * 50
        reward += side_touches * 30

        if contact_points >= 3 and side_touches >= 2:
            reward += 300
        if contact_points == 0:
            reward -= 100

        # Edge tower detection logic on temp_grid
        edge_height_threshold = int(GRID_HEIGHT * 0.75)
        left_height, left_holes = analyze_edge_column(temp_grid, 0)
        right_height, right_holes = analyze_edge_column(temp_grid, GRID_WIDTH - 1)

        if left_height > edge_height_threshold:
            height_penalty = ((left_height - edge_height_threshold) ** 2) * 100
            hole_penalty = left_holes * left_height * 25
            total_penalty = height_penalty + hole_penalty
            reward -= total_penalty

        if right_height > edge_height_threshold:
            height_penalty = ((right_height - edge_height_threshold) ** 2) * 100
            hole_penalty = right_holes * right_height * 25
            total_penalty = height_penalty + hole_penalty
            reward -= total_penalty

        # Additional height penalty
        height_penalty = 0
        for col_height in before_col_heights:
            if col_height > GRID_HEIGHT * 0.6:
                height_penalty += (col_height - GRID_HEIGHT * 0.6) * 5
            if col_height > GRID_HEIGHT * 0.8:
                height_penalty += (col_height - GRID_HEIGHT * 0.8) * 10
        reward -= height_penalty

        # Penalties for holes and bumpiness
        reward -= bumpiness * 10
        reward -= holes * 100

        print(f"Locked piece - Height bonus: {height_bonus}, Holes: {holes}, Bumpiness: {bumpiness}")

    # Reward for lines cleared
    if lines_cleared > 0:
        line_rewards = {1: 1000, 2: 2500, 3: 5000, 4: 8000}
        reward += line_rewards.get(lines_cleared, lines_cleared * 100)
        # Bonus if our max height is still below half the board
        if before_max_height < GRID_HEIGHT * 0.5:
            reward += lines_cleared * 50

    # Game over penalty
    if game_over:
        reward -= 500

    # Construct next state
    next_state = construct_state_tensor(grid, score, lines_cleared, pieces_placed, level, device)
    if locked:
        print(f"Final reward: {reward}")
    return reward, next_state

def get_height_bonus(tetrimino_shape, y):
    """Calculate bonus based on how low the piece is placed."""
    piece_height = len(tetrimino_shape)
    piece_bottom_y = y + piece_height  # Get the bottom of the piece
    height_from_bottom = (GRID_HEIGHT - piece_bottom_y) / GRID_HEIGHT
    height_bonus = (1 - height_from_bottom) * 500  # Increased bonus for lower placement
    print(f"Height bonus: {height_bonus} for y={y}, bottom_y={piece_bottom_y}")  # Debug print
    return height_bonus
    
def analyze_edge_column(grid, col):
    """Analyze a column for height and holes."""
    first_block = None
    holes = 0
    
    # Find first block from bottom
    for row in range(GRID_HEIGHT - 1, -1, -1):
        if grid[row][col] != 0:
            first_block = row
            break
    
    if first_block is None:
        return 0, 0  # Empty column
        
    height = GRID_HEIGHT - first_block
    
    # Count holes (empty spaces below blocks)
    has_block_above = False
    for row in range(GRID_HEIGHT - 1, -1, -1):
        if grid[row][col] != 0:
            has_block_above = True
        elif has_block_above:
            holes += 1
            
    return height, holes

def measure_edge_tower(col_heights, edge='left'):
    """Measure the height of towers along the edges."""
    if edge == 'left':
        idx = 0
    else:  # right
        idx = len(col_heights) - 1
        
    # Get adjacent column height
    adjacent_idx = 1 if edge == 'left' else len(col_heights) - 2
    height_diff = col_heights[idx] - col_heights[adjacent_idx]
    
    # Only count as tower if significantly higher than adjacent column
    return max(0, height_diff) if height_diff > 2 else 0

def count_contact_points(grid, tetrimino_shape, x, y):
    """Count number of bottom contact points for a piece."""
    contacts = 0
    for i, row in enumerate(tetrimino_shape):
        for j, block in enumerate(row):
            if block:
                # Check if there's a block below or it's the bottom of the grid
                if (y + i + 1 >= GRID_HEIGHT or 
                    (y + i + 1 < GRID_HEIGHT and grid[y + i + 1][x + j] != 0)):
                    contacts += 1
    return contacts

def count_side_touches(grid, tetrimino_shape, x, y):
    """Count number of side contacts with existing pieces."""
    touches = 0
    for i, row in enumerate(tetrimino_shape):
        for j, block in enumerate(row):
            if block:
                # Check left side
                if x + j > 0 and grid[y + i][x + j - 1] != 0:
                    touches += 1
                # Check right side
                if x + j < GRID_WIDTH - 1 and grid[y + i][x + j + 1] != 0:
                    touches += 1
    return touches

# -------------- Helper Functions ---------------

def print_grid(grid):
    print("\n--- Current Grid State ---")
    for row in grid:
        print(' '.join(str(cell) for cell in row))
    print("---------------------------\n")

def count_holes(board, shape=None, x=0, y=0):
    """
    Remove the current piece from the board copy, then count holes.
    """
    temp_board = [row[:] for row in board]  # make copy
    
    # Remove the current piece’s blocks from temp_board
    if shape is not None:
        for i, row_data in enumerate(shape):
            for j, block in enumerate(row_data):
                if block:
                    # Check boundaries, then remove
                    br = y + i
                    bc = x + j
                    if 0 <= br < GRID_HEIGHT and 0 <= bc < GRID_WIDTH:
                        temp_board[br][bc] = 0
    
    # Now do your hole logic on temp_board instead of board
    return count_holes_on_clean_board(temp_board)

def count_holes_excluding_piece(board, shape, x, y):
    """
    Makes a copy of 'board', removes the current piece from that copy,
    then counts holes in the copy.
    """
    # 1) Copy the board so we don't affect the real one
    temp_board = [row[:] for row in board]
    
    # 2) Remove the current (falling) piece from temp_board
    #    so it doesn't create 'artificial' holes.
    for i, row_data in enumerate(shape):
        for j, block in enumerate(row_data):
            if block:
                board_r = y + i
                board_c = x + j
                if 0 <= board_r < GRID_HEIGHT and 0 <= board_c < GRID_WIDTH:
                    temp_board[board_r][board_c] = 0

    # 3) Now count holes (top-down or BFS) on temp_board
    return count_holes_top_down(temp_board)

def count_holes_top_down(clean_board):
    """
    Standard top-down hole count:
    Once we find a block in a column, every subsequent empty cell below it is a hole.
    """
    holes = 0
    for col in range(GRID_WIDTH):
        found_block_above = False
        for row in range(GRID_HEIGHT):  # row=0 is top
            if clean_board[row][col] != 0:
                found_block_above = True
            elif found_block_above and clean_board[row][col] == 0:
                holes += 1
    return holes

def count_holes_on_clean_board(clean_board):
    """
    Standard 'top-down' hole count: 
    Once we see a block in a column, 
    every subsequent empty cell below it is a hole.
    """
    holes = 0
    for col in range(GRID_WIDTH):
        found_block_above = False
        for row in range(GRID_HEIGHT):  # row=0 is top
            if clean_board[row][col] != 0:
                found_block_above = True
            elif found_block_above and clean_board[row][col] == 0:
                holes += 1
    return holes
    
def get_max_height_and_aggregate_height(board, current_tetrimino=None, tetrimino_x=0, tetrimino_y=0):
    """Returns the maximum column height and aggregate height of locked pieces only."""
    # Create a copy of the board to work with
    temp_board = [row[:] for row in board]
    
    # Remove current tetrimino from height calculation if provided
    if current_tetrimino and tetrimino_x is not None and tetrimino_y is not None:
        for i, row in enumerate(current_tetrimino):
            for j, block in enumerate(row):
                if block and 0 <= tetrimino_y + i < GRID_HEIGHT and 0 <= tetrimino_x + j < GRID_WIDTH:
                    temp_board[tetrimino_y + i][tetrimino_x + j] = 0

    col_heights = []
    for col in range(GRID_WIDTH):
        h = 0
        for row in reversed(range(GRID_HEIGHT)):
            if temp_board[row][col] != 0:
                h = GRID_HEIGHT - row
                break
        col_heights.append(h)
        
    max_height = max(col_heights) if col_heights else 0
    aggregate_height = sum(col_heights)
    
    return max_height, aggregate_height, col_heights
    
def get_bumpiness(col_heights):
    """Sum of absolute differences between adjacent columns."""
    bumpiness = 0
    for i in range(len(col_heights)-1):
        bumpiness += abs(col_heights[i] - col_heights[i+1])
    return bumpiness

def detect_wells(col_heights):
    """Detect useful wells in the column heights."""
    wells = []
    for i in range(1, len(col_heights) - 1):
        if (col_heights[i] + 2 <= col_heights[i-1] and 
            col_heights[i] + 2 <= col_heights[i+1]):
            wells.append(i)
    return wells

# Initialize Pygame
pygame.init()
pygame.mixer.init()

# Constants
pygame.display.set_caption("Ezekiel")  # Set window title to "Ezekiel"
WINDOW_WIDTH = 300
WINDOW_HEIGHT = 600
BLOCK_SIZE = 30

GRID_WIDTH = 10
GRID_HEIGHT = 20

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
    
    return cleared_lines

def main_game(agent, base_save_path, save_lock=None, player_actions=False, rewards_list=None):
    """
    Runs a single game session where the AI agent plays Tetris.
    After the game ends, the function returns control to the main script.
    
    Args:
        rewards_list (list, optional): If provided, rewards will be appended to this list
    """
    print("[DEBUG] Starting a new game session.")
    logging.debug("Starting a new game session.")
    
    # Reset the board/game state
    grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    next_tetriminos = [generate_tetrimino() for _ in range(3)]
    current_tetrimino = next_tetriminos.pop(0)
    next_tetriminos.append(generate_tetrimino())
    x, y = current_tetrimino['position']
    tetrimino_shape = current_tetrimino['shape']
    color = current_tetrimino['color']
    key = current_tetrimino['key']

    # Reset the D action flag for the new block
    d_action_processed = False

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
    pieces_placed = 0
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

    print("[DEBUG] Entering the main game loop.")
    logging.debug("Entering the main game loop.")

    # --------------- MAIN GAME LOOP ---------------
    while not game_over:
        try:
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
                        lock_timer = 500
                        logging.debug(f"Action {action}: Rotated CW. Rotations: {rotations}")
                elif action == 1:  # Rotate CCW
                    rotated_tetrimino = copy.deepcopy(current_tetrimino)
                    rotate_tetrimino_ccw(rotated_tetrimino, grid, x, y)
                    if is_valid_move(grid, rotated_tetrimino['shape'], x, y):
                        current_tetrimino['shape'] = rotated_tetrimino['shape']
                        rotations += 1
                        lock_timer = 500
                        logging.debug(f"Action {action}: Rotated CCW. Rotations: {rotations}")
                elif action == 2:  # Left
                    if is_valid_move(grid, current_tetrimino['shape'], x - 1, y):
                        current_tetrimino['position'] = (x - 1, y)
                        x -= 1
                        lock_timer = 500
                        moved_horizontally = True
                        logging.debug(f"Action {action}: Moved Left to position ({x}, {y})")
                elif action == 3:  # Right
                    if is_valid_move(grid, current_tetrimino['shape'], x + 1, y):
                        current_tetrimino['position'] = (x + 1, y)
                        x += 1
                        lock_timer = 500
                        moved_horizontally = True
                        logging.debug(f"Action {action}: Moved Right to position ({x}, {y})")
                elif action == 4:  # Soft drop
                    if is_valid_move(grid, current_tetrimino['shape'], x, y + 1):
                        current_tetrimino['position'] = (x, y + 1)
                        y += 1
                        fall_time = 0
                        lock_timer = 500
                        moved_down = True
                        logging.debug(f"Action {action}: Soft Dropped to position ({x}, {y})")
                    else:
                        lock_timer += 500
                        logging.debug(f"Action {action}: Soft drop attempted but cannot move down. Initiating lock.")

            # 3) Handle locking vs gravity
            if hard_drop_executed or not is_valid_move(grid, current_tetrimino['shape'], x, y + 1):
                lock_timer += dt
                if lock_timer >= 500:
                    try:
                        lines_cleared = merge_tetrimino(grid, current_tetrimino, x, y)
                        total_lines_cleared += lines_cleared
                        score += SCORING_RULES.get(lines_cleared, lines_cleared * SCORE_INCREMENT)
                        pieces_placed += 1
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

                        # Reset the D action flag for the new block
                        d_action_processed = False

                        # Check for game over
                        if not is_valid_move(grid, tetrimino_shape, x, y):
                            game_over = True
                            logging.warning("Game Over: New tetrimino spawn position invalid.")

                        hard_drop_executed = False
                        lock_timer = 0
                        rotations = 0
                        
                        locked = True
                        
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
                
                if rewards_list is not None:
                    rewards_list.append(reward)
                    
            except Exception as e:
                print(f"Error getting reward and next state: {e}")
                logging.error(f"Error getting reward and next state: {e}", exc_info=True)
                pygame.quit()
                sys.exit()

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
        except Exception as e:
            print(f"Error: {e}")
    print("[DEBUG] Game session ended. Exiting main_game.")
    logging.debug("Game session ended. Exiting main_game.")
    
def find_latest_valid_checkpoint(checkpoint_dir):
    """
    Finds and returns the path to the latest valid checkpoint.

    Returns:
    - tuple: (generation_number, checkpoint_path)
    - None: If no valid checkpoints are found.
    """
    checkpoint_pattern = re.compile(r'trained_bfs_model_gen(\d+)\.pth$')
    checkpoints = []

    # If directory doesn't exist or is invalid:
    if not os.path.isdir(checkpoint_dir):
        print(f"[DEBUG] Checkpoint directory does not exist or is not a directory: {checkpoint_dir}")
        logging.warning(f"Checkpoint directory does not exist or is not a directory: {checkpoint_dir}")
        return None

    try:
        for filename in os.listdir(checkpoint_dir):
            match = checkpoint_pattern.match(filename)
            if match:
                generation = int(match.group(1))
                checkpoint_path = os.path.join(checkpoint_dir, filename)
                if check_checkpoint_integrity(checkpoint_path):
                    checkpoints.append((generation, checkpoint_path))
    except Exception as e:
        print(f"[DEBUG] Error accessing checkpoint directory {checkpoint_dir}: {e}")
        logging.error(f"Error accessing checkpoint directory {checkpoint_dir}: {e}", exc_info=True)
        return None

    if not checkpoints:
        print("[DEBUG] No valid checkpoints found.")
        logging.info("No valid checkpoints found.")
        return None

    # Sort by generation descending
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    latest_generation, latest_checkpoint = checkpoints[0]
    print(f"[DEBUG] Latest checkpoint: Generation {latest_generation}, Path: {latest_checkpoint}")
    logging.info(f"Latest checkpoint: Generation {latest_generation}, Path: {latest_checkpoint}")

    # Return a 2-element tuple for easy unpacking
    return (latest_generation, latest_checkpoint)
    
def check_checkpoint_integrity(checkpoint_path):
    """
    Checks whether the checkpoint file contains all required keys.
    Returns True if valid, False otherwise.
    """
    if not os.path.exists(checkpoint_path):
        print(f"[DEBUG] Checkpoint file {checkpoint_path} does not exist.")
        logging.warning(f"Checkpoint file {checkpoint_path} does not exist.")
        return False

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"[DEBUG] Failed to load checkpoint {checkpoint_path}: {e}")
        logging.error(f"Failed to load checkpoint {checkpoint_path}: {e}", exc_info=True)
        return False

    required_keys = ['model_state_dict', 'optimizer_state_dict', 'generation', 'memory']
    missing_keys = [key for key in required_keys if key not in checkpoint]

    if missing_keys:
        print(f"[DEBUG] Checkpoint {checkpoint_path} is missing required keys: {missing_keys}")
        logging.warning(f"Checkpoint {checkpoint_path} is missing required keys: {missing_keys}")
        return False

    print(f"[DEBUG] Checkpoint {checkpoint_path} is valid with generation {checkpoint['generation']}.")
    logging.info(f"Checkpoint {checkpoint_path} is valid with generation {checkpoint['generation']}.")
    return True
    
def cleanup_old_checkpoints(directory, pattern, keep=10):
    try:
        checkpoints = sorted(glob.glob(os.path.join(directory, pattern)), key=os.path.getmtime, reverse=True)
        for ckpt in checkpoints[keep:]:
            try:
                os.remove(ckpt)
                logging.info(f"Removed old checkpoint: {ckpt}")
                print(f"[DEBUG] Removed old checkpoint: {ckpt}")
            except Exception as e:
                logging.error(f"Failed to remove old checkpoint {ckpt}: {e}")
                print(f"[DEBUG] Failed to remove old checkpoint {ckpt}: {e}")
    except Exception as e:
        print(f"[DEBUG] Error during cleanup of checkpoints: {e}")
        logging.error(f"Error during cleanup of checkpoints: {e}", exc_info=True)
        
def get_latest_checkpoint_path(checkpoint_dir):
    """
    Retrieves the path of the latest checkpoint in the given directory.
    
    Parameters:
    - checkpoint_dir (str): Directory where checkpoints are stored.
    
    Returns:
    - tuple: (generation_number, checkpoint_path)
    - None: If no checkpoints are found.
    """
    checkpoint_pattern = re.compile(r'trained_bfs_model_gen(\d+)\.pth$')
    checkpoints = []
    
    if not os.path.exists(checkpoint_dir):
        print(f"[DEBUG] Checkpoint directory does not exist: {checkpoint_dir}")
        logging.warning(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return None
    
    for filename in os.listdir(checkpoint_dir):
        match = checkpoint_pattern.search(filename)
        if match:
            generation = int(match.group(1))
            checkpoint_path = os.path.join(checkpoint_dir, filename)
            if check_checkpoint_integrity(checkpoint_path):
                checkpoints.append((generation, checkpoint_path))
    
    if checkpoints:
        # Sort checkpoints by generation number in descending order
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        latest_generation, latest_checkpoint = checkpoints[0]
        print(f"[DEBUG] Latest checkpoint: Generation {latest_generation}, Path: {latest_checkpoint}")
        logging.info(f"Latest checkpoint: Generation {latest_generation}, Path: {latest_checkpoint}")
        return latest_checkpoint
    else:
        print("[DEBUG] No valid checkpoints found.")
        logging.info("No valid checkpoints found.")
        return None

def plot_rewards(rewards, generation, save_dir):
    """Plot and save rewards trend."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title(f'Rewards Over Time (Up to Generation {generation})')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'rewards_gen_{generation}.png'))
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = TetrisAgent(input_height=20, input_width=10, action_size=6, sync_frequency=4)
    agent.device = device
    agent.gpu_model.to(device)
    
    checkpoint_dir = os.path.abspath("checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize rewards list
    all_rewards = []

    # Find the latest valid checkpoint
    checkpoint_info = find_latest_valid_checkpoint(checkpoint_dir)
    if checkpoint_info:
        generation_number, checkpoint_path = checkpoint_info
        try:
            generation = load_training_state(agent, checkpoint_path)
            print(f"[DEBUG] Loaded checkpoint from generation {generation}")
        except Exception as e:
            print(f"Error loading training state: {e}")
            logging.error(f"Error loading training state: {e}", exc_info=True)
            generation = 0
            agent.optimizer = optim.Adam(agent.gpu_model.parameters())
    else:
        generation = 0
        agent.optimizer = optim.Adam(agent.gpu_model.parameters())
        print("[DEBUG] No checkpoint found. Starting from scratch.")

    TOTAL_GENERATIONS = 2000
    for gen in range(generation + 1, TOTAL_GENERATIONS + 1):
        print(f"\n=== Starting Generation {gen} ===")
        logging.info(f"=== Starting Generation {gen} ===")
        
        # Run game and collect rewards
        try:
            episode_rewards = []
            main_game(agent, checkpoint_dir, rewards_list=episode_rewards)
            all_rewards.extend(episode_rewards)
        except Exception as e:
            print(f"[DEBUG] An error occurred during the game loop: {e}")
            traceback.print_exc()
            logging.error(f"An error occurred during the game loop: {e}", exc_info=True)
            pygame.quit()
            sys.exit()
        
        # Save model and plot rewards every 10 generations
        if gen % 10 == 0:
            try:
                # Save model
                new_generation = gen
                model_save_path = os.path.join(checkpoint_dir, f"trained_bfs_model_gen{new_generation}.pth")
                agent.save_model(model_save_path, new_generation)
                
                # Plot rewards
                plot_rewards(all_rewards, new_generation, os.path.dirname(model_save_path))
                
                print(f"[DEBUG] Model and rewards plot saved at generation {new_generation}")
                logging.info(f"Model and rewards plot saved at generation {new_generation}")
                
            except Exception as e:
                print(f"[DEBUG] Error saving model or plotting rewards: {e}")
                logging.error(f"Error saving model or plotting rewards: {e}", exc_info=True)
        
        # Cleanup old checkpoints
        cleanup_old_checkpoints(checkpoint_dir, 'trained_bfs_model_gen*.pth', keep=10)
        
if __name__ == "__main__":
    main()
