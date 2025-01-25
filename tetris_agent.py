import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pickle
import numpy as np
import os
import logging
import random  # Added for sampling transitions

# ----------------- LOGGING SETUP -----------------
logging.basicConfig(
    filename='tetris_nn_training.log',
    filemode='a',  # Changed to append mode to preserve logs across runs
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ----------------- NEURAL NETWORK ARCHITECTURE -----------------
class TetrisCNN(nn.Module):
    def __init__(self, input_channels=5, action_size=6):
        super(TetrisCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 128, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(512 * 2 * 1, 1024)  # Adjusted based on conv output
        self.fc2 = nn.Linear(1024, action_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ----------------- AGENT CLASS -----------------
class TetrisAgent:
    def __init__(self, input_height=20, input_width=10, action_size=6, sync_frequency=4):
        self.action_size = action_size
        self.sync_frequency = sync_frequency
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize TetrisCNN with input_channels=5 to match state tensor
        self.gpu_model = TetrisCNN(input_channels=5, action_size=action_size).to(self.device)
        self.memory = deque(maxlen=2000)
        self.optimizer = optim.Adam(self.gpu_model.parameters(), lr=1e-3)
        self.loss_fn = nn.CrossEntropyLoss()
        self.epsilon = 1.0  # Starting with full exploration
        self.epsilon_min = 0.1  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate per episode
        logging.info("Initialized TetrisAgent with device: {}".format(self.device))
    
    def select_action(self, state):
        """Select an action based on the current state using an epsilon-greedy policy."""
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_size - 1)
            action_name = {0: 'CW', 1: 'CCW', 2: 'L', 3: 'R', 4: 'D', 5: 'SPACE'}.get(action, 'NOP')
            logging.debug(f"Randomly selected Action: {action} ({action_name})")
            return action
        else:
            self.gpu_model.eval()
            state = state.to(self.device)
            with torch.no_grad():
                output = self.gpu_model(state.unsqueeze(0))  # Add batch dimension
                probabilities = torch.softmax(output, dim=1)
                action = torch.argmax(probabilities, dim=1).item()
                action_name = {0: 'CW', 1: 'CCW', 2: 'L', 3: 'R', 4: 'D', 5: 'SPACE'}.get(action, 'NOP')
                logging.debug(f"Selected Action: {action} ({action_name}) with probabilities {probabilities.cpu().numpy()}")
                return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store the transition in memory."""
        # Type Assertions to ensure correct types
        assert isinstance(state, torch.Tensor), f"State must be a torch.Tensor, got {type(state)}"
        assert isinstance(action, int), f"Action must be an int, got {type(action)}"
        assert isinstance(reward, float) or isinstance(reward, int), f"Reward must be a float or int, got {type(reward)}"
        assert isinstance(next_state, torch.Tensor), f"Next state must be a torch.Tensor, got {type(next_state)}"
        assert isinstance(done, bool), f"Done must be a bool, got {type(done)}"
        
        self.memory.append((state, action, reward, next_state, done))
        logging.debug(f"Stored transition: Action={action}, Reward={reward}, Done={done}")
    
    def train_model(self, batch_size=32, gamma=0.99):
        """Train the model using a batch of transitions from memory."""
        if len(self.memory) < batch_size:
            logging.debug("Not enough samples to train. Current memory size: {}".format(len(self.memory)))
            return  # Not enough samples to train
        
        # Sample a random batch of transitions
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors and perform training steps
        states = torch.stack(states)  # Shape: (B, 5, 20, 10)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)  # Shape: (B,)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)  # Shape: (B,)
        next_states = torch.stack(next_states)  # Shape: (B, 5, 20, 10)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)  # Shape: (B,)
        
        # Compute current Q values
        current_q = self.gpu_model(states).gather(1, actions.unsqueeze(1)).squeeze(1)  # Shape: (B,)
        
        # Compute target Q values
        with torch.no_grad():
            next_q = self.gpu_model(next_states).max(1)[0]
            target_q = rewards + gamma * next_q * (1 - dones)
        
        # Compute loss
        loss = self.loss_fn(current_q, target_q)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        logging.debug(f"Trained model on batch with loss: {loss.item()}")
        
        # Decay epsilon
        self.decay_epsilon()
    
    def decay_epsilon(self):
        """Decay the exploration rate."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
            logging.debug(f"Decayed epsilon to {self.epsilon}")
    
    def save_model(self, path, generation):
        """Save the model's state, optimizer state, generation, and memory."""
        try:
            checkpoint = {
                'model_state_dict': self.gpu_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'generation': generation,
                'memory': list(self.memory)  # Convert deque to list for serialization
            }
            torch.save(checkpoint, path)
            logging.info(f"Full checkpoint saved to {path} at generation {generation}")
            print(f"[DEBUG] Full checkpoint saved to {path} at generation {generation}")
        except Exception as e:
            logging.error(f"Failed to save checkpoint to {path}: {e}")
            print(f"[DEBUG] Failed to save checkpoint to {path}: {e}")
    
    def load_model(self, path, device):
        """Load the full checkpoint including model, optimizer, generation, and memory."""
        if os.path.exists(path):
            try:
                # Remove weights_only=True to load the full checkpoint
                checkpoint = torch.load(path, map_location=device)
                required_keys = ['model_state_dict', 'optimizer_state_dict', 'generation', 'memory']
                if all(key in checkpoint for key in required_keys):
                    self.gpu_model.load_state_dict(checkpoint['model_state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.memory = deque(checkpoint['memory'], maxlen=2000)
                    generation = checkpoint['generation']
                    self.gpu_model.to(device)
                    self.gpu_model.eval()
                    logging.info(f"Full checkpoint loaded from {path} at generation {generation}")
                    print(f"[DEBUG] Full checkpoint loaded from {path} at generation {generation}")
                    return generation
                else:
                    missing = [key for key in required_keys if key not in checkpoint]
                    logging.warning(f"Checkpoint {path} missing keys: {missing}")
                    print(f"[DEBUG] Checkpoint {path} missing keys: {missing}")
                    return 0
            except Exception as e:
                logging.error(f"Failed to load checkpoint from {path}: {e}")
                print(f"[DEBUG] Failed to load checkpoint from {path}: {e}")
                return 0
        else:
            logging.error(f"Checkpoint file {path} does not exist.")
            print(f"[DEBUG] Checkpoint file {path} does not exist.")
            return 0
