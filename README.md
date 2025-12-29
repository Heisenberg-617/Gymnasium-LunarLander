# Deep Q-Learning for Lunar Lander

## Overview
This project implements a **Deep Q-Network (DQN)** agent to solve the OpenAI Gym **LunarLander-v3** environment. The agent learns to control the lander by choosing optimal actions based on the current state of the environment using reinforcement learning techniques.

The project demonstrates:
- Reinforcement Learning with **Deep Q-Learning**.
- Experience replay and target networks for stable training.
- Visualization of the trained agent.

## Features
- **Deep Q-Network (DQN)** with two neural networks (local and target) to approximate Q-values.
- **Experience Replay**: stores past experiences to break correlations between consecutive updates.
- **Epsilon-Greedy Policy**: balances exploration and exploitation during training.
- **GPU Support**: Automatically uses GPU if available.
- **Video Rendering**: Generates a video of the agent performing in the environment.

## Project Structure
- `dqn_lunar_lander.ipynb`: Jupyter Notebook containing the full implementation.
- `Network` class: Neural network architecture for Q-value approximation.
- `ReplayMemory` class: Experience replay buffer.
- `Agent` class: Handles action selection, learning, and network updates.
- Video generation scripts to visualize the trained agent.

## Requirements
```bash
pip install gymnasium
pip install "gymnasium[atari, accept-rom-license]"
pip install gymnasium[box2d]
pip install torch torchvision torchaudio
pip install imageio
```

## Usage

1. Clone the repository.
2. Run the notebook or Python script to train the agent:
```bash
python dqn_lunar_lander.py
```
3. After training, a checkpoint.pth file will be saved with the trained model.
4. Visualize the agent's performance using the built-in video rendering function.

## Key Hyperparameters
-Learning rate: 5e-4
-Discount factor (γ): 0.99
-Batch size: 100
-Replay buffer size: 100,000
-Soft update parameter (τ): 1e-3
-Epsilon decay: 0.995

## Results

The agent achieves an average score ≥ 200, successfully landing the lunar module.
Video demonstration included for qualitative evaluation.
