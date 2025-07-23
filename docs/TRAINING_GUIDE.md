# Training Guide

Complete guide for training reinforcement learning agents with Genesis-Gymnasium integration.

## 🎯 Overview

This guide covers everything you need to know about training RL agents on Genesis environments, from basic concepts to advanced techniques.

## Quick Training Examples

### 1. Basic Training Run

```bash
# Start with the simple demo
uv run python single_env_rl_demo.py

# Expected output:
# 🎮 Single Environment Genesis RL Demo
# ✅ Training on: cuda
# 📊 Final Results: Success Rate: 20.0% (2/10)
```

### 2. Full Training Pipeline

```bash
# Train PPO agent
uv run python rl_training_example.py --train --algorithm PPO

# Train SAC agent  
uv run python rl_training_example.py --train --algorithm SAC

# Evaluate trained model
uv run python rl_training_example.py --evaluate --algorithm PPO
```

## Environment Setup

### Creating Training Environments

```python
from genesis_gym_wrapper import GenesisGymWrapper
from stable_baselines3.common.monitor import Monitor

# Basic environment
env = GenesisGymWrapper(
    robot_type="franka",
    use_gpu=True,
    show_viewer=False  # Disable for training speed
)

# Wrap for training
env = Monitor(env)
```

### Custom Task Environments

```python
class ReachingEnv(GenesisGymWrapper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target_pos = np.array([0.5, 0.0, 0.3])
        self.target_tolerance = 0.05
        
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        
        # Randomize target position
        self.target_pos = np.array([
            np.random.uniform(0.3, 0.7),    # x: forward reach
            np.random.uniform(-0.3, 0.3),   # y: left-right  
            np.random.uniform(0.2, 0.5),    # z: height
        ])
        
        # Add target to observation
        obs_with_target = np.concatenate([obs, self.target_pos])
        return obs_with_target, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Compute end-effector position (simplified)
        joint_pos = obs[:self.n_ctrl_dofs]
        ee_pos = self._compute_forward_kinematics(joint_pos)
        
        # Distance-based reward
        distance = np.linalg.norm(ee_pos - self.target_pos)
        reward = -distance * 5.0
        
        # Success bonus
        if distance < self.target_tolerance:
            reward += 10.0
            terminated = True
            
        # Add target to observation
        obs_with_target = np.concatenate([obs, self.target_pos])
        
        info.update({
            'distance_to_target': distance,
            'is_success': distance < self.target_tolerance
        })
        
        return obs_with_target, reward, terminated, truncated, info
```

## Algorithm Configuration

### PPO (Recommended for Beginners)

**Best for**: Continuous control, stable training, robotics

```python
from stable_baselines3 import PPO

model = PPO(
    "MlpPolicy",
    env,
    # Core hyperparameters
    learning_rate=3e-4,
    n_steps=2048,           # Steps per rollout
    batch_size=64,          # Minibatch size
    n_epochs=10,            # Training epochs per rollout
    
    # Advanced settings
    gamma=0.99,             # Discount factor
    gae_lambda=0.95,        # GAE parameter
    clip_range=0.2,         # PPO clip range
    ent_coef=0.01,          # Entropy coefficient
    vf_coef=0.5,            # Value function coefficient
    
    # Performance
    device="cuda",          # Use GPU for neural networks
    verbose=1,
    tensorboard_log="./tensorboard_logs/"
)
```

**Training characteristics:**
- ✅ Stable and reliable
- ✅ Good sample efficiency
- ✅ Works well with continuous actions
- ⚠️ Can be slower than off-policy methods

### SAC (Advanced Users)

**Best for**: Sample efficiency, continuous control, exploration

```python
from stable_baselines3 import SAC

model = SAC(
    "MlpPolicy",
    env,
    # Core hyperparameters
    learning_rate=3e-4,
    buffer_size=100000,     # Replay buffer size
    learning_starts=10000,  # Steps before training starts
    batch_size=256,         # Batch size for training
    
    # SAC-specific
    tau=0.005,              # Soft update coefficient
    gamma=0.99,             # Discount factor
    train_freq=1,           # Train every N steps
    gradient_steps=1,       # Gradient steps per env step
    
    # Exploration
    target_update_interval=1,
    target_entropy="auto",
    
    # Performance
    device="cuda",
    verbose=1,
    tensorboard_log="./tensorboard_logs/"
)
```

**Training characteristics:**
- ✅ High sample efficiency
- ✅ Good exploration
- ✅ Off-policy learning
- ⚠️ More complex hyperparameter tuning

### TD3 (Alternative)

**Best for**: Deterministic policies, continuous control

```python
from stable_baselines3 import TD3

model = TD3(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    buffer_size=100000,
    learning_starts=25000,
    batch_size=100,
    tau=0.005,
    gamma=0.99,
    train_freq=(1, "step"),
    gradient_steps=-1,
    policy_delay=2,         # TD3-specific: delayed policy updates
    target_policy_noise=0.2,
    target_noise_clip=0.5,
    device="cuda",
    verbose=1
)
```

## Training Strategies

### 1. Curriculum Learning

Start with easier tasks and gradually increase difficulty:

```python
class CurriculumReachingEnv(ReachingEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.difficulty = 0.1  # Start easy
        
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        
        # Easy targets close to robot
        self.target_pos = np.array([
            0.4 + self.difficulty * np.random.uniform(-0.3, 0.3),
            self.difficulty * np.random.uniform(-0.3, 0.3),
            0.25 + self.difficulty * np.random.uniform(-0.1, 0.1)
        ])
        
        return obs, info
    
    def increase_difficulty(self):
        self.difficulty = min(1.0, self.difficulty + 0.1)
```

### 2. Reward Shaping

Design informative reward functions:

```python
def compute_shaped_reward(self, obs, action, ee_pos, target_pos):
    # Distance reward (dense)
    distance = np.linalg.norm(ee_pos - target_pos)
    distance_reward = -distance * 2.0
    
    # Velocity penalty (encourage smooth motion)
    joint_vel = obs[self.n_ctrl_dofs:]
    velocity_penalty = -0.01 * np.sum(np.square(joint_vel))
    
    # Action penalty (encourage efficiency)
    action_penalty = -0.001 * np.sum(np.square(action))
    
    # Success bonus (sparse)
    success_bonus = 10.0 if distance < 0.05 else 0.0
    
    # Progress bonus (encourage getting closer)
    if hasattr(self, 'prev_distance'):
        progress = self.prev_distance - distance
        progress_bonus = progress * 5.0
    else:
        progress_bonus = 0.0
    
    self.prev_distance = distance
    
    return distance_reward + velocity_penalty + action_penalty + success_bonus + progress_bonus
```

### 3. Environment Randomization

Improve generalization with domain randomization:

```python
class RandomizedEnv(ReachingEnv):
    def reset(self, seed=None, options=None):
        # Randomize physics properties
        self.scene.set_gravity(
            np.random.uniform(-10, -8),  # Gravity variation
            np.random.uniform(-0.5, 0.5), # Wind forces
            np.random.uniform(-0.5, 0.5)
        )
        
        # Randomize robot properties
        joint_damping = np.random.uniform(0.8, 1.2, self.n_ctrl_dofs)
        self.robot.set_dofs_kv(joint_damping, np.arange(self.n_ctrl_dofs))
        
        return super().reset(seed=seed, options=options)
```

## Training Best Practices

### 1. Monitoring Training

```python
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

# Create evaluation environment
eval_env = Monitor(ReachingEnv(use_gpu=True, show_viewer=False))

# Early stopping on success
stop_callback = StopTrainingOnRewardThreshold(
    reward_threshold=8.0,  # Adjust based on your reward scale
    verbose=1
)

# Evaluation callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/",
    log_path="./logs/",
    eval_freq=10000,
    deterministic=True,
    render=False,
    callback_on_new_best=stop_callback
)

# Train with callbacks
model.learn(
    total_timesteps=500000,
    callback=eval_callback,
    progress_bar=True
)
```

### 2. Hyperparameter Tuning

Use Optuna for systematic hyperparameter optimization:

```python
import optuna
from optuna.pruners import MedianPruner

def optimize_ppo(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    
    # Create and train model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        verbose=0
    )
    
    model.learn(total_timesteps=50000)
    
    # Evaluate performance
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10)
    
    return mean_reward

# Run optimization
study = optuna.create_study(direction="maximize")
study.optimize(optimize_ppo, n_trials=50)
```

### 3. Multi-Stage Training

Train in stages for complex tasks:

```python
# Stage 1: Learn basic motor skills
basic_env = BasicMotorEnv()
model = PPO("MlpPolicy", basic_env)
model.learn(total_timesteps=100000)

# Stage 2: Transfer to reaching task
reaching_env = ReachingEnv()
model.set_env(reaching_env)
model.learn(total_timesteps=200000)

# Stage 3: Fine-tune on complex manipulation
manipulation_env = ManipulationEnv()
model.set_env(manipulation_env)
model.learn(total_timesteps=300000)
```

## Performance Optimization

### 1. GPU Utilization

```python
# Check GPU usage
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Monitor during training
nvidia-smi -l 1
```

### 2. Training Speed

```python
# Optimize for speed
model = PPO(
    "MlpPolicy",
    env,
    device="cuda",           # GPU for neural networks
    n_envs=1,               # Single env (Genesis limitation)
    policy_kwargs=dict(
        net_arch=[64, 64]   # Smaller networks train faster
    )
)

# Use compiled environments
env = GenesisGymWrapper(use_gpu=True, show_viewer=False)
```

### 3. Memory Management

```python
# Clear GPU memory between runs
import torch
torch.cuda.empty_cache()

# Monitor memory usage
def check_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory - Allocated: {allocated:.1f}GB, Reserved: {reserved:.1f}GB")

check_memory()
```

## Evaluation and Testing

### 1. Model Evaluation

```python
from stable_baselines3.common.evaluation import evaluate_policy

# Load trained model
model = PPO.load("./models/ppo_reaching_final")

# Evaluate performance
mean_reward, std_reward = evaluate_policy(
    model, 
    eval_env,
    n_eval_episodes=20,
    deterministic=True
)

print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
```

### 2. Success Rate Analysis

```python
def evaluate_success_rate(model, env, n_episodes=50):
    successes = 0
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_length = 0
        
        for step in range(200):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_length += 1
            
            if terminated or truncated:
                if info.get('is_success', False):
                    successes += 1
                break
        
        episode_lengths.append(episode_length)
    
    success_rate = successes / n_episodes
    avg_length = np.mean(episode_lengths)
    
    print(f"Success Rate: {success_rate:.1%}")
    print(f"Average Episode Length: {avg_length:.1f}")
    
    return success_rate, avg_length
```

### 3. Visualization

```python
# Visualize trained agent
env_vis = ReachingEnv(use_gpu=True, show_viewer=True)
obs, _ = env_vis.reset()

for step in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env_vis.step(action)
    
    if terminated or truncated:
        break
        
env_vis.close()
```

## Troubleshooting Training Issues

### 1. Low Performance
- **Check GPU usage**: Ensure CUDA backend is active
- **Reduce observation noise**: Filter or normalize observations
- **Adjust learning rate**: Try 1e-4 to 1e-3 range
- **Increase training time**: Allow more timesteps for convergence

### 2. Unstable Training
- **Use PPO**: More stable than SAC for beginners
- **Reduce learning rate**: Slower but more stable learning
- **Add reward clipping**: Limit extreme reward values
- **Check environment**: Ensure deterministic resets

### 3. No Learning
- **Verify reward function**: Check that rewards are informative
- **Check action space**: Ensure actions affect the environment
- **Scale observations**: Normalize to [-1, 1] or [0, 1] range
- **Adjust hyperparameters**: Try default values first

## Next Steps

After mastering basic training:

1. **Advanced Environments**: Multi-robot, manipulation tasks
2. **Hierarchical RL**: Combine low-level and high-level policies
3. **Sim-to-Real**: Transfer learned policies to real robots
4. **Multi-Modal**: Add vision and tactile sensors

Check out [`ADVANCED_USAGE.md`](ADVANCED_USAGE.md) for more complex scenarios!