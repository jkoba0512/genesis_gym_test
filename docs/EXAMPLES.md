# Examples and Use Cases

Practical examples demonstrating different applications of the Genesis-Gymnasium integration.

## 🎯 Basic Examples

### 1. Simple Robot Control

```python
#!/usr/bin/env python3
"""
Simple robot control example - move joints in a sine wave pattern.
"""

import numpy as np
import time
from genesis_gym_wrapper import GenesisGymWrapper

def sine_wave_control():
    """Control robot joints with sine wave pattern."""
    
    # Create environment with viewer
    env = GenesisGymWrapper(
        robot_type="franka",
        use_gpu=True,
        show_viewer=True  # Enable visualization
    )
    
    obs, info = env.reset()
    print(f"🤖 Robot initialized with {info['n_ctrl_dofs']} controllable DOFs")
    
    # Control parameters
    frequency = 0.5  # Hz
    amplitude = 0.3  # radians
    duration = 10.0  # seconds
    
    start_time = time.time()
    step_count = 0
    
    while time.time() - start_time < duration:
        # Generate sine wave actions
        t = time.time() - start_time
        action = env.initial_joint_pos + amplitude * np.sin(2 * np.pi * frequency * t)
        
        # Apply action
        obs, reward, terminated, truncated, info = env.step(action)
        
        step_count += 1
        if step_count % 100 == 0:
            print(f"Step {step_count}: Reward = {reward:.3f}")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
    print(f"✅ Completed {step_count} control steps")

if __name__ == "__main__":
    sine_wave_control()
```

### 2. Random Policy Evaluation

```python
#!/usr/bin/env python3
"""
Evaluate a random policy to understand environment dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
from genesis_gym_wrapper import GenesisGymWrapper

def evaluate_random_policy(n_episodes=10):
    """Evaluate random policy performance."""
    
    env = GenesisGymWrapper(use_gpu=True, show_viewer=False)
    
    episode_rewards = []
    episode_lengths = []
    joint_position_history = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        joint_positions = []
        
        print(f"🎮 Episode {episode + 1}/{n_episodes}")
        
        for step in range(200):  # Max 200 steps per episode
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Collect data
            episode_reward += reward
            episode_length += 1
            joint_positions.append(obs[:env.n_ctrl_dofs].copy())
            
            if terminated or truncated:
                print(f"  Terminated at step {step + 1}")
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        joint_position_history.append(np.array(joint_positions))
        
        print(f"  Total reward: {episode_reward:.2f}")
    
    env.close()
    
    # Analysis
    print(f"\n📊 Random Policy Analysis:")
    print(f"Average episode reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Reward range: [{np.min(episode_rewards):.2f}, {np.max(episode_rewards):.2f}]")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Episode rewards
    axes[0,0].plot(episode_rewards, 'bo-')
    axes[0,0].set_title('Episode Rewards')
    axes[0,0].set_xlabel('Episode')
    axes[0,0].set_ylabel('Total Reward')
    
    # Episode lengths
    axes[0,1].plot(episode_lengths, 'ro-')
    axes[0,1].set_title('Episode Lengths')
    axes[0,1].set_xlabel('Episode')
    axes[0,1].set_ylabel('Steps')
    
    # Joint position distribution
    all_joint_pos = np.concatenate(joint_position_history)
    axes[1,0].boxplot([all_joint_pos[:, i] for i in range(env.n_ctrl_dofs)])
    axes[1,0].set_title('Joint Position Distribution')
    axes[1,0].set_xlabel('Joint Index')
    axes[1,0].set_ylabel('Position (rad)')
    
    # Sample trajectory
    if joint_position_history:
        sample_traj = joint_position_history[0]
        for i in range(min(3, env.n_ctrl_dofs)):  # Plot first 3 joints
            axes[1,1].plot(sample_traj[:, i], label=f'Joint {i}')
        axes[1,1].set_title('Sample Joint Trajectory')
        axes[1,1].set_xlabel('Time Step')
        axes[1,1].set_ylabel('Position (rad)')
        axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('random_policy_analysis.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    evaluate_random_policy()
```

## 🎯 Reinforcement Learning Examples

### 3. PPO Training with Custom Reward

```python
#!/usr/bin/env python3
"""
Train PPO agent on custom reaching task with detailed logging.
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from genesis_gym_wrapper import GenesisGymWrapper
from gymnasium import spaces

class DetailedReachingEnv(GenesisGymWrapper):
    """Reaching environment with detailed reward components."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target_pos = np.array([0.5, 0.0, 0.3])
        self.max_steps = 100
        
        # Tracking variables
        self.episode_count = 0
        self.success_count = 0
        self.best_distance = float('inf')
    
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        
        # Randomize target position
        self.target_pos = np.array([
            np.random.uniform(0.3, 0.7),
            np.random.uniform(-0.2, 0.2),
            np.random.uniform(0.15, 0.4)
        ])
        
        self.episode_count += 1
        self.best_distance = float('inf')
        
        # Add target to observation
        obs_extended = np.concatenate([obs, self.target_pos])
        return obs_extended, info
    
    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)
        
        # Compute end-effector position (simplified forward kinematics)
        joint_pos = obs[:self.n_ctrl_dofs]
        ee_pos = self._compute_ee_position(joint_pos)
        
        # Distance to target
        distance = np.linalg.norm(ee_pos - self.target_pos)
        self.best_distance = min(self.best_distance, distance)
        
        # Reward components
        rewards = {
            'distance': -distance * 3.0,
            'progress': (self.best_distance - distance) * 10.0 if distance < self.best_distance else 0.0,
            'success': 15.0 if distance < 0.08 else 0.0,
            'efficiency': -0.01 * np.sum(np.square(action - self.initial_joint_pos)),
            'smoothness': -0.005 * np.sum(np.square(obs[self.n_ctrl_dofs:]))  # velocity penalty
        }
        
        total_reward = sum(rewards.values())
        
        # Termination conditions
        success = distance < 0.08
        if success:
            terminated = True
            self.success_count += 1
        
        # Extended observation
        obs_extended = np.concatenate([obs, self.target_pos])
        
        # Enhanced info
        info.update({
            'distance': distance,
            'best_distance': self.best_distance,
            'success': success,
            'success_rate': self.success_count / self.episode_count,
            'reward_components': rewards,
            'ee_position': ee_pos,
            'target_position': self.target_pos
        })
        
        return obs_extended, total_reward, terminated, truncated, info
    
    def _compute_ee_position(self, joint_pos):
        """Simplified forward kinematics for end-effector position."""
        # This is a rough approximation - replace with actual FK
        x = 0.2 + 0.4 * np.cos(joint_pos[0]) * np.cos(joint_pos[1])
        y = 0.4 * np.sin(joint_pos[0]) * np.cos(joint_pos[1])
        z = 0.1 + 0.4 * np.sin(joint_pos[1]) + 0.3 * np.sin(joint_pos[1] + joint_pos[3])
        return np.array([x, y, z])
    
    def _setup_spaces(self):
        """Extended observation space including target position."""
        # Action space (unchanged)
        self.action_space = spaces.Box(
            low=self.joint_limits_low,
            high=self.joint_limits_high,
            shape=(self.n_ctrl_dofs,),
            dtype=np.float32
        )
        
        # Extended observation: joint state + target position
        obs_dim = self.n_ctrl_dofs * 2 + 3  # positions + velocities + target
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

class TrainingCallback(BaseCallback):
    """Custom callback for detailed training logging."""
    
    def __init__(self, log_freq=1000):
        super().__init__()
        self.log_freq = log_freq
        
    def _on_step(self):
        # Log detailed metrics every log_freq steps
        if self.n_calls % self.log_freq == 0:
            # Get environment info
            if hasattr(self.training_env, 'get_attr'):
                success_rates = self.training_env.get_attr('success_count')
                episode_counts = self.training_env.get_attr('episode_count')
                
                if episode_counts[0] > 0:
                    avg_success_rate = success_rates[0] / episode_counts[0]
                    self.logger.record('train/success_rate', avg_success_rate)
        
        return True

def train_detailed_ppo():
    """Train PPO with detailed monitoring."""
    
    # Create training environment
    env = DetailedReachingEnv(use_gpu=True, show_viewer=False)
    env = Monitor(env)
    
    print("🚀 Starting detailed PPO training")
    print(f"Environment: {env.observation_space.shape[0]}-dim obs, {env.action_space.shape[0]}-dim action")
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./tensorboard_logs/detailed_ppo",
        device="cuda"
    )
    
    # Train with callback
    callback = TrainingCallback(log_freq=2048)
    model.learn(
        total_timesteps=200000,
        callback=callback,
        progress_bar=True
    )
    
    # Save model
    model.save("detailed_ppo_reaching")
    
    # Evaluate trained model
    print("\n🎯 Evaluating trained model...")
    test_env = DetailedReachingEnv(use_gpu=True, show_viewer=True)
    
    for episode in range(5):
        obs, _ = test_env.reset()
        episode_reward = 0
        
        print(f"\n📍 Test Episode {episode + 1}")
        print(f"Target: [{test_env.target_pos[0]:.2f}, {test_env.target_pos[1]:.2f}, {test_env.target_pos[2]:.2f}]")
        
        for step in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            episode_reward += reward
            
            if step % 20 == 0:
                print(f"  Step {step}: Distance = {info['distance']:.3f}m, Reward = {reward:.2f}")
            
            if terminated or truncated:
                success = "✅ SUCCESS" if info['success'] else "❌ FAILED"
                print(f"  {success} - Final distance: {info['distance']:.3f}m")
                print(f"  Episode reward: {episode_reward:.2f}")
                break
    
    test_env.close()
    print("\n✅ Training and evaluation completed!")

if __name__ == "__main__":
    train_detailed_ppo()
```

### 4. Curriculum Learning Example

```python
#!/usr/bin/env python3
"""
Curriculum learning example - gradually increase task difficulty.
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from genesis_gym_wrapper import GenesisGymWrapper

class CurriculumReachingEnv(GenesisGymWrapper):
    """Reaching environment with curriculum learning support."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Curriculum parameters
        self.curriculum_level = 0
        self.target_range = 0.2  # Start with close targets
        self.height_variation = 0.05
        self.add_obstacles = False
        
        # Performance tracking
        self.episode_count = 0
        self.success_count = 0
        self.recent_successes = []  # Track recent performance
        
    def set_curriculum_level(self, level):
        """Update curriculum difficulty level."""
        self.curriculum_level = level
        
        if level == 0:
            # Level 0: Very close targets
            self.target_range = 0.15
            self.height_variation = 0.02
            self.add_obstacles = False
            print("📚 Curriculum Level 0: Close targets")
            
        elif level == 1:
            # Level 1: Medium distance targets
            self.target_range = 0.3
            self.height_variation = 0.08
            self.add_obstacles = False
            print("📚 Curriculum Level 1: Medium targets")
            
        elif level == 2:
            # Level 2: Distant targets
            self.target_range = 0.45
            self.height_variation = 0.15
            self.add_obstacles = False
            print("📚 Curriculum Level 2: Distant targets")
            
        elif level == 3:
            # Level 3: Distant targets with obstacles
            self.target_range = 0.45
            self.height_variation = 0.15
            self.add_obstacles = True
            print("📚 Curriculum Level 3: Targets with obstacles")
    
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        
        # Generate target based on current curriculum
        self.target_pos = self._generate_curriculum_target()
        
        # Add obstacles if required
        if self.add_obstacles:
            self._add_obstacles()
        
        self.episode_count += 1
        
        # Extended observation
        obs_extended = np.concatenate([obs, self.target_pos])
        return obs_extended, info
    
    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)
        
        # Compute distance to target
        joint_pos = obs[:self.n_ctrl_dofs]
        ee_pos = self._approximate_ee_position(joint_pos)
        distance = np.linalg.norm(ee_pos - self.target_pos)
        
        # Curriculum-adaptive reward
        reward = self._compute_curriculum_reward(distance, action, obs)
        
        # Success criteria (adaptive based on level)
        success_threshold = 0.12 - 0.02 * self.curriculum_level  # Tighter as level increases
        success = distance < success_threshold
        
        if success:
            terminated = True
            self.success_count += 1
        
        # Track recent performance
        if terminated or truncated:
            self.recent_successes.append(float(success))
            if len(self.recent_successes) > 50:  # Keep only recent 50 episodes
                self.recent_successes.pop(0)
        
        # Extended observation and info
        obs_extended = np.concatenate([obs, self.target_pos])
        info.update({
            'distance': distance,
            'success': success,
            'curriculum_level': self.curriculum_level,
            'recent_success_rate': np.mean(self.recent_successes) if self.recent_successes else 0.0
        })
        
        return obs_extended, reward, terminated, truncated, info
    
    def _generate_curriculum_target(self):
        """Generate target position based on curriculum level."""
        # Base position (front of robot)
        base_x = 0.4
        base_y = 0.0
        base_z = 0.25
        
        # Add randomization based on curriculum
        x = base_x + np.random.uniform(-self.target_range, self.target_range)
        y = base_y + np.random.uniform(-self.target_range, self.target_range)
        z = base_z + np.random.uniform(-self.height_variation, self.height_variation)
        
        return np.array([x, y, z])
    
    def _add_obstacles(self):
        """Add obstacles for advanced curriculum levels."""
        # This would add obstacle entities to the scene
        # Simplified for this example
        pass
    
    def _compute_curriculum_reward(self, distance, action, obs):
        """Compute reward adapted to curriculum level."""
        # Base distance reward
        distance_reward = -distance * (3.0 + self.curriculum_level)
        
        # Success bonus (increases with level)
        success_bonus = 10.0 + 5.0 * self.curriculum_level if distance < 0.1 else 0.0
        
        # Efficiency penalty (more important at higher levels)
        efficiency_penalty = -0.01 * (1 + 0.5 * self.curriculum_level) * np.sum(np.square(action))
        
        return distance_reward + success_bonus + efficiency_penalty
    
    def get_recent_success_rate(self):
        """Get recent success rate for curriculum advancement."""
        return np.mean(self.recent_successes) if self.recent_successes else 0.0

class CurriculumCallback(BaseCallback):
    """Callback to manage curriculum advancement."""
    
    def __init__(self, env, check_freq=10000, success_threshold=0.8):
        super().__init__()
        self.env = env
        self.check_freq = check_freq
        self.success_threshold = success_threshold
        self.last_check = 0
        
    def _on_step(self):
        if self.n_calls - self.last_check >= self.check_freq:
            self.last_check = self.n_calls
            
            # Check if we should advance curriculum
            if hasattr(self.training_env, 'env_method'):
                success_rates = self.training_env.env_method('get_recent_success_rate')
                current_level = self.training_env.get_attr('curriculum_level')[0]
                
                if success_rates[0] >= self.success_threshold and current_level < 3:
                    new_level = current_level + 1
                    self.training_env.env_method('set_curriculum_level', new_level)
                    print(f"🎓 Advanced to curriculum level {new_level} (success rate: {success_rates[0]:.1%})")
                    
                    # Log curriculum advancement
                    self.logger.record('curriculum/level', new_level)
                    self.logger.record('curriculum/success_rate', success_rates[0])
        
        return True

def train_with_curriculum():
    """Train agent using curriculum learning."""
    
    # Create environment
    env = CurriculumReachingEnv(use_gpu=True, show_viewer=False)
    env = Monitor(env)
    
    print("🎓 Starting curriculum learning training")
    
    # Start with easiest level
    env.set_curriculum_level(0)
    
    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        verbose=1,
        tensorboard_log="./tensorboard_logs/curriculum",
        device="cuda"
    )
    
    # Train with curriculum callback
    curriculum_callback = CurriculumCallback(
        env, 
        check_freq=10000, 
        success_threshold=0.7
    )
    
    model.learn(
        total_timesteps=500000,
        callback=curriculum_callback,
        progress_bar=True
    )
    
    # Save final model
    model.save("curriculum_trained_model")
    
    # Test on all curriculum levels
    print("\n🎯 Testing on all curriculum levels...")
    test_env = CurriculumReachingEnv(use_gpu=True, show_viewer=True)
    
    for level in range(4):
        print(f"\n📚 Testing Level {level}")
        test_env.set_curriculum_level(level)
        
        successes = 0
        for episode in range(10):
            obs, _ = test_env.reset()
            
            for step in range(100):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = test_env.step(action)
                
                if terminated or truncated:
                    if info['success']:
                        successes += 1
                    break
        
        print(f"Level {level} success rate: {successes}/10 ({successes*10}%)")
    
    test_env.close()

if __name__ == "__main__":
    train_with_curriculum()
```

### 5. Multi-Modal Observation Example

```python
#!/usr/bin/env python3
"""
Multi-modal observation example combining vision, proprioception, and force.
"""

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from genesis_gym_wrapper import GenesisGymWrapper
from gymnasium import spaces

class MultiModalEnv(GenesisGymWrapper):
    """Environment with multi-modal observations."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.img_size = 64
        
    def _setup_scene(self):
        """Add camera and force sensors."""
        super()._setup_scene()
        
        # Add camera
        self.camera = self.scene.add_camera(
            pos=(0.7, 0.3, 0.5),
            lookat=(0.0, 0.0, 0.2),
            fov=45,
            width=self.img_size,
            height=self.img_size
        )
        
        # Add target object to track
        self.target_object = self.scene.add_entity(
            gs.morphs.Sphere(radius=0.03),
            pos=(0.4, 0.1, 0.25)
        )
        
    def _setup_spaces(self):
        """Define multi-modal observation space."""
        # Action space (unchanged)
        self.action_space = spaces.Box(
            low=self.joint_limits_low,
            high=self.joint_limits_high,
            shape=(self.n_ctrl_dofs,),
            dtype=np.float32
        )
        
        # Multi-modal observation space
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0, high=255,
                shape=(self.img_size, self.img_size, 3),
                dtype=np.uint8
            ),
            'proprioception': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.n_ctrl_dofs * 2,),  # positions + velocities
                dtype=np.float32
            ),
            'force': spaces.Box(
                low=-100, high=100,
                shape=(6,),  # 3D force + 3D torque
                dtype=np.float32
            ),
            'target_info': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(6,),  # target position + velocity
                dtype=np.float32
            )
        })
    
    def _get_observation(self):
        """Get multi-modal observation."""
        # Proprioception
        joint_pos = self.robot.get_dofs_position()[:self.n_ctrl_dofs]
        joint_vel = self.robot.get_dofs_velocity()[:self.n_ctrl_dofs]
        
        if hasattr(joint_pos, 'cpu'):
            joint_pos = joint_pos.cpu().numpy()
            joint_vel = joint_vel.cpu().numpy()
        
        proprioception = np.concatenate([joint_pos, joint_vel]).astype(np.float32)
        
        # Vision
        image = self.camera.get_image()
        if hasattr(image, 'cpu'):
            image = image.cpu().numpy()
        image = (image * 255).astype(np.uint8)
        
        # Force (simulated end-effector force)
        ee_force = self._get_end_effector_force()
        
        # Target information
        target_pos = self.target_object.get_pos()
        target_vel = self.target_object.get_vel()
        if hasattr(target_pos, 'cpu'):
            target_pos = target_pos.cpu().numpy()
            target_vel = target_vel.cpu().numpy()
        
        target_info = np.concatenate([target_pos, target_vel]).astype(np.float32)
        
        return {
            'image': image,
            'proprioception': proprioception,
            'force': ee_force,
            'target_info': target_info
        }
    
    def _get_end_effector_force(self):
        """Get simulated end-effector force."""
        # Simplified force estimation based on joint torques
        joint_forces = self.robot.get_dofs_force()[:self.n_ctrl_dofs]
        if hasattr(joint_forces, 'cpu'):
            joint_forces = joint_forces.cpu().numpy()
        
        # Simple approximation of Cartesian force
        force_x = np.sum(joint_forces[:3]) * 0.1
        force_y = np.sum(joint_forces[1:4]) * 0.1
        force_z = np.sum(joint_forces[2:5]) * 0.1
        torque_x = joint_forces[4] * 0.05 if len(joint_forces) > 4 else 0
        torque_y = joint_forces[5] * 0.05 if len(joint_forces) > 5 else 0
        torque_z = joint_forces[6] * 0.05 if len(joint_forces) > 6 else 0
        
        return np.array([force_x, force_y, force_z, torque_x, torque_y, torque_z], dtype=np.float32)

class MultiModalFeaturesExtractor(BaseFeaturesExtractor):
    """Features extractor for multi-modal observations."""
    
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        
        # Vision encoder (CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        cnn_output_dim = 128 * 4 * 4  # 2048
        
        # Proprioception encoder
        proprio_dim = observation_space['proprioception'].shape[0]
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Force encoder
        force_dim = observation_space['force'].shape[0]
        self.force_encoder = nn.Sequential(
            nn.Linear(force_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Target encoder
        target_dim = observation_space['target_info'].shape[0]
        self.target_encoder = nn.Sequential(
            nn.Linear(target_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Fusion network
        total_features = cnn_output_dim + 64 + 16 + 16  # 2144
        self.fusion = nn.Sequential(
            nn.Linear(total_features, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations):
        # Process each modality
        image = observations['image'].float() / 255.0
        image = image.permute(0, 3, 1, 2)  # NHWC -> NCHW
        vision_features = self.cnn(image)
        
        proprio_features = self.proprio_encoder(observations['proprioception'])
        force_features = self.force_encoder(observations['force'])
        target_features = self.target_encoder(observations['target_info'])
        
        # Fuse all modalities
        combined = torch.cat([
            vision_features, 
            proprio_features, 
            force_features, 
            target_features
        ], dim=1)
        
        return self.fusion(combined)

def train_multimodal_agent():
    """Train agent with multi-modal observations."""
    
    # Create multi-modal environment
    env = MultiModalEnv(use_gpu=True, show_viewer=False)
    
    print("🔬 Training with multi-modal observations")
    print(f"Observation spaces:")
    for key, space in env.observation_space.spaces.items():
        print(f"  {key}: {space.shape}")
    
    # Create policy with custom features extractor
    policy_kwargs = dict(
        features_extractor_class=MultiModalFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=512),
    )
    
    model = PPO(
        "MultiInputPolicy",  # For dict observations
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=1024,  # Smaller steps due to complexity
        batch_size=32,
        verbose=1,
        tensorboard_log="./tensorboard_logs/multimodal",
        device="cuda"
    )
    
    # Train model
    model.learn(total_timesteps=100000, progress_bar=True)
    
    # Save model
    model.save("multimodal_agent")
    
    # Test trained agent
    print("🎯 Testing multi-modal agent...")
    test_env = MultiModalEnv(use_gpu=True, show_viewer=True)
    
    obs, _ = test_env.reset()
    for step in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        
        if step % 50 == 0:
            print(f"Step {step}: Reward = {reward:.3f}")
            print(f"  Force: [{obs['force'][0]:.2f}, {obs['force'][1]:.2f}, {obs['force'][2]:.2f}]")
        
        if terminated or truncated:
            break
    
    test_env.close()

if __name__ == "__main__":
    train_multimodal_agent()
```

## 🛠️ Utility Examples

### 6. Environment Benchmarking

```python
#!/usr/bin/env python3
"""
Benchmark Genesis environment performance across different configurations.
"""

import time
import numpy as np
from genesis_gym_wrapper import GenesisGymWrapper
import matplotlib.pyplot as plt

def benchmark_environment():
    """Comprehensive environment benchmarking."""
    
    configs = [
        {"name": "GPU + No Viewer", "use_gpu": True, "show_viewer": False},
        {"name": "GPU + Viewer", "use_gpu": True, "show_viewer": True},
        {"name": "CPU + No Viewer", "use_gpu": False, "show_viewer": False},
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n🔬 Benchmarking: {config['name']}")
        
        try:
            env = GenesisGymWrapper(**{k: v for k, v in config.items() if k != 'name'})
            
            # Warm-up
            obs, _ = env.reset()
            for _ in range(100):
                action = env.action_space.sample()
                env.step(action)
            
            # Benchmark
            n_steps = 1000
            start_time = time.time()
            
            for i in range(n_steps):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    env.reset()
            
            end_time = time.time()
            
            fps = n_steps / (end_time - start_time)
            results[config['name']] = fps
            
            print(f"✅ {config['name']}: {fps:.1f} FPS")
            
            env.close()
            
        except Exception as e:
            print(f"❌ {config['name']}: Failed - {e}")
            results[config['name']] = 0
    
    # Plot results
    names = list(results.keys())
    values = list(results.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, values)
    plt.title('Genesis Environment Performance Benchmark')
    plt.ylabel('FPS (Frames Per Second)')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{value:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('environment_benchmark.png', dpi=150)
    plt.show()
    
    return results

if __name__ == "__main__":
    benchmark_environment()
```

These examples provide a comprehensive foundation for using the Genesis-Gymnasium integration across different scenarios, from basic control to advanced multi-modal learning. Each example is self-contained and can be run independently to explore specific aspects of the system.