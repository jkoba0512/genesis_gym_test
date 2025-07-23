#!/usr/bin/env python3
"""
Complete RL Training Example with Genesis-Gymnasium Integration
Using Stable-Baselines3 for reinforcement learning training.
"""

import os
import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize

from genesis_gym_wrapper import GenesisGymWrapper


class ReachingEnv(GenesisGymWrapper):
    """
    Reaching task: Robot arm tries to reach a target position.
    Inherits from the complete Genesis wrapper.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Target position (in end-effector space)
        self.target_pos = np.array([0.5, 0.0, 0.3])
        self.target_tolerance = 0.05
        
        # Track success for evaluation
        self.success_count = 0
        self.episode_count = 0
    
    def reset(self, seed=None, options=None):
        """Reset environment and randomize target position."""
        obs, info = super().reset(seed=seed, options=options)
        
        # Randomize target position
        self.target_pos = np.array([
            np.random.uniform(0.3, 0.7),    # x: forward reach
            np.random.uniform(-0.3, 0.3),   # y: left-right
            np.random.uniform(0.2, 0.5),    # z: height
        ])
        
        # Add target to observation
        obs_with_target = np.concatenate([obs, self.target_pos])
        
        # Reset success tracking
        self.episode_count += 1
        
        return obs_with_target, info
    
    def step(self, action):
        """Step with reaching-specific reward."""
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Get end-effector position (simplified - would need forward kinematics)
        # For now, use a proxy based on joint positions
        joint_pos = obs[:self.n_ctrl_dofs]
        ee_pos_estimate = np.array([
            0.3 + 0.3 * np.sin(joint_pos[0]) * np.cos(joint_pos[1]),
            0.3 * np.cos(joint_pos[0]) * np.sin(joint_pos[2]),
            0.2 + 0.3 * np.sin(joint_pos[1] + joint_pos[3])
        ])
        
        # Distance to target
        distance = np.linalg.norm(ee_pos_estimate - self.target_pos)
        
        # Reaching reward
        reaching_reward = -distance * 2.0  # Dense reward
        
        # Success bonus
        success_bonus = 0.0
        if distance < self.target_tolerance:
            success_bonus = 10.0
            self.success_count += 1
            terminated = True
        
        # Control penalty (encourage smooth actions)
        control_penalty = -0.01 * np.sum(np.square(action))
        
        # Total reward
        reward = reaching_reward + success_bonus + control_penalty
        
        # Add target to observation
        obs_with_target = np.concatenate([obs, self.target_pos])
        
        # Add success info
        info.update({
            'distance_to_target': distance,
            'is_success': distance < self.target_tolerance,
            'success_rate': self.success_count / max(1, self.episode_count),
            'target_position': self.target_pos.copy(),
            'ee_position_estimate': ee_pos_estimate.copy(),
        })
        
        return obs_with_target, reward, terminated, truncated, info
    
    def _setup_spaces(self):
        """Override to include target position in observation space."""
        super()._setup_spaces()
        
        # Extend observation space to include target position (3D)
        original_obs_dim = self.observation_space.shape[0]
        extended_obs_dim = original_obs_dim + 3  # + target_pos
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(extended_obs_dim,),
            dtype=np.float32
        )


def train_ppo_agent():
    """Train PPO agent on reaching task."""
    print("🚀 Starting PPO Training on Genesis Reaching Task")
    print("=" * 60)
    
    # Create environment
    def make_env():
        env = ReachingEnv(
            robot_type="franka",
            scene_type="empty",
            use_gpu=True,
            show_viewer=False
        )
        env = Monitor(env)
        return env
    
    # Create vectorized environment
    env = make_vec_env(make_env, n_envs=4)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Create evaluation environment
    eval_env = make_vec_env(make_env, n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    
    # PPO agent configuration
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./tensorboard_logs/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"✅ Using device: {model.device}")
    print(f"📊 Training with {env.num_envs} parallel environments")
    
    # Callbacks
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=8.0, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./logs/",
        eval_freq=10000,
        deterministic=True,
        render=False,
        callback_on_new_best=stop_callback
    )
    
    # Train the agent
    print("\n🎯 Starting training...")
    model.learn(
        total_timesteps=500000,
        callback=eval_callback,
        progress_bar=True
    )
    
    # Save final model
    model.save("./models/ppo_genesis_reaching_final")
    env.save("./models/vecnormalize_genesis_reaching.pkl")
    
    print("\n✅ Training completed!")
    print("📁 Models saved to ./models/")
    
    return model, env


def train_sac_agent():
    """Train SAC agent on reaching task."""
    print("🚀 Starting SAC Training on Genesis Reaching Task")
    print("=" * 60)
    
    # Create environment
    def make_env():
        env = ReachingEnv(
            robot_type="franka",
            scene_type="empty",
            use_gpu=True,
            show_viewer=False
        )
        env = Monitor(env)
        return env
    
    # Single environment for SAC (off-policy)
    env = make_env()
    env = VecNormalize(make_vec_env(lambda: env, n_envs=1), norm_obs=True, norm_reward=True)
    
    # Create evaluation environment
    eval_env = VecNormalize(
        make_vec_env(make_env, n_envs=1),
        norm_obs=True, norm_reward=False, training=False
    )
    
    # SAC agent configuration
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./tensorboard_logs/",
        learning_rate=3e-4,
        buffer_size=100000,
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"✅ Using device: {model.device}")
    
    # Callbacks
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=8.0, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./logs/",
        eval_freq=5000,
        deterministic=True,
        render=False,
        callback_on_new_best=stop_callback
    )
    
    # Train the agent
    print("\n🎯 Starting SAC training...")
    model.learn(
        total_timesteps=200000,
        callback=eval_callback,
        progress_bar=True
    )
    
    # Save final model
    model.save("./models/sac_genesis_reaching_final")
    env.save("./models/vecnormalize_sac_genesis_reaching.pkl")
    
    print("\n✅ SAC Training completed!")
    print("📁 Models saved to ./models/")
    
    return model, env


def evaluate_trained_model(model_path="./models/ppo_genesis_reaching_final", algorithm="PPO"):
    """Evaluate a trained model."""
    print(f"🔍 Evaluating {algorithm} model: {model_path}")
    print("=" * 60)
    
    # Load model
    if algorithm == "PPO":
        model = PPO.load(model_path)
    elif algorithm == "SAC":
        model = SAC.load(model_path)
    else:
        raise ValueError(f"Algorithm {algorithm} not supported")
    
    # Create evaluation environment
    env = ReachingEnv(
        robot_type="franka",
        scene_type="empty",
        use_gpu=True,
        show_viewer=True  # Show visualization
    )
    
    # Run evaluation episodes
    n_episodes = 10
    success_count = 0
    total_reward = 0
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        print(f"\n📍 Episode {episode + 1}/{n_episodes}")
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
            # Print progress
            if 'distance_to_target' in info:
                print(f"  Distance: {info['distance_to_target']:.3f}m", end='\r')
        
        total_reward += episode_reward
        if info.get('is_success', False):
            success_count += 1
            print(f"  ✅ Success! Reward: {episode_reward:.2f}")
        else:
            print(f"  ❌ Failed. Reward: {episode_reward:.2f}")
    
    # Print results
    avg_reward = total_reward / n_episodes
    success_rate = success_count / n_episodes
    
    print(f"\n📊 Evaluation Results:")
    print(f"  Success Rate: {success_rate:.1%} ({success_count}/{n_episodes})")
    print(f"  Average Reward: {avg_reward:.2f}")
    
    env.close()
    return success_rate, avg_reward


def main():
    """Main training and evaluation pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Genesis-Gymnasium RL Training")
    parser.add_argument("--algorithm", choices=["PPO", "SAC"], default="PPO",
                       help="RL algorithm to use")
    parser.add_argument("--train", action="store_true", help="Train new model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate existing model")
    parser.add_argument("--model-path", type=str, help="Path to model for evaluation")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("tensorboard_logs", exist_ok=True)
    
    if args.train:
        if args.algorithm == "PPO":
            model, env = train_ppo_agent()
        elif args.algorithm == "SAC":
            model, env = train_sac_agent()
    
    if args.evaluate:
        model_path = args.model_path or f"./models/{args.algorithm.lower()}_genesis_reaching_final"
        evaluate_trained_model(model_path, args.algorithm)


if __name__ == "__main__":
    # Quick demo if run directly
    print("🎮 Genesis-Gymnasium RL Integration Demo")
    print("=" * 60)
    
    # Test environment creation
    print("\n1. Testing environment creation...")
    env = ReachingEnv(use_gpu=True, show_viewer=False)
    obs, _ = env.reset()
    print(f"✅ Environment created! Observation shape: {obs.shape}")
    
    # Test random actions
    print("\n2. Testing random actions...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {i+1}: reward={reward:.3f}, distance={info.get('distance_to_target', 0):.3f}m")
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    env.close()
    print("\n✅ Demo completed!")
    print("\nTo train an agent, run:")
    print("  python rl_training_example.py --train --algorithm PPO")
    print("To evaluate a trained model, run:")
    print("  python rl_training_example.py --evaluate --algorithm PPO")