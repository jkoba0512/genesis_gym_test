#!/usr/bin/env python3
"""
Single Environment RL Demo - Avoid Genesis multi-initialization issues
by using only one environment instance and sequential training.
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from genesis_gym_wrapper import GenesisGymWrapper
from gymnasium import spaces


class SimpleReachingEnv(GenesisGymWrapper):
    """Simplified reaching environment for single-env training."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target_pos = np.array([0.5, 0.0, 0.3])
        self.max_steps = 50  # Shorter episodes
        self.episode_count = 0
        self.success_count = 0
    
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        
        # Randomize target slightly
        self.target_pos = np.array([
            0.4 + 0.2 * np.random.uniform(-1, 1),
            0.1 * np.random.uniform(-1, 1),
            0.25 + 0.1 * np.random.uniform(-1, 1)
        ])
        
        self.episode_count += 1
        obs_with_target = np.concatenate([obs, self.target_pos])
        return obs_with_target, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Simple end-effector proxy
        joint_pos = obs[:self.n_ctrl_dofs]
        ee_pos = np.array([
            0.2 + 0.4 * np.clip(joint_pos[0], -1, 1),
            0.3 * np.clip(joint_pos[1], -1, 1),
            0.15 + 0.2 * np.clip(joint_pos[3], -1, 1)
        ])
        
        # Distance-based reward
        distance = np.linalg.norm(ee_pos - self.target_pos)
        reward = -distance * 10.0
        
        # Success bonus
        success = distance < 0.08
        if success:
            reward += 10.0
            terminated = True
            self.success_count += 1
        
        # Control penalty
        reward -= 0.01 * np.sum(np.square(action))
        
        obs_with_target = np.concatenate([obs, self.target_pos])
        
        info.update({
            'distance': distance,
            'success': success,
            'success_rate': self.success_count / max(1, self.episode_count)
        })
        
        return obs_with_target, reward, terminated, truncated, info
    
    def _setup_spaces(self):
        super()._setup_spaces()
        
        # Extend observation space for target
        original_dim = self.observation_space.shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(original_dim + 3,), dtype=np.float32
        )


def main():
    print("🎮 Single Environment Genesis RL Demo")
    print("=" * 50)
    
    # Create single environment
    print("\n1️⃣ Creating environment...")
    env = SimpleReachingEnv(use_gpu=True, show_viewer=False)
    env = Monitor(env)
    
    obs, _ = env.reset()
    print(f"✅ Environment ready! Obs shape: {obs.shape}")
    
    # Test random actions
    print("\n2️⃣ Testing random actions...")
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {i+1}: reward={reward:.2f}, distance={info.get('distance', 0):.3f}")
        if terminated or truncated:
            obs, _ = env.reset()
    
    # Create PPO agent
    print("\n3️⃣ Creating PPO agent...")
    model = PPO(
        "MlpPolicy", 
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,  # Smaller buffer for single env
        batch_size=32,
        n_epochs=4,
        device="auto"
    )
    
    print(f"✅ Training on: {model.device}")
    
    # Short training
    print("\n4️⃣ Training for 10k steps...")
    model.learn(
        total_timesteps=10000, 
        progress_bar=True,
        log_interval=10
    )
    
    # Test trained agent
    print("\n5️⃣ Testing trained agent...")
    successes = 0
    total_episodes = 10
    
    for episode in range(total_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        
        for step in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        success = info.get('success', False)
        if success:
            successes += 1
        
        print(f"  Episode {episode+1}: reward={episode_reward:.1f}, "
              f"distance={info.get('distance', 0):.3f}, "
              f"success={'✅' if success else '❌'}")
    
    env.close()
    
    success_rate = successes / total_episodes
    print(f"\n📊 Final Results:")
    print(f"  Success Rate: {success_rate:.1%} ({successes}/{total_episodes})")
    print(f"  Environment Success Rate: {info.get('success_rate', 0):.1%}")
    
    if success_rate >= 0.5:
        print("🎉 Great! Genesis-Gymnasium integration working well!")
    elif success_rate >= 0.2:
        print("👍 Good progress! More training would improve performance")
    else:
        print("⚠️ Needs more training time or hyperparameter tuning")
    
    print("\n✅ Single environment demo completed!")


if __name__ == "__main__":
    main()