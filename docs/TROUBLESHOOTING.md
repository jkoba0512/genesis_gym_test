# Troubleshooting Guide

Common issues and solutions for the Genesis-Gymnasium integration.

## 🔧 Installation Issues

### pymeshlab Platform Error

**Error:**
```
Distribution `pymeshlab==2025.7` can't be installed because it doesn't have wheels for manylinux_2_39_x86_64
```

**Solution:**
```bash
# Pin to compatible version
uv add "pymeshlab==2023.12" genesis-world

# Or edit pyproject.toml
[project]
dependencies = [
    "pymeshlab==2023.12",
    # ... other deps
]
```

**Explanation:** The latest pymeshlab version lacks Linux wheels. Version 2023.12 is stable and compatible.

### Genesis Installation Fails

**Error:**
```
ERROR: Could not find a version that satisfies the requirement genesis-world
```

**Solutions:**

1. **Update pip/uv:**
```bash
pip install --upgrade pip
# or
uv self update
```

2. **Check Python version:**
```bash
python --version  # Should be 3.10+
```

3. **Install from specific index:**
```bash
pip install genesis-world --index-url https://pypi.org/simple/
```

### CUDA Installation Issues

**Error:**
```
RuntimeError: CUDA not available
```

**Check CUDA setup:**
```bash
# Check NVIDIA driver
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🚀 Genesis Runtime Issues

### Genesis Already Initialized

**Error:**
```
genesis.GenesisException: Genesis already initialized.
```

**Cause:** Genesis can only be initialized once per Python process.

**Solutions:**

1. **Use single environment:**
```python
# Instead of multiple environments
envs = [GenesisGymWrapper() for _ in range(4)]  # ❌ Fails

# Use single environment
env = GenesisGymWrapper()  # ✅ Works
```

2. **Restart Python process:**
```bash
# Kill Python and restart
pkill python
uv run python your_script.py
```

3. **Use process isolation:**
```python
import multiprocessing as mp

def create_env():
    return GenesisGymWrapper()

# Each process gets own Genesis instance
env = mp.Process(target=create_env)
```

### GPU Backend Not Supported

**Error:**
```
backend gpu not supported for platform Linux
```

**Solution:**
```python
# Use correct backend name
gs.init(backend=gs.cuda)  # ✅ Correct
# not
gs.init(backend="gpu")    # ❌ Wrong
```

**Available backends:**
- `gs.cuda` - NVIDIA GPU acceleration
- `gs.vulkan` - Vulkan GPU support  
- `gs.cpu` - CPU fallback

### CUDA Tensor Conversion Error

**Error:**
```
TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
```

**Solution:**
Already handled in the wrapper:
```python
def _get_observation(self):
    joint_pos = self.robot.get_dofs_position()
    
    # Convert GPU tensors to CPU
    if hasattr(joint_pos, 'cpu'):
        joint_pos = joint_pos.cpu().numpy()
    
    return joint_pos
```

**Manual fix for custom code:**
```python
# Always convert tensors
if hasattr(tensor, 'cpu'):
    numpy_array = tensor.cpu().numpy()
else:
    numpy_array = np.array(tensor)
```

## 🎮 Environment Issues

### Robot Not Loading

**Error:**
```
FileNotFoundError: xml/franka_emika_panda/panda.xml
```

**Cause:** Robot MJCF file not found in Genesis assets.

**Solutions:**

1. **Check available robots:**
```python
import genesis as gs
print(gs.assets.list_robots())
```

2. **Use absolute path:**
```python
import genesis as gs
robot_path = gs.assets.get_robot_path("franka_panda")
robot = gs.morphs.MJCF(file=robot_path)
```

3. **Download missing assets:**
```bash
# Genesis should auto-download, but manual check:
python -c "import genesis as gs; gs.assets.download_all()"
```

### Environment Hangs on Build

**Symptom:** `scene.build()` never completes.

**Solutions:**

1. **Check GPU memory:**
```bash
nvidia-smi
# If memory is full, close other GPU processes
```

2. **Reduce scene complexity:**
```python
# Simpler scene
env = GenesisGymWrapper(
    scene_type="empty",  # Instead of "table"
    show_viewer=False
)
```

3. **Use CPU backend:**
```python
env = GenesisGymWrapper(use_gpu=False)
```

### Slow Performance

**Symptom:** Low FPS (< 100 steps/sec)

**Solutions:**

1. **Verify GPU usage:**
```python
# Check Genesis is using GPU
# Look for: "Using Genesis CUDA backend (GPU)"
```

2. **Close viewer:**
```python
env = GenesisGymWrapper(show_viewer=False)
```

3. **Monitor GPU:**
```bash
# Check GPU utilization
nvidia-smi -l 1
```

4. **Reduce simulation complexity:**
```python
self.scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=0.02,  # Larger timestep = faster simulation
        gravity=(0, 0, -9.81),
    ),
)
```

## 🤖 Training Issues

### Agent Not Learning

**Symptom:** Reward doesn't improve over time.

**Debugging steps:**

1. **Check reward function:**
```python
# Test reward manually
env = GenesisGymWrapper()
obs, _ = env.reset()

for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Action: {action[:3]}, Reward: {reward:.3f}")
```

2. **Verify action space:**
```python
# Ensure actions affect environment
print(f"Action space: {env.action_space}")
print(f"Action bounds: {env.action_space.low} to {env.action_space.high}")
```

3. **Check observation scaling:**
```python
# Observations should be reasonable scale
obs, _ = env.reset()
print(f"Obs range: {obs.min():.3f} to {obs.max():.3f}")
print(f"Obs mean: {obs.mean():.3f}, std: {obs.std():.3f}")
```

4. **Try simpler reward:**
```python
def _compute_reward(self, obs, action):
    # Dense reward based on state
    joint_pos = obs[:self.n_ctrl_dofs]
    target_pos = self.initial_joint_pos
    
    distance = np.linalg.norm(joint_pos - target_pos)
    return -distance  # Simple distance reward
```

### Training Unstable

**Symptom:** Reward oscillates wildly or crashes.

**Solutions:**

1. **Reduce learning rate:**
```python
model = PPO(
    "MlpPolicy", 
    env,
    learning_rate=1e-4,  # Reduced from 3e-4
)
```

2. **Clip rewards:**
```python
def _compute_reward(self, obs, action):
    reward = self._raw_reward(obs, action)
    return np.clip(reward, -10.0, 10.0)  # Prevent extreme rewards
```

3. **Use reward normalization:**
```python
from stable_baselines3.common.vec_env import VecNormalize

env = VecNormalize(make_vec_env(lambda: env), norm_reward=True)
```

4. **Check for NaN values:**
```python
def step(self, action):
    obs, reward, terminated, truncated, info = super().step(action)
    
    # Debug NaN values
    if np.isnan(obs).any():
        print("⚠️ NaN in observation!")
    if np.isnan(reward):
        print("⚠️ NaN in reward!")
        
    return obs, reward, terminated, truncated, info
```

### Memory Issues

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size:**
```python
model = PPO(
    "MlpPolicy",
    env,
    batch_size=32,  # Reduced from 64
    n_steps=512,    # Reduced from 2048
)
```

2. **Clear GPU cache:**
```python
import torch
torch.cuda.empty_cache()
```

3. **Use CPU for neural networks:**
```python
model = PPO("MlpPolicy", env, device="cpu")
```

4. **Monitor memory usage:**
```python
def check_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"GPU memory: {allocated:.1f}GB")

check_memory()  # Call periodically
```

## 🔍 Debugging Tools

### Environment Testing

```python
def test_environment(env, n_steps=100):
    """Comprehensive environment test."""
    print("🧪 Testing environment...")
    
    # Test reset
    try:
        obs, info = env.reset()
        print(f"✅ Reset successful. Obs shape: {obs.shape}")
    except Exception as e:
        print(f"❌ Reset failed: {e}")
        return
    
    # Test steps
    rewards = []
    for i in range(n_steps):
        try:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            
            if terminated or truncated:
                obs, info = env.reset()
                
        except Exception as e:
            print(f"❌ Step {i} failed: {e}")
            break
    
    # Report statistics
    if rewards:
        print(f"✅ Completed {len(rewards)} steps")
        print(f"📊 Reward - Mean: {np.mean(rewards):.3f}, Std: {np.std(rewards):.3f}")
        print(f"📊 Reward - Min: {np.min(rewards):.3f}, Max: {np.max(rewards):.3f}")
    
    env.close()

# Usage
env = GenesisGymWrapper()
test_environment(env)
```

### Training Diagnostics

```python
def diagnose_training(model, env, n_episodes=5):
    """Diagnose training issues."""
    print("🔍 Training diagnostics...")
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(200):
            # Get action from policy
            action, _ = model.predict(obs, deterministic=False)
            
            # Check action validity
            if not env.action_space.contains(action):
                print(f"⚠️ Invalid action: {action}")
                action = np.clip(action, env.action_space.low, env.action_space.high)
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            # Check for issues
            if np.isnan(obs).any():
                print(f"⚠️ NaN in observation at step {step}")
            if np.isnan(reward):
                print(f"⚠️ NaN reward at step {step}")
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode}: Reward={episode_reward:.2f}, Length={episode_length}")
    
    # Summary
    print(f"\n📊 Diagnostics Summary:")
    print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
```

### Performance Profiling

```python
import time
import cProfile

def profile_environment(env, n_steps=1000):
    """Profile environment performance."""
    print("⏱️ Profiling environment...")
    
    # Reset profiling
    obs, _ = env.reset()
    
    # Profile step function
    def run_steps():
        for _ in range(n_steps):
            action = env.action_space.sample()
            env.step(action)
    
    start_time = time.time()
    cProfile.run('run_steps()', 'profile_stats')
    end_time = time.time()
    
    fps = n_steps / (end_time - start_time)
    print(f"🚀 Performance: {fps:.0f} steps/second")
    
    # Load and display profile
    import pstats
    stats = pstats.Stats('profile_stats')
    stats.sort_stats('tottime').print_stats(10)
```

## 🆘 Getting Help

### Enable Debug Mode

```python
import os
os.environ['GENESIS_DEBUG'] = '1'

# More verbose Genesis logging
import genesis as gs
gs.init(backend=gs.cuda, logging_level='DEBUG')
```

### Collect System Info

```python
def collect_system_info():
    """Collect system information for bug reports."""
    import platform
    import torch
    import genesis as gs
    import gymnasium
    import stable_baselines3
    
    print("🖥️ System Information:")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    print(f"Genesis: {gs.__version__}")
    print(f"Gymnasium: {gymnasium.__version__}")
    print(f"Stable-Baselines3: {stable_baselines3.__version__}")

collect_system_info()
```

### Minimal Reproduction

When reporting issues, provide a minimal example:

```python
"""
Minimal reproduction script for Genesis issues.
Run with: uv run python minimal_repro.py
"""

import genesis as gs
import numpy as np
from genesis_gym_wrapper import GenesisGymWrapper

def main():
    print("🔬 Minimal reproduction test")
    
    try:
        # Initialize Genesis
        gs.init(backend=gs.cuda)
        print("✅ Genesis initialized")
        
        # Create environment
        env = GenesisGymWrapper(use_gpu=True, show_viewer=False)
        print("✅ Environment created")
        
        # Test basic functionality
        obs, _ = env.reset()
        print(f"✅ Reset successful: {obs.shape}")
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✅ Step successful: reward={reward:.3f}")
        
        env.close()
        print("✅ Test completed successfully")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
```

## 📞 Support Resources

1. **GitHub Issues**: [Report bugs and get help](https://github.com/anthropics/claude-code/issues)
2. **Genesis Documentation**: [Official Genesis docs](https://genesis-embodied-ai.github.io/)
3. **Gymnasium Documentation**: [Gymnasium API reference](https://gymnasium.farama.org/)
4. **Stable-Baselines3**: [SB3 documentation](https://stable-baselines3.readthedocs.io/)

## 🔄 Common Fix Checklist

When encountering issues, try these steps in order:

1. ✅ **Restart Python process** (fixes Genesis initialization issues)
2. ✅ **Check GPU availability** (`nvidia-smi` and `torch.cuda.is_available()`)
3. ✅ **Verify dependencies** (run `uv sync` or `pip install -e .`)
4. ✅ **Test minimal example** (use the reproduction script above)
5. ✅ **Check system resources** (GPU memory, disk space)
6. ✅ **Try CPU fallback** (set `use_gpu=False`)
7. ✅ **Clear caches** (`torch.cuda.empty_cache()`)
8. ✅ **Update packages** (`uv sync --upgrade`)

Most issues are resolved by the first few steps!