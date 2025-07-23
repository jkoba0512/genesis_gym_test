# Getting Started with Genesis-Gymnasium Integration

## 🚀 Quick Start Guide

This guide will get you up and running with the Genesis-Gymnasium integration for reinforcement learning in just a few minutes.

## Prerequisites

- **Python 3.10+** 
- **CUDA-compatible GPU** (recommended for best performance)
- **Linux/macOS** (Windows support may vary)

## Installation

### 1. Clone and Setup Environment

```bash
# Navigate to your project directory
cd genesis_gym_test

# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 2. Verify Installation

Test that Genesis works with GPU acceleration:

```bash
uv run python test_simple_genesis_gpu.py
```

You should see output like:
```
🎮 Genesis GPU Verification
✅ Completed 1000 steps in 2.341s
⚡ Performance: 427 steps/sec on GPU!
🎉 Genesis is successfully using your RTX 3060 Ti!
```

## Your First Genesis Environment

### 1. Basic Environment Test

Create a simple test script:

```python
from genesis_gym_wrapper import GenesisGymWrapper

# Create environment
env = GenesisGymWrapper(
    robot_type="franka",
    use_gpu=True,
    show_viewer=False
)

# Reset and run a few steps
obs, info = env.reset()
print(f"Observation shape: {obs.shape}")

for i in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i+1}: reward={reward:.3f}")
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### 2. Run the Complete Demo

Try the working RL demo:

```bash
uv run python single_env_rl_demo.py
```

This will:
- Create a Genesis environment with Franka robot
- Train a PPO agent for 10k steps (~30 seconds)
- Test the trained agent on a reaching task
- Show success rate and performance metrics

Expected output:
```
🎮 Single Environment Genesis RL Demo
✅ Environment ready! Obs shape: (17,)
✅ Training on: cuda
📊 Final Results:
  Success Rate: 20.0% (2/10)
✅ Single environment demo completed!
```

## Understanding the Output

### Performance Metrics
- **FPS**: Simulation frames per second (400+ is excellent)
- **Steps/sec**: RL environment steps per second
- **Success Rate**: Percentage of episodes where robot reaches target

### GPU Acceleration
Genesis automatically uses CUDA if available:
- Look for "Using Genesis CUDA backend (GPU)" message
- Performance should be 10-50x faster than CPU
- Memory usage shown for your GPU

## Next Steps

1. **Explore Examples**: Try `rl_training_example.py` for full training
2. **Customize Environment**: Modify the reaching task parameters
3. **Add Your Robot**: Extend the wrapper for different robots
4. **Scale Up**: Run longer training sessions

## Common First-Time Issues

### Genesis Won't Initialize
```bash
# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test CPU fallback
python -c "import genesis as gs; gs.init(backend=gs.cpu); print('Genesis works!')"
```

### Low Performance
- Ensure GPU is being used (check the initialization message)
- Close other GPU-intensive applications
- Try smaller environments first

### Import Errors
```bash
# Verify all dependencies
uv run python -c "import genesis, gymnasium, stable_baselines3; print('All imports work!')"
```

## Ready to Train!

You're now ready to start training RL agents with Genesis! Check out:
- [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md) - Detailed training instructions
- [`API_REFERENCE.md`](API_REFERENCE.md) - Complete API documentation
- [`EXAMPLES.md`](EXAMPLES.md) - More example use cases

Happy training! 🎯