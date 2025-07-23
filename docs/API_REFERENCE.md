# API Reference

Complete documentation for the Genesis-Gymnasium integration classes and methods.

## GenesisGymWrapper

The main wrapper class that provides Gymnasium API compliance for Genesis physics simulation.

### Class Definition

```python
class GenesisGymWrapper(gym.Env):
    """
    Genesis-Gymnasium wrapper for Genesis 0.2.1.
    
    Provides full Gymnasium API compliance with proper Genesis
    0.2.1 robot control, state extraction, and GPU acceleration.
    """
```

### Constructor

```python
def __init__(
    self,
    robot_type: str = "franka",
    scene_type: str = "empty", 
    use_gpu: bool = True,
    show_viewer: bool = False,
    render_mode: Optional[str] = None
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `robot_type` | `str` | `"franka"` | Type of robot to load ("franka") |
| `scene_type` | `str` | `"empty"` | Scene configuration ("empty", "table") |
| `use_gpu` | `bool` | `True` | Whether to use GPU acceleration |
| `show_viewer` | `bool` | `False` | Whether to show Genesis viewer |
| `render_mode` | `Optional[str]` | `None` | Rendering mode for visualization |

#### Example

```python
# Basic usage
env = GenesisGymWrapper()

# Custom configuration
env = GenesisGymWrapper(
    robot_type="franka",
    scene_type="table",
    use_gpu=True,
    show_viewer=True
)
```

### Properties

#### Action Space
```python
env.action_space: spaces.Box
```
- **Shape**: `(7,)` for Franka arm joints
- **Range**: Joint position limits `[-2.8973, 2.8973]` (varies by joint)
- **Type**: `np.float32`

#### Observation Space
```python
env.observation_space: spaces.Box  
```
- **Shape**: `(14,)` - joint positions (7) + velocities (7)
- **Range**: `[-inf, inf]`
- **Type**: `np.float32`

#### Robot Properties
```python
env.n_dofs: int              # Total DOFs (9 for Franka)
env.n_ctrl_dofs: int         # Controllable DOFs (7 for Franka arm)
env.joint_limits_low: np.ndarray   # Lower joint limits
env.joint_limits_high: np.ndarray  # Upper joint limits
env.initial_joint_pos: np.ndarray  # Home position
```

### Core Methods

#### reset()
```python
def reset(
    self, 
    seed: Optional[int] = None, 
    options: Optional[Dict] = None
) -> Tuple[np.ndarray, Dict]
```

Reset environment to initial state.

**Parameters:**
- `seed`: Random seed for reproducibility
- `options`: Additional reset options

**Returns:**
- `observation`: Initial observation array `(14,)`
- `info`: Dictionary with environment information

**Example:**
```python
obs, info = env.reset(seed=42)
print(f"Initial joint positions: {obs[:7]}")
print(f"Robot DOFs: {info['n_dofs']}")
```

#### step()
```python
def step(
    self,
    action: np.ndarray
) -> Tuple[np.ndarray, float, bool, bool, Dict]
```

Execute one environment step.

**Parameters:**
- `action`: Joint positions array `(7,)` in joint space

**Returns:**
- `observation`: New observation after action
- `reward`: Reward for this step
- `terminated`: Whether episode terminated naturally
- `truncated`: Whether episode was truncated (time limit)
- `info`: Additional information dictionary

**Example:**
```python
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

if terminated or truncated:
    obs, info = env.reset()
```

#### render()
```python
def render(self, mode: Optional[str] = None)
```

Render the environment.

**Parameters:**
- `mode`: Rendering mode ("human", "rgb_array")

#### close()
```python
def close(self)
```

Clean up environment resources.

### Internal Methods

#### _apply_action()
```python
def _apply_action(self, action: np.ndarray)
```

Apply action to robot using position control.

#### _get_observation()
```python
def _get_observation(self) -> np.ndarray
```

Extract current robot state as observation.

**Returns:**
- Joint positions and velocities concatenated `(14,)`
- Automatically converts GPU tensors to numpy arrays

#### _compute_reward()
```python
def _compute_reward(self, obs: np.ndarray, action: np.ndarray) -> float
```

Compute reward for current state and action.

**Default Implementation:**
- Position error penalty: `-0.1 * ||pos - home_pos||²`
- Velocity penalty: `-0.01 * ||velocity||²`
- Alive bonus: `+0.1`

#### _check_terminated()
```python
def _check_terminated(self, obs: np.ndarray) -> bool
```

Check if episode should terminate naturally.

**Termination Conditions:**
- Joint positions outside safety limits
- Can be overridden in subclasses

#### _get_info()
```python
def _get_info(self) -> Dict[str, Any]
```

Get additional information about current state.

**Returns:**
```python
{
    "step": current_step,
    "max_steps": max_steps,
    "n_dofs": total_dofs,
    "n_ctrl_dofs": controlled_dofs,
    "joint_positions": current_positions,
    "joint_velocities": current_velocities
}
```

## Usage Patterns

### Basic Environment Loop
```python
env = GenesisGymWrapper(use_gpu=True)

for episode in range(10):
    obs, info = env.reset()
    total_reward = 0
    
    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"Episode {episode}: Total reward = {total_reward:.2f}")

env.close()
```

### Custom Environment Subclass
```python
class MyCustomEnv(GenesisGymWrapper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target_pos = np.array([0.5, 0.0, 0.3])
    
    def _compute_reward(self, obs, action):
        # Custom reward function
        joint_pos = obs[:self.n_ctrl_dofs]
        ee_pos = self._forward_kinematics(joint_pos)
        distance = np.linalg.norm(ee_pos - self.target_pos)
        return -distance * 5.0
    
    def _check_terminated(self, obs):
        # Custom termination condition
        joint_pos = obs[:self.n_ctrl_dofs]
        ee_pos = self._forward_kinematics(joint_pos)
        return np.linalg.norm(ee_pos - self.target_pos) < 0.05
```

## Error Handling

### Common Exceptions

#### GenesisException
```python
genesis.GenesisException: Genesis already initialized
```
**Cause**: Trying to initialize Genesis multiple times in same process
**Solution**: Use initialization checks in wrapper

#### CUDA Errors
```python
RuntimeError: can't convert cuda:0 device type tensor to numpy
```
**Cause**: Trying to convert GPU tensors directly to numpy
**Solution**: Use `.cpu().numpy()` conversion (handled automatically)

### Best Practices

1. **Single Environment per Process**: Genesis can only be initialized once
2. **GPU Memory Management**: Close environments when done
3. **Error Handling**: Wrap environment creation in try-catch blocks
4. **Performance**: Use GPU when available, CPU as fallback

```python
try:
    env = GenesisGymWrapper(use_gpu=True)
    print("✅ GPU environment created")
except Exception as e:
    print(f"⚠️ GPU failed, using CPU: {e}")
    env = GenesisGymWrapper(use_gpu=False)
```

## Integration with RL Libraries

### Stable-Baselines3
```python
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

env = Monitor(GenesisGymWrapper())
model = PPO("MlpPolicy", env, device="cuda")
model.learn(total_timesteps=100000)
```

### Ray RLlib
```python
import ray.rllib.algorithms.ppo as ppo

config = ppo.PPOConfig().environment(
    env=GenesisGymWrapper,
    env_config={"use_gpu": True}
)
```

## Performance Notes

- **GPU Acceleration**: 10-50x speedup over CPU
- **Simulation Speed**: 400+ FPS typical on modern GPUs  
- **Memory Usage**: ~500MB GPU memory for basic scenes
- **Parallelization**: Single environment recommended due to Genesis limitations