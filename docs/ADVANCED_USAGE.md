# Advanced Usage Guide

Advanced techniques and use cases for the Genesis-Gymnasium integration.

## 🚀 Beyond Basic Training

This guide covers sophisticated applications including multi-robot environments, custom robots, vision integration, and advanced RL techniques.

## Custom Robot Integration

### Adding New Robot Types

```python
class GenesisGymWrapperExtended(GenesisGymWrapper):
    def _setup_robot(self):
        """Load different robot types."""
        if self.robot_type == "franka":
            self.robot = self.scene.add_entity(
                gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml')
            )
            self.n_ctrl_dofs = 7
            self.joint_limits_low = np.array([-2.8973, -1.7628, -2.8973, -3.0718, 
                                            -2.8973, -0.0175, -2.8973])
            self.joint_limits_high = np.array([2.8973, 1.7628, 2.8973, -0.0698,
                                             2.8973, 3.7525, 2.8973])
                                             
        elif self.robot_type == "ur5":
            self.robot = self.scene.add_entity(
                gs.morphs.MJCF(file='xml/universal_robots_ur5e/ur5e.xml')
            )
            self.n_ctrl_dofs = 6
            self.joint_limits_low = np.array([-2*np.pi] * 6)
            self.joint_limits_high = np.array([2*np.pi] * 6)
            
        elif self.robot_type == "go1":
            # Quadruped robot
            self.robot = self.scene.add_entity(
                gs.morphs.MJCF(file='xml/unitree_go1/go1.xml')
            )
            self.n_ctrl_dofs = 12  # 4 legs × 3 joints
            self.joint_limits_low = np.array([-1.0] * 12)
            self.joint_limits_high = np.array([1.0] * 12)
            
        elif self.robot_type == "g1_humanoid":
            # Humanoid robot
            self.robot = self.scene.add_entity(
                gs.morphs.MJCF(file='xml/unitree_g1/g1.xml')
            )
            self.n_ctrl_dofs = 23  # Full body control
            self.joint_limits_low = np.array([-2.0] * 23)
            self.joint_limits_high = np.array([2.0] * 23)
            
        else:
            raise ValueError(f"Robot type '{self.robot_type}' not supported")
```

### Custom Robot Morphologies

```python
class CustomRobotEnv(GenesisGymWrapper):
    def _setup_robot(self):
        """Create a custom robot from scratch."""
        
        # Create robot base
        base = gs.morphs.Box(size=(0.2, 0.2, 0.1))
        
        # Create arm links
        link1 = gs.morphs.Cylinder(radius=0.02, height=0.3)
        link2 = gs.morphs.Cylinder(radius=0.015, height=0.25)
        link3 = gs.morphs.Cylinder(radius=0.01, height=0.2)
        
        # Assemble robot with joints
        self.robot = self.scene.add_entity(
            gs.morphs.Articulation(
                links=[base, link1, link2, link3],
                joints=[
                    gs.joints.RevoluteJoint(axis=(0, 0, 1)),  # Base rotation
                    gs.joints.RevoluteJoint(axis=(0, 1, 0)),  # Shoulder
                    gs.joints.RevoluteJoint(axis=(0, 1, 0)),  # Elbow
                ],
                joint_limits=[
                    (-np.pi, np.pi),
                    (-np.pi/2, np.pi/2),
                    (-np.pi, 0)
                ]
            )
        )
        
        self.n_ctrl_dofs = 3
```

## Vision Integration

### Adding Camera Observations

```python
class VisualReachingEnv(GenesisGymWrapper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.img_width = 84
        self.img_height = 84
        
    def _setup_scene(self):
        """Add camera to the scene."""
        super()._setup_scene()
        
        # Add camera
        self.camera = self.scene.add_camera(
            pos=(0.8, 0.0, 0.6),
            lookat=(0.0, 0.0, 0.2),
            fov=45,
            width=self.img_width,
            height=self.img_height
        )
    
    def _setup_spaces(self):
        """Define spaces with vision observations."""
        # Action space (unchanged)
        self.action_space = spaces.Box(
            low=self.joint_limits_low,
            high=self.joint_limits_high,
            shape=(self.n_ctrl_dofs,),
            dtype=np.float32
        )
        
        # Observation space: image + proprioception
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0, high=255,
                shape=(self.img_height, self.img_width, 3),
                dtype=np.uint8
            ),
            'proprioception': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.n_ctrl_dofs * 2,),  # pos + vel
                dtype=np.float32
            )
        })
    
    def _get_observation(self):
        """Get visual + proprioceptive observations."""
        # Get proprioceptive data
        joint_pos = self.robot.get_dofs_position()[:self.n_ctrl_dofs]
        joint_vel = self.robot.get_dofs_velocity()[:self.n_ctrl_dofs]
        
        # Convert tensors to numpy
        if hasattr(joint_pos, 'cpu'):
            joint_pos = joint_pos.cpu().numpy()
        if hasattr(joint_vel, 'cpu'):
            joint_vel = joint_vel.cpu().numpy()
            
        proprioception = np.concatenate([joint_pos, joint_vel]).astype(np.float32)
        
        # Get camera image
        image = self.camera.get_image()
        if hasattr(image, 'cpu'):
            image = image.cpu().numpy()
        
        # Convert to uint8 format
        image = (image * 255).astype(np.uint8)
        
        return {
            'image': image,
            'proprioception': proprioception
        }
```

### Using CNN Policies

```python
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn

class VisualFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        
        # Image encoder
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute CNN output size
        with torch.no_grad():
            sample_image = torch.zeros(1, 3, 84, 84)
            cnn_output_size = self.cnn(sample_image).shape[1]
        
        # Proprioception encoder
        proprio_dim = observation_space['proprioception'].shape[0]
        self.proprio_net = nn.Sequential(
            nn.Linear(proprio_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(cnn_output_size + 64, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations):
        # Process image
        image = observations['image'].float() / 255.0
        image = image.permute(0, 3, 1, 2)  # NHWC -> NCHW
        visual_features = self.cnn(image)
        
        # Process proprioception
        proprio_features = self.proprio_net(observations['proprioception'])
        
        # Fuse features
        combined = torch.cat([visual_features, proprio_features], dim=1)
        return self.fusion(combined)

# Train with visual features
policy_kwargs = dict(
    features_extractor_class=VisualFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=256),
)

model = PPO(
    "MultiInputPolicy",  # For dict observations
    visual_env,
    policy_kwargs=policy_kwargs,
    device="cuda"
)
```

## Multi-Robot Environments

### Parallel Manipulation

```python
class MultiArmEnv(GenesisGymWrapper):
    def __init__(self, n_robots=2, **kwargs):
        self.n_robots = n_robots
        super().__init__(**kwargs)
    
    def _setup_robot(self):
        """Setup multiple robot arms."""
        self.robots = []
        
        for i in range(self.n_robots):
            # Position robots in a line
            pos = (i * 0.8 - (self.n_robots-1) * 0.4, 0.0, 0.0)
            
            robot = self.scene.add_entity(
                gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
                pos=pos
            )
            self.robots.append(robot)
        
        self.n_ctrl_dofs = 7 * self.n_robots  # 7 DOF per robot
    
    def _setup_spaces(self):
        """Define spaces for multiple robots."""
        # Action space: joint positions for all robots
        low = np.tile(self.joint_limits_low, self.n_robots)
        high = np.tile(self.joint_limits_high, self.n_robots)
        
        self.action_space = spaces.Box(
            low=low, high=high, 
            shape=(self.n_ctrl_dofs,), 
            dtype=np.float32
        )
        
        # Observation space: joint data for all robots
        obs_dim = self.n_ctrl_dofs * 2  # positions + velocities
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
    
    def _apply_action(self, action):
        """Apply actions to all robots."""
        action = action.reshape(self.n_robots, 7)  # Split by robot
        
        for i, robot in enumerate(self.robots):
            robot.control_dofs_position(
                position=action[i],
                dofs_idx_local=np.arange(7)
            )
    
    def _get_observation(self):
        """Get observations from all robots."""
        observations = []
        
        for robot in self.robots:
            joint_pos = robot.get_dofs_position()[:7]
            joint_vel = robot.get_dofs_velocity()[:7]
            
            # Convert tensors
            if hasattr(joint_pos, 'cpu'):
                joint_pos = joint_pos.cpu().numpy()
            if hasattr(joint_vel, 'cpu'):
                joint_vel = joint_vel.cpu().numpy()
            
            observations.extend([joint_pos, joint_vel])
        
        return np.concatenate(observations).astype(np.float32)
```

### Cooperative Tasks

```python
class CooperativeAssemblyEnv(MultiArmEnv):
    def __init__(self, **kwargs):
        super().__init__(n_robots=2, **kwargs)
        self.object_target_pos = np.array([0.0, 0.0, 0.5])
    
    def _setup_world(self):
        """Add objects for assembly task."""
        super()._setup_world()
        
        # Add object to be assembled
        self.assembly_object = self.scene.add_entity(
            gs.morphs.Box(size=(0.1, 0.05, 0.05)),
            pos=(0.0, 0.0, 0.2)
        )
        
        # Target location
        self.target_marker = self.scene.add_entity(
            gs.morphs.Sphere(radius=0.02),
            pos=self.object_target_pos
        )
    
    def _compute_reward(self, obs, action):
        """Compute cooperative reward."""
        # Get object position
        obj_pos = self.assembly_object.get_pos()
        if hasattr(obj_pos, 'cpu'):
            obj_pos = obj_pos.cpu().numpy()
        
        # Distance to target
        distance = np.linalg.norm(obj_pos - self.object_target_pos)
        distance_reward = -distance * 5.0
        
        # Cooperation bonus (both robots near object)
        robot_positions = []
        for robot in self.robots:
            ee_pos = self._get_end_effector_pos(robot)
            robot_positions.append(ee_pos)
        
        robot_distance = np.linalg.norm(robot_positions[0] - robot_positions[1])
        cooperation_bonus = 2.0 if robot_distance < 0.3 else 0.0
        
        # Action coordination penalty
        action1, action2 = action[:7], action[7:]
        coordination_penalty = -0.1 * np.sum(np.square(action1 - action2))
        
        return distance_reward + cooperation_bonus + coordination_penalty
```

## Advanced RL Techniques

### Hierarchical Reinforcement Learning

```python
class HierarchicalReachingEnv(GenesisGymWrapper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.skill_duration = 10  # Steps per high-level action
        self.current_skill = None
        self.skill_timer = 0
        
    def _setup_spaces(self):
        """Define hierarchical action space."""
        # High-level actions: discrete skills
        self.high_level_action_space = spaces.Discrete(4)  # reach, grasp, lift, place
        
        # Low-level actions: continuous joint control
        self.low_level_action_space = spaces.Box(
            low=self.joint_limits_low,
            high=self.joint_limits_high,
            shape=(self.n_ctrl_dofs,),
            dtype=np.float32
        )
        
        # Use low-level action space for training
        self.action_space = self.low_level_action_space
    
    def step_hierarchical(self, high_level_action):
        """Execute hierarchical action."""
        if self.skill_timer == 0:
            self.current_skill = high_level_action
            self.skill_timer = self.skill_duration
        
        # Generate low-level action based on current skill
        low_level_action = self._generate_skill_action(self.current_skill)
        
        # Execute low-level action
        obs, reward, terminated, truncated, info = self.step(low_level_action)
        
        self.skill_timer -= 1
        
        return obs, reward, terminated, truncated, info
    
    def _generate_skill_action(self, skill):
        """Generate primitive actions for each skill."""
        if skill == 0:  # Reach
            target = self.target_pos
            return self._compute_reach_action(target)
        elif skill == 1:  # Grasp
            return self._compute_grasp_action()
        elif skill == 2:  # Lift
            return self._compute_lift_action()
        elif skill == 3:  # Place
            return self._compute_place_action()
```

### Curriculum Learning

```python
class CurriculumManager:
    def __init__(self, env):
        self.env = env
        self.current_level = 0
        self.success_threshold = 0.8
        self.evaluation_episodes = 20
        
    def should_advance(self, success_rate):
        return success_rate >= self.success_threshold
    
    def advance_curriculum(self):
        self.current_level += 1
        print(f"🎓 Advancing to curriculum level {self.current_level}")
        
        if self.current_level == 1:
            # Level 1: Closer targets
            self.env.target_range = 0.3
        elif self.current_level == 2:
            # Level 2: Normal targets
            self.env.target_range = 0.5
        elif self.current_level == 3:
            # Level 3: Distant targets + obstacles
            self.env.target_range = 0.7
            self.env.add_obstacles = True

def train_with_curriculum(env, model, curriculum_manager):
    for stage in range(5):
        print(f"🎯 Training curriculum stage {stage}")
        
        # Train for this stage
        model.learn(total_timesteps=100000)
        
        # Evaluate current performance
        success_rate = evaluate_success_rate(model, env)
        print(f"Success rate: {success_rate:.1%}")
        
        # Advance curriculum if ready
        if curriculum_manager.should_advance(success_rate):
            curriculum_manager.advance_curriculum()
        else:
            print("📚 Continuing current level...")
```

## Sim-to-Real Transfer

### Domain Randomization

```python
class DomainRandomizedEnv(GenesisGymWrapper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.randomize_physics = True
        self.randomize_appearance = True
        
    def reset(self, seed=None, options=None):
        if self.randomize_physics:
            self._randomize_physics()
        
        if self.randomize_appearance:
            self._randomize_appearance()
        
        return super().reset(seed=seed, options=options)
    
    def _randomize_physics(self):
        """Randomize physics parameters."""
        # Gravity variation
        gravity_noise = np.random.uniform(-1.0, 1.0)
        self.scene.set_gravity((0, 0, -9.81 + gravity_noise))
        
        # Joint friction
        friction_multiplier = np.random.uniform(0.5, 2.0)
        for i in range(self.n_ctrl_dofs):
            current_damping = self.robot.get_dofs_damping()[i]
            new_damping = current_damping * friction_multiplier
            self.robot.set_dofs_damping(new_damping, [i])
        
        # Mass variation
        mass_multiplier = np.random.uniform(0.8, 1.2)
        self.robot.set_mass_scale(mass_multiplier)
    
    def _randomize_appearance(self):
        """Randomize visual appearance."""
        # Random lighting
        self.scene.set_ambient_light(
            color=np.random.uniform(0.5, 1.0, 3)
        )
        
        # Random textures/colors
        random_color = np.random.uniform(0.2, 0.8, 3)
        self.robot.set_color(random_color)
```

### Real Robot Interface

```python
class RealRobotInterface:
    def __init__(self, robot_ip="192.168.1.100"):
        self.robot_ip = robot_ip
        self.connect()
    
    def connect(self):
        """Connect to real robot."""
        # Implementation depends on robot API
        pass
    
    def send_action(self, action):
        """Send action to real robot."""
        # Convert sim action to real robot commands
        joint_positions = action
        # Send to robot via API
        pass
    
    def get_observation(self):
        """Get real robot state."""
        # Read from robot sensors
        joint_positions = self._read_joint_positions()
        joint_velocities = self._read_joint_velocities()
        return np.concatenate([joint_positions, joint_velocities])

class SimToRealWrapper:
    def __init__(self, trained_model, real_robot):
        self.model = trained_model
        self.robot = real_robot
        
    def execute_policy(self, n_steps=100):
        """Execute trained policy on real robot."""
        obs = self.robot.get_observation()
        
        for step in range(n_steps):
            # Get action from trained policy
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Send to real robot
            self.robot.send_action(action)
            
            # Get new observation
            obs = self.robot.get_observation()
            
            # Safety checks
            if self._safety_check(obs):
                break
```

## Performance Optimization

### Batch Processing

```python
class BatchedGenesisEnv:
    def __init__(self, n_envs=4, **env_kwargs):
        # Note: Limited by Genesis single initialization
        self.n_envs = min(n_envs, 1)  # Genesis limitation
        self.envs = [GenesisGymWrapper(**env_kwargs)]
        
    def reset(self):
        return [env.reset() for env in self.envs]
    
    def step(self, actions):
        results = []
        for env, action in zip(self.envs, actions):
            results.append(env.step(action))
        return results
```

### Memory Optimization

```python
import gc
import torch

def optimize_memory():
    """Optimize GPU memory usage."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

class MemoryEfficientEnv(GenesisGymWrapper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_buffer = None
        
    def _get_observation(self):
        """Reuse observation buffer to reduce allocations."""
        obs = super()._get_observation()
        
        if self.observation_buffer is None:
            self.observation_buffer = np.zeros_like(obs)
        
        np.copyto(self.observation_buffer, obs)
        return self.observation_buffer
```

## Debugging and Visualization

### Real-time Monitoring

```python
import matplotlib.pyplot as plt
from collections import deque

class TrainingMonitor:
    def __init__(self):
        self.rewards = deque(maxlen=1000)
        self.success_rates = deque(maxlen=100)
        
    def update(self, reward, success):
        self.rewards.append(reward)
        self.success_rates.append(float(success))
        
    def plot_progress(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot rewards
        ax1.plot(list(self.rewards))
        ax1.set_title('Episode Rewards')
        ax1.set_ylabel('Reward')
        
        # Plot success rate
        if len(self.success_rates) > 10:
            window_size = min(50, len(self.success_rates))
            windowed_success = np.convolve(
                list(self.success_rates), 
                np.ones(window_size)/window_size, 
                mode='valid'
            )
            ax2.plot(windowed_success)
        
        ax2.set_title('Success Rate (50-episode average)')
        ax2.set_ylabel('Success Rate')
        ax2.set_xlabel('Episode')
        
        plt.tight_layout()
        plt.show()
```

This advanced guide provides the foundation for sophisticated robotics applications with Genesis. Next, explore domain-specific applications like mobile manipulation, multi-agent systems, or human-robot interaction!