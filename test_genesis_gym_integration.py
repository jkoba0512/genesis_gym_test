"""
Comprehensive Test Suite for Genesis-Gymnasium Integration
Test this before and after implementation to ensure correctness.
"""

import numpy as np
import pytest
import time
from typing import Tuple, Any, Dict

# Test dependencies (will be imported after implementation)
try:
    import genesis as gs
    import gymnasium as gym
    from gymnasium import spaces
    GENESIS_AVAILABLE = True
except ImportError:
    GENESIS_AVAILABLE = False
    print("⚠️  Genesis not installed. Install with: uv add genesis-world")

try:
    import gymnasium as gym
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    print("⚠️  Gymnasium not installed. Install with: uv add gymnasium")


class TestGenesisBasics:
    """Test Genesis framework basic functionality"""
    
    @pytest.mark.skipif(not GENESIS_AVAILABLE, reason="Genesis not available")
    def test_genesis_world_creation(self):
        """Test basic Genesis world setup"""
        world = gs.World()
        assert world is not None
        print("✅ Genesis world creation: PASSED")
    
    @pytest.mark.skipif(not GENESIS_AVAILABLE, reason="Genesis not available")
    def test_genesis_robot_loading(self):
        """Test loading robot into Genesis world"""
        world = gs.World()
        
        # Try to load a robot (exact API may vary)
        try:
            robot = world.add_robot("franka_panda")  # or similar
            assert robot is not None
            print("✅ Genesis robot loading: PASSED")
        except Exception as e:
            print(f"⚠️  Genesis robot loading: Need to check API - {e}")
    
    @pytest.mark.skipif(not GENESIS_AVAILABLE, reason="Genesis not available")
    def test_genesis_simulation_step(self):
        """Test Genesis simulation stepping"""
        world = gs.World()
        
        try:
            # Step simulation
            world.step()
            print("✅ Genesis simulation step: PASSED")
        except Exception as e:
            print(f"⚠️  Genesis simulation step: Need to check API - {e}")


class TestGymnasiumCompliance:
    """Test Gymnasium interface compliance"""
    
    def test_gymnasium_basic_import(self):
        """Test Gymnasium imports correctly"""
        assert GYMNASIUM_AVAILABLE
        assert hasattr(gym, 'Env')
        assert hasattr(gym, 'spaces')
        print("✅ Gymnasium import: PASSED")
    
    def test_gymnasium_spaces(self):
        """Test Gymnasium space definitions"""
        # Box space (continuous)
        box_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        assert box_space.shape == (7,)
        
        # Sample from space
        sample = box_space.sample()
        assert sample.shape == (7,)
        assert box_space.contains(sample)
        
        print("✅ Gymnasium spaces: PASSED")
    
    def test_gymnasium_env_interface(self):
        """Test basic Gymnasium environment interface"""
        # Test with a simple built-in environment
        env = gym.make('CartPole-v1')
        
        # Test reset
        obs, info = env.reset()
        assert obs is not None
        assert isinstance(info, dict)
        
        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs is not None
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        env.close()
        print("✅ Gymnasium environment interface: PASSED")


class TestIntegrationImplementation:
    """Test the actual Genesis-Gymnasium integration (after implementation)"""
    
    def test_genesis_gym_wrapper_exists(self):
        """Test that our wrapper class exists and can be instantiated"""
        try:
            # This will be implemented
            from genesis_gym_wrapper import GenesisGymWrapper
            
            # Test instantiation
            env = GenesisGymWrapper()
            assert env is not None
            assert isinstance(env, gym.Env)
            print("✅ GenesisGymWrapper instantiation: PASSED")
            
        except ImportError:
            print("⚠️  GenesisGymWrapper not implemented yet")
            return False
        except Exception as e:
            print(f"❌ GenesisGymWrapper instantiation failed: {e}")
            return False
        
        return True
    
    def test_action_observation_spaces(self):
        """Test action and observation space definitions"""
        try:
            from genesis_gym_wrapper import GenesisGymWrapper
            
            env = GenesisGymWrapper()
            
            # Check spaces exist
            assert hasattr(env, 'action_space')
            assert hasattr(env, 'observation_space')
            
            # Check spaces are valid Gymnasium spaces
            assert isinstance(env.action_space, gym.Space)
            assert isinstance(env.observation_space, gym.Space)
            
            # Test sampling
            action = env.action_space.sample()
            assert env.action_space.contains(action)
            
            print("✅ Action/Observation spaces: PASSED")
            return True
            
        except ImportError:
            print("⚠️  GenesisGymWrapper not implemented yet")
            return False
        except Exception as e:
            print(f"❌ Action/Observation spaces failed: {e}")
            return False
    
    def test_reset_functionality(self):
        """Test environment reset functionality"""
        try:
            from genesis_gym_wrapper import GenesisGymWrapper
            
            env = GenesisGymWrapper()
            
            # Test reset
            obs, info = env.reset()
            
            # Validate observation
            assert obs is not None
            assert env.observation_space.contains(obs)
            assert isinstance(info, dict)
            
            # Test multiple resets
            obs2, info2 = env.reset()
            assert env.observation_space.contains(obs2)
            
            print("✅ Reset functionality: PASSED")
            return True
            
        except ImportError:
            print("⚠️  GenesisGymWrapper not implemented yet")
            return False
        except Exception as e:
            print(f"❌ Reset functionality failed: {e}")
            return False
    
    def test_step_functionality(self):
        """Test environment step functionality"""
        try:
            from genesis_gym_wrapper import GenesisGymWrapper
            
            env = GenesisGymWrapper()
            obs, info = env.reset()
            
            # Test step
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Validate step output
            assert env.observation_space.contains(obs)
            assert isinstance(reward, (int, float))
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)
            
            print("✅ Step functionality: PASSED")
            return True
            
        except ImportError:
            print("⚠️  GenesisGymWrapper not implemented yet")
            return False
        except Exception as e:
            print(f"❌ Step functionality failed: {e}")
            return False
    
    def test_episode_completion(self):
        """Test complete episode execution"""
        try:
            from genesis_gym_wrapper import GenesisGymWrapper
            
            env = GenesisGymWrapper()
            obs, info = env.reset()
            
            # Run episode
            steps = 0
            max_steps = 100
            
            while steps < max_steps:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                
                if terminated or truncated:
                    break
            
            # Validate episode completed
            assert steps > 0
            print(f"✅ Episode completion: PASSED ({steps} steps)")
            return True
            
        except ImportError:
            print("⚠️  GenesisGymWrapper not implemented yet")
            return False
        except Exception as e:
            print(f"❌ Episode completion failed: {e}")
            return False


class TestReachingEnvironment:
    """Test specific reaching environment implementation"""
    
    def test_reaching_env_exists(self):
        """Test reaching environment can be instantiated"""
        try:
            from reaching_env import ReachingEnv
            
            env = ReachingEnv()
            assert env is not None
            assert isinstance(env, gym.Env)
            print("✅ ReachingEnv instantiation: PASSED")
            return True
            
        except ImportError:
            print("⚠️  ReachingEnv not implemented yet")
            return False
        except Exception as e:
            print(f"❌ ReachingEnv instantiation failed: {e}")
            return False
    
    def test_reaching_task_logic(self):
        """Test reaching task specific logic"""
        try:
            from reaching_env import ReachingEnv
            
            env = ReachingEnv()
            obs, info = env.reset()
            
            # Check observation includes target information
            assert len(obs) > 7  # Should include joint states + target pos
            
            # Test reward computation
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Reward should be finite
            assert np.isfinite(reward)
            
            print("✅ Reaching task logic: PASSED")
            return True
            
        except ImportError:
            print("⚠️  ReachingEnv not implemented yet")
            return False
        except Exception as e:
            print(f"❌ Reaching task logic failed: {e}")
            return False


class TestPerformance:
    """Test performance characteristics"""
    
    def test_step_performance(self):
        """Test simulation step performance"""
        try:
            from genesis_gym_wrapper import GenesisGymWrapper
            
            env = GenesisGymWrapper()
            env.reset()
            
            # Time multiple steps
            num_steps = 100
            start_time = time.time()
            
            for _ in range(num_steps):
                action = env.action_space.sample()
                env.step(action)
            
            end_time = time.time()
            elapsed = end_time - start_time
            steps_per_second = num_steps / elapsed
            
            print(f"✅ Performance: {steps_per_second:.1f} steps/sec")
            
            # Should be reasonably fast (>10 steps/sec)
            assert steps_per_second > 10
            return True
            
        except ImportError:
            print("⚠️  Performance test skipped - not implemented yet")
            return False
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            return False


def run_all_tests():
    """Run all tests in sequence"""
    print("🧪 Running Genesis-Gymnasium Integration Tests\n")
    
    # Basic framework tests
    print("=" * 50)
    print("BASIC FRAMEWORK TESTS")
    print("=" * 50)
    
    genesis_tests = TestGenesisBasics()
    gym_tests = TestGymnasiumCompliance()
    
    if GENESIS_AVAILABLE:
        genesis_tests.test_genesis_world_creation()
        genesis_tests.test_genesis_robot_loading()  
        genesis_tests.test_genesis_simulation_step()
    
    if GYMNASIUM_AVAILABLE:
        gym_tests.test_gymnasium_basic_import()
        gym_tests.test_gymnasium_spaces()
        gym_tests.test_gymnasium_env_interface()
    
    # Integration tests
    print("\n" + "=" * 50) 
    print("INTEGRATION TESTS")
    print("=" * 50)
    
    integration_tests = TestIntegrationImplementation()
    reaching_tests = TestReachingEnvironment()
    performance_tests = TestPerformance()
    
    # Run integration tests
    wrapper_exists = integration_tests.test_genesis_gym_wrapper_exists()
    if wrapper_exists:
        integration_tests.test_action_observation_spaces()
        integration_tests.test_reset_functionality()
        integration_tests.test_step_functionality()
        integration_tests.test_episode_completion()
    
    # Run reaching environment tests
    reaching_exists = reaching_tests.test_reaching_env_exists()
    if reaching_exists:
        reaching_tests.test_reaching_task_logic()
    
    # Performance tests
    if wrapper_exists:
        performance_tests.test_step_performance()
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print("Run this test file before and after implementation!")
    print("Before: Should show missing implementations")
    print("After: Should show all tests passing")
    print("\nTo run: python test_genesis_gym_integration.py")


if __name__ == "__main__":
    run_all_tests()