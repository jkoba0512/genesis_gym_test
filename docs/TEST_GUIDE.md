# Test Guide for Genesis-Gymnasium Integration

## Overview

This directory contains comprehensive tests for validating the Genesis-Gymnasium integration implementation.

## Test Files

### 1. `quick_test.py` - Fast Validation
**Purpose**: Quick smoke test to verify basic functionality  
**Usage**: `python quick_test.py` or `./quick_test.py`  
**Runtime**: ~5 seconds  
**Best for**: Development iteration and CI/CD

```bash
# Quick validation during development
./quick_test.py
```

### 2. `test_genesis_gym_integration.py` - Comprehensive Testing
**Purpose**: Full test suite covering all aspects of integration  
**Usage**: `python test_genesis_gym_integration.py`  
**Runtime**: ~30 seconds  
**Best for**: Thorough validation and debugging

```bash
# Complete test suite
python test_genesis_gym_integration.py
```

### 3. `example_usage.py` - Usage Examples
**Purpose**: Demonstrates how to use the implementation  
**Usage**: `python example_usage.py`  
**Runtime**: ~10 seconds  
**Best for**: Learning and benchmarking

```bash
# See usage examples and performance
python example_usage.py
```

## Test Development Workflow

### Before Implementation
1. Run tests to see what's missing:
```bash
./quick_test.py  # Should show missing implementations
```

### During Implementation
1. Implement core wrapper
2. Test immediately:
```bash
./quick_test.py  # Should show progress
```

### After Implementation
1. Run full validation:
```bash
./quick_test.py                          # Should pass
python test_genesis_gym_integration.py   # Should pass all tests
python example_usage.py                  # Should show working examples
```

## Expected Test Results

### Before Implementation
```
❌ GenesisGymWrapper not found
❌ ReachingEnv not found
⚠️  Implementation not ready
```

### After Implementation
```
✅ Genesis imported
✅ Gymnasium imported  
✅ GenesisGymWrapper found
✅ ReachingEnv found
✅ Basic environment workflow works
✅ Episode completed (X steps)
🎉 All quick tests PASSED!
```

## Test Categories

### Framework Tests
- Genesis world creation and robot loading
- Gymnasium interface compliance
- Space definitions and sampling

### Integration Tests
- Wrapper instantiation and functionality
- Action/observation space mapping
- Reset and step functionality
- Episode completion

### Performance Tests
- Simulation speed benchmarks
- Memory usage validation
- Parallel execution testing

### Task-Specific Tests
- Reaching environment logic
- Reward computation validation
- Termination condition testing

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Install dependencies
   uv add genesis-world gymnasium numpy torch
   ```

2. **Genesis API Changes**
   - Check Genesis documentation: https://genesis-world.readthedocs.io/
   - Update API calls in wrapper implementation

3. **Performance Issues**
   - Verify GPU acceleration is enabled
   - Check Genesis world configuration
   - Profile simulation step timing

### Debug Mode
Add debug prints to see what's happening:
```python
# In your implementation
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Continuous Integration

For automated testing:
```bash
# In CI pipeline
./quick_test.py && echo "Quick tests passed" || exit 1
python test_genesis_gym_integration.py && echo "Full tests passed" || exit 1
```

## Performance Benchmarks

Expected performance targets:
- **Reset speed**: >50 resets/sec
- **Step speed**: >100 steps/sec  
- **Memory usage**: <2GB for single environment
- **Episode time**: <10 seconds for simple tasks

Run benchmarks:
```bash
python example_usage.py  # Includes performance section
```

## Next Steps

Once all tests pass:
1. Implement more complex environments
2. Add RL training examples
3. Optimize for production use
4. Scale to parallel environments