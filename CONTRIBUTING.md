# Contributing to Genesis-Gymnasium Integration

Thank you for your interest in contributing to the Genesis-Gymnasium Integration project! This project aims to provide high-performance robot learning with Genesis physics simulation and Gymnasium RL environments.

## 🎯 How to Contribute

### 🐛 Bug Reports

If you find a bug, please open an issue with:
- **Clear description** of the problem
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **System information** (OS, GPU, Python version, Genesis version)
- **Minimal code example** demonstrating the issue

### ✨ Feature Requests

We welcome feature requests! Please include:
- **Clear description** of the proposed feature
- **Use case** and motivation
- **Possible implementation approach** (if you have ideas)

### 🔧 Pull Requests

1. **Fork the repository** and create a feature branch
2. **Install development dependencies**:
   ```bash
   uv sync --dev
   ```
3. **Make your changes** with clear, focused commits
4. **Add tests** for new functionality
5. **Run the test suite**:
   ```bash
   uv run python -m pytest
   ```
6. **Update documentation** if needed
7. **Submit a pull request** with a clear description

## 🎨 Development Guidelines

### Code Style
- Follow **PEP 8** Python style guidelines
- Use **type hints** for function parameters and returns
- Write **clear docstrings** for all public functions and classes
- Keep **line length** under 100 characters

### Testing
- Add **unit tests** for new functionality
- Include **integration tests** for complex features
- Test on **GPU and CPU** backends when applicable
- Verify **performance** doesn't regress

### Documentation
- Update **API documentation** for new features
- Add **examples** demonstrating new functionality
- Update the **README.md** if project structure changes
- Keep documentation **clear and concise**

## 🚀 Priority Contribution Areas

We're especially interested in contributions for:

### 🤖 **Robot Integration**
- Support for additional robot morphologies (UR5, Kuka, quadrupeds, humanoids)
- Advanced robot control (force control, compliance, etc.)
- Multi-robot environments and coordination

### 👁️ **Sensor Integration**
- Camera observations and vision processing
- Force/tactile sensor integration
- Multi-modal observation spaces

### 🎓 **RL Algorithms**
- Advanced training techniques (curriculum learning, HER, etc.)
- Multi-agent reinforcement learning
- Hierarchical RL implementations
- Imitation learning and behavior cloning

### 🔄 **Sim-to-Real**
- Domain randomization techniques
- Real robot interface implementations
- Transfer learning improvements

### ⚡ **Performance**
- Multi-environment parallelization
- Memory optimization
- Faster simulation techniques

## 🧪 Testing Your Contributions

Before submitting, please test your changes:

### Basic Functionality
```bash
# Test basic environment creation
uv run python test_simple_genesis_gpu.py

# Test RL integration
uv run python single_env_rl_demo.py

# Run full test suite
uv run python -m pytest test_*.py -v
```

### Performance Testing
```bash
# Benchmark environment performance
uv run python -c "
from genesis_gym_wrapper import GenesisGymWrapper
import time

env = GenesisGymWrapper(use_gpu=True)
start = time.time()
for i in range(1000):
    action = env.action_space.sample()
    env.step(action)
print(f'Performance: {1000/(time.time()-start):.0f} FPS')
"
```

### Documentation Testing
```bash
# Verify all imports work
uv run python -c "from genesis_gym_wrapper import GenesisGymWrapper; print('✅ Import successful')"

# Test example code from documentation
uv run python docs/examples/basic_usage.py
```

## 📋 Development Setup

### Full Development Environment
```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/genesis-gymnasium-integration.git
cd genesis-gymnasium-integration

# Install development dependencies
uv sync --dev

# Install pre-commit hooks (optional but recommended)
uv run pre-commit install

# Verify setup
uv run python test_simple_genesis_gpu.py
```

### GPU Development
For the best development experience, use a system with:
- **NVIDIA GPU** with CUDA support
- **8GB+ GPU memory** for complex environments
- **Python 3.10+**
- **Ubuntu 20.04+** or **macOS** (Windows support varies)

## 📝 Commit Guidelines

### Commit Messages
Use clear, descriptive commit messages:
```
feat: add support for UR5 robot arm
fix: resolve GPU tensor conversion issue in observation extraction
docs: update API reference for new robot parameters
test: add integration tests for multi-robot environments
perf: optimize environment reset for 20% speed improvement
```

### Commit Types
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `test`: Test additions or modifications
- `perf`: Performance improvements
- `refactor`: Code refactoring
- `style`: Code style changes

## 🤝 Community Guidelines

- **Be respectful** and inclusive in all interactions
- **Help newcomers** get started with the project
- **Share knowledge** and explain your approaches
- **Provide constructive feedback** on pull requests
- **Ask questions** when something is unclear

## 📚 Resources

### Project Documentation
- [Getting Started Guide](docs/GETTING_STARTED.md)
- [API Reference](docs/API_REFERENCE.md)
- [Training Guide](docs/TRAINING_GUIDE.md)
- [Advanced Usage](docs/ADVANCED_USAGE.md)

### External Resources
- [Genesis Documentation](https://genesis-embodied-ai.github.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)

## ❓ Questions?

If you have questions about contributing:
- **Open an issue** for general questions
- **Join discussions** on existing issues and PRs
- **Check the documentation** first
- **Look at existing code** for examples

---

**Thank you for contributing to high-performance robot learning!** 🚀

Your contributions help make Genesis-Gymnasium integration better for everyone in the robotics and RL communities.