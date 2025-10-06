# Contributing to Traffic Flow Prediction

Thank you for your interest in contributing to the Traffic Flow Prediction project! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Issues

Before creating an issue, please:
1. Check if the issue already exists
2. Use the issue templates provided
3. Provide detailed information about the problem

### Suggesting Enhancements

We welcome suggestions for:
- New model architectures
- Performance improvements
- Additional datasets
- Documentation improvements
- UI/UX enhancements

### Code Contributions

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Add tests** (if applicable)
5. **Update documentation**
6. **Commit your changes**: `git commit -m 'Add amazing feature'`
7. **Push to the branch**: `git push origin feature/amazing-feature`
8. **Open a Pull Request**

## 🛠️ Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- pip or conda

### Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/traffic-flow-prediction.git
cd traffic-flow-prediction

# Add upstream remote
git remote add upstream https://github.com/originalowner/traffic-flow-prediction.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available
```

### Development Dependencies

```bash
pip install pytest pytest-cov black flake8 mypy jupyter
```

## 📝 Coding Standards

### Python Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write docstrings for all functions and classes
- Keep functions small and focused

### Code Formatting

We use Black for code formatting:

```bash
black src/ scripts/ tests/
```

### Linting

We use flake8 for linting:

```bash
flake8 src/ scripts/ tests/
```

### Type Checking

We use mypy for type checking:

```bash
mypy src/ scripts/
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_models.py
```

### Writing Tests

- Write tests for new functionality
- Aim for high test coverage
- Use descriptive test names
- Test edge cases and error conditions

### Test Structure

```
tests/
├── test_models.py          # Model tests
├── test_data_loading.py    # Data loading tests
├── test_preprocessing.py   # Preprocessing tests
└── test_utils.py          # Utility function tests
```

## 📚 Documentation

### Code Documentation

- Use docstrings for all public functions and classes
- Follow Google docstring format
- Include type hints
- Provide usage examples

### README Updates

- Update README.md for significant changes
- Include new features in the feature list
- Update installation instructions if needed
- Add new dependencies to requirements.txt

### API Documentation

- Document all public APIs
- Include parameter descriptions
- Provide return value information
- Add usage examples

## 🏗️ Project Structure

### Adding New Models

1. Create model class in `src/models/`
2. Add training logic in `src/main.py`
3. Include evaluation metrics
4. Update dashboard visualization
5. Add tests

### Adding New Datasets

1. Add dataset loading function in `src/data_loading.py`
2. Update preprocessing pipeline
3. Modify model input dimensions
4. Test with new data format
5. Update documentation

### Adding New Features

1. Create feature branch
2. Implement feature
3. Add tests
4. Update documentation
5. Create pull request

## 🔍 Pull Request Process

### Before Submitting

- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] No merge conflicts
- [ ] Commit messages are clear

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added (if applicable)
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No merge conflicts
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests and linting
2. **Code Review**: Maintainers review the code
3. **Testing**: Manual testing if needed
4. **Approval**: At least one approval required
5. **Merge**: Squash and merge to main branch

## 🐛 Bug Reports

When reporting bugs, please include:

- **Description**: Clear description of the bug
- **Steps to Reproduce**: Detailed steps to reproduce
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Environment**: OS, Python version, package versions
- **Screenshots**: If applicable
- **Logs**: Error messages or logs

## 💡 Feature Requests

When requesting features, please include:

- **Description**: Clear description of the feature
- **Use Case**: Why this feature would be useful
- **Proposed Solution**: How you think it should work
- **Alternatives**: Other solutions you've considered
- **Additional Context**: Any other relevant information

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Email**: [your.email@example.com]
- **Discord/Slack**: [if available]

## 🏆 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation
- GitHub contributors page

## 📋 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors.

### Expected Behavior

- Be respectful and inclusive
- Use welcoming and inclusive language
- Be respectful of differing viewpoints
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy towards other community members

### Unacceptable Behavior

- Harassment, trolling, or inappropriate comments
- Personal attacks or political discussions
- Public or private harassment
- Publishing private information without permission
- Other unprofessional conduct

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the Traffic Flow Prediction project! 🚗
