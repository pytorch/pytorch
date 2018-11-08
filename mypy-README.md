### Optional type checking with mypy

mypy is an optional static typechecker that works with Python 3.
To use it, install the following dependencies:
```bash
# Install dependencies
pip install mypy mypy-extensions

# Run type checker in the pytorch/ directory
mypy @mypy-files.txt
```
