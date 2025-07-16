# PyTorch Documentation Guide

Sphinx-based documentation system for generating PyTorch's official documentation website and API references.

## 🏗️ Directory Organization

### Core Documentation (`source/`)
- **`index.md`** - Main documentation homepage
- **`pytorch-api.md`** - Core PyTorch API documentation
- **`torch.rst`** - Main torch module documentation  
- **`nn.rst`** - Neural network module documentation
- **`autograd.md`** - Automatic differentiation documentation

### API Documentation
- **`torch.compiler*.md`** - TorchDynamo and Inductor compiler documentation
- **`distributed*.md`** - Distributed training documentation
- **`onnx*.md`** - ONNX export and interoperability
- **`quantization.rst`** - Model quantization documentation
- **`jit*.rst/md`** - TorchScript JIT compiler documentation

### Developer Guides (`notes/`)
- **`extending.rst`** - Extending PyTorch with custom operators
- **`autograd.rst`** - Deep dive into autograd system
- **`cuda.rst`** - CUDA programming and optimization
- **`fsdp.rst`** - Fully Sharded Data Parallel
- **`ddp.rst`** - DistributedDataParallel

### Community Documentation (`community/`)
- **`contribution_guide.rst`** - How to contribute to PyTorch
- **`governance.rst`** - PyTorch project governance
- **`design.rst`** - Design principles and decisions

### Build System
- **`Makefile`** - Documentation build configuration
- **`make.bat`** - Windows build script
- **`requirements.txt`** - Documentation build dependencies
- **`conf.py`** - Sphinx configuration

### C++ Documentation (`cpp/`)
- **`source/conf.py`** - C++ API documentation configuration
- **`source/Doxyfile`** - Doxygen configuration for C++ docs
- **`Makefile`** - C++ documentation build

### Assets (`_static/`)
- **`img/`** - Documentation images and diagrams
- **`css/`** - Custom CSS styling
- Templates and static assets for documentation rendering

## 🚀 Building Documentation

### Python Documentation
```bash
# Install documentation dependencies
pip install -r docs/requirements.txt

# Build HTML documentation
cd docs
make html

# Clean previous builds
make clean
```

### C++ Documentation
```bash
# Build C++ API documentation
cd docs/cpp
make html
```

### Development Server
```bash
# Serve docs locally for development
cd docs/build/html
python -m http.server 8000
# Visit http://localhost:8000
```

## 📝 Writing Documentation

### Adding New Documentation
1. Create `.md` or `.rst` file in appropriate `source/` subdirectory
2. Add to relevant table of contents (`index.md` or section index)
3. Build and test locally
4. Follow PyTorch documentation style guide

### API Documentation
```python
# Python docstring format for API docs
def my_function(x: Tensor) -> Tensor:
    """Brief description.
    
    Longer description with examples.
    
    Args:
        x: Input tensor description
        
    Returns:
        Output tensor description
        
    Example::
        >>> import torch
        >>> result = my_function(torch.randn(2, 3))
    """
```

### Cross-References
```rst
# Link to other documentation
:func:`torch.add`
:class:`torch.nn.Module`
:doc:`/notes/autograd`
```

## 🔧 Development Workflow

### Documentation Testing
```bash
# Test documentation builds
make html

# Check for coverage
make coverage

# Check for broken links
make linkcheck
```

## 📁 Key Files

### Configuration
- `source/conf.py` - Main Sphinx configuration
- `cpp/source/conf.py` - C++ documentation configuration
- `requirements.txt` - Build dependencies

### Content Organization
- `source/index.md` - Documentation homepage
- `source/notes/` - Developer-focused guides
- `source/community/` - Community and contribution docs
- `source/_templates/` - Custom Sphinx templates

### Build Scripts  
- `Makefile` - Primary build automation
- `make.bat` - Windows build script
- `source/scripts/` - Documentation generation scripts

## 📝 Notes for Claude

This documentation system provides:
- **Comprehensive API coverage**: All PyTorch modules and functions
- **Developer guides**: Extending PyTorch, internals documentation
- **Multi-format support**: RST, Markdown, Jupyter notebooks
- **Cross-references**: Extensive linking between documentation sections
- **Examples**: Executable code examples with output
- **Multi-language**: Python and C++ API documentation

Key documentation areas:
- Core tensor operations and autograd
- Neural network layers and training
- Distributed training and parallelism
- Compiler stack (TorchDynamo, Inductor)
- Mobile and edge deployment
- ONNX interoperability
- Performance optimization guides

The system uses Sphinx with custom extensions for:
- Automatic API documentation generation
- Code example execution and validation
- Cross-platform build support
- Integration with PyTorch's CI system