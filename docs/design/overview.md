# Auto-Running Code Examples in PyTorch Documentation - Design Overview

**Related to Issue:** [#6662](https://github.com/pytorch/pytorch/issues/6662)  
**Branch:** `docs/gen-example-autotest`  
**Author:** DHANUSHRAJA22  
**Date:** August 27, 2025

## Executive Summary

This document outlines a prototype design for automatically running and including code example outputs in PyTorch documentation. The goal is to enhance documentation quality by ensuring code examples are always up-to-date with executable outputs, improving user experience and reducing maintenance overhead.

## Background and Motivation

PyTorch documentation contains numerous code examples across tutorials, API references, and guides. Currently, many code examples show static outputs that may become outdated as the library evolves. This leads to:

- Inconsistent documentation with potentially incorrect output examples
- Manual maintenance burden for documentation contributors
- User confusion when actual outputs differ from documented examples
- Reduced trust in documentation accuracy

## Current State Analysis

### Existing Capabilities in Sphinx/nbsphinx

**Sphinx Documentation System:**
- Supports `doctest` extension for testing code snippets
- Provides `sphinx-gallery` for tutorial generation from Python scripts
- Includes `autodoc` for automatic API documentation generation
- Supports custom directives and extensions

**nbsphinx Integration:**
- Converts Jupyter notebooks to Sphinx documentation
- Automatically includes cell outputs from executed notebooks
- Supports both static and dynamic notebook rendering
- Handles rich media outputs (images, plots, HTML)

**Current PyTorch Documentation Workflow:**
- Uses Sphinx with custom extensions
- Relies on manual code example verification
- Limited automated testing of documentation examples
- Inconsistent output formatting across different documentation sections

### Identified Gaps

1. **No Automated Code Execution**: Documentation examples are not automatically executed during build
2. **Output Synchronization**: No mechanism to ensure documented outputs match actual execution results
3. **Environment Consistency**: No standardized environment for running documentation examples
4. **Error Handling**: No robust system for handling failing code examples during documentation builds
5. **Performance Overhead**: No optimization for build time when executing numerous code examples
6. **Version Compatibility**: No system to test examples across different PyTorch versions

## Proposed Solution: Minimal Prototype Design

### Architecture Overview

```
Documentation Build Pipeline
├── Source Files (.rst, .py, .ipynb)
├── Code Example Extractor
├── Execution Environment Manager  
├── Code Runner & Output Capturer
├── Output Integration & Formatting
└── Final Documentation Generation
```

### Core Components

#### 1. Code Example Extractor
- **Purpose**: Identify and extract executable code blocks from documentation sources
- **Input**: RST files, Python scripts, Jupyter notebooks
- **Output**: Structured list of code examples with metadata
- **Key Features**:
  - Support for different code block formats (```python, .. code-block::, etc.)
  - Metadata extraction (expected output, execution context)
  - Dependency identification

#### 2. Execution Environment Manager
- **Purpose**: Provide consistent, isolated environments for code execution
- **Features**:
  - Docker-based isolation
  - PyTorch version matrix support
  - Dependency management
  - Resource limitation (memory, CPU, execution time)

#### 3. Code Runner & Output Capturer
- **Purpose**: Execute code examples and capture comprehensive outputs
- **Capabilities**:
  - Standard output/error capture
  - Rich media output handling (plots, tensors, models)
  - Execution timing and performance metrics
  - Error handling and graceful degradation

#### 4. Output Integration System
- **Purpose**: Integrate captured outputs back into documentation
- **Features**:
  - Format-aware output rendering
  - Diff detection for output changes
  - Manual override mechanisms
  - Version control integration

### Minimal Script/Test Setup

#### Phase 1: Proof of Concept Script

**File Structure:**
```
docs/tools/auto-example/
├── extract_examples.py     # Extract code blocks from docs
├── execute_examples.py     # Run extracted examples  
├── integrate_outputs.py    # Update documentation with outputs
├── config.yaml            # Configuration settings
├── requirements.txt       # Dependencies
└── test_examples/         # Sample test cases
    ├── basic_tensor.py
    ├── neural_network.py
    └── expected_outputs/
```

**Core Script: `execute_examples.py`**
```python
#!/usr/bin/env python3
"""
Minimal auto-execution prototype for PyTorch documentation examples.
"""

import ast
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

class DocumentationExampleRunner:
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.results = []
    
    def extract_code_blocks(self, rst_file: Path) -> List[str]:
        """Extract Python code blocks from RST files."""
        # Implementation to parse RST and extract code blocks
        pass
    
    def execute_code(self, code: str) -> Tuple[str, str, int]:
        """Execute Python code and capture output."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            return result.stdout, result.stderr, result.returncode
        finally:
            Path(temp_file).unlink()
    
    def process_documentation(self, docs_path: Path) -> Dict:
        """Process all documentation files and execute examples."""
        results = {'passed': 0, 'failed': 0, 'examples': []}
        
        for rst_file in docs_path.rglob('*.rst'):
            code_blocks = self.extract_code_blocks(rst_file)
            
            for code in code_blocks:
                stdout, stderr, returncode = self.execute_code(code)
                
                example_result = {
                    'file': str(rst_file),
                    'code': code,
                    'stdout': stdout,
                    'stderr': stderr,
                    'success': returncode == 0
                }
                
                results['examples'].append(example_result)
                if returncode == 0:
                    results['passed'] += 1
                else:
                    results['failed'] += 1
        
        return results

if __name__ == '__main__':
    runner = DocumentationExampleRunner()
    results = runner.process_documentation(Path('docs/source'))
    print(f"Executed {len(results['examples'])} examples")
    print(f"Passed: {results['passed']}, Failed: {results['failed']}")
```

#### Phase 2: Integration Testing

**Test Cases:**
1. Basic tensor operations
2. Neural network forward pass
3. Data loading examples
4. Model training snippets
5. Visualization code (matplotlib integration)

**Continuous Integration Integration:**
```yaml
# .github/workflows/doc-examples-test.yml
name: Documentation Examples Test

on:
  pull_request:
    paths:
      - 'docs/**'
      - 'torch/**'

jobs:
  test-doc-examples:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install torch torchvision
          pip install -r docs/requirements.txt
      - name: Run documentation example tests
        run: |
          cd docs/tools/auto-example
          python execute_examples.py
```

## Implementation Roadmap

### Phase 1: Prototype Development (2-3 weeks)
- [ ] Implement basic code extraction from RST files
- [ ] Create minimal code execution framework
- [ ] Develop output capture and formatting
- [ ] Test with 10-20 documentation examples
- [ ] Create initial CI integration

### Phase 2: Enhanced Features (3-4 weeks)
- [ ] Add support for Jupyter notebook integration
- [ ] Implement rich media output handling
- [ ] Create execution environment isolation
- [ ] Add performance optimizations
- [ ] Expand test coverage to 100+ examples

### Phase 3: Production Integration (4-6 weeks)
- [ ] Full documentation build pipeline integration
- [ ] Error handling and graceful degradation
- [ ] Manual override systems
- [ ] Performance benchmarking and optimization
- [ ] Documentation and contributor guidelines

## Risk Assessment & Mitigation

### Technical Risks
1. **Build Time Increase**: Mitigation through parallel execution and caching
2. **Environment Dependencies**: Containerization and dependency management
3. **Flaky Tests**: Robust error handling and retry mechanisms
4. **Resource Usage**: Execution timeouts and resource limits

### Process Risks
1. **Contributor Adoption**: Clear documentation and tooling
2. **Maintenance Overhead**: Automated monitoring and alerts
3. **Compatibility Issues**: Version matrix testing

## Success Metrics

### Technical Metrics
- **Coverage**: >80% of documentation code examples automatically executed
- **Accuracy**: <5% false positive rate for example failures
- **Performance**: <20% increase in documentation build time
- **Reliability**: >95% successful execution rate for valid examples

### Quality Metrics
- **Documentation Consistency**: Measurable reduction in output mismatches
- **User Experience**: Reduced bug reports related to documentation examples
- **Maintainer Efficiency**: Reduced time spent on manual example verification

## Next Steps & Commitment

### Immediate Actions (Next 2 weeks)
1. **Implement Core Prototype**: Create working version of minimal script setup
2. **Test with Sample Examples**: Validate approach with 10-15 representative code examples
3. **Performance Baseline**: Measure execution time and resource usage
4. **Documentation**: Create setup and usage instructions

### Deliverables for Core Reviewer Feedback
1. **Working Prototype**: Demonstrable auto-execution system
2. **Test Results**: Execution results for sample documentation examples
3. **Performance Analysis**: Build time impact assessment
4. **Integration Plan**: Detailed steps for full PyTorch docs integration
5. **Code Review Package**: Clean, documented codebase ready for review

### Feedback Request
We seek core reviewer feedback on:
- Architecture and design approach alignment with PyTorch standards
- Integration points with existing documentation infrastructure
- Technical implementation preferences and constraints
- Timeline and resource allocation recommendations
- Testing and validation strategies

---

**Commitment**: All development work will remain exclusively on the `docs/gen-example-autotest` branch until core reviewer approval and integration planning is complete. Regular progress updates will be provided through GitHub issues and this documentation will be maintained as the single source of truth for the project status.

## References

- [Issue #6662](https://github.com/pytorch/pytorch/issues/6662): Original feature request
- [Sphinx Documentation](https://www.sphinx-doc.org/): Primary documentation framework
- [nbsphinx Documentation](https://nbsphinx.readthedocs.io/): Jupyter integration
- [PyTorch Documentation Guidelines](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md): Contributing standards
