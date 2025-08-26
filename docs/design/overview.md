# Auto-Running Code Examples in PyTorch Documentation - Design Overview

## Background & Motivation

Automatically generated and validated code outputs in ML documentation provide significant benefits:

- **Accuracy**: Ensures documentation examples produce expected outputs, eliminating discrepancies between code and results
- **Freshness**: Automatically updates outputs when API changes, keeping documentation current
- **Trust**: Users gain confidence in examples that demonstrably work
- **Maintenance**: Reduces manual effort in keeping examples synchronized with codebase changes
- **Quality**: Catches breaking changes early in the development cycle

## Current PyTorch Documentation Infrastructure

### Sphinx Autodoc
PyTorch uses Sphinx autodoc to automatically extract docstrings from Python modules. Key features:
- Automatic API documentation generation from docstrings
- Cross-referencing capabilities between modules
- Support for various docstring formats (Google, NumPy, Sphinx)
- Integration with type hints and signatures

### Doctest Integration
Sphinx doctest extension provides:
- Execution of code examples in docstrings
- Output verification against expected results
- Integration with pytest for testing workflows
- Support for setup/teardown code blocks

### nbsphinx for Jupyter Integration
nbsphinx enables:
- Direct inclusion of Jupyter notebooks in documentation
- Automatic execution of notebooks during build
- Cell output preservation and display
- Cross-referencing between notebooks and standard docs

## Gaps and Technical Challenges

### Current Limitations
1. **Inconsistent Coverage**: Not all code examples are automatically validated
2. **Manual Maintenance**: Many examples require manual output updates
3. **Environment Dependencies**: Complex setup requirements for GPU/distributed examples
4. **Build Performance**: Execution overhead can significantly slow documentation builds
5. **Flaky Examples**: Non-deterministic outputs or timing-dependent code

### Technical Challenges
1. **Resource Management**: GPU memory, computation time, and parallel execution
2. **Version Compatibility**: Ensuring examples work across supported PyTorch versions
3. **Error Handling**: Graceful handling of failed examples without breaking builds
4. **Output Formatting**: Consistent presentation of execution results
5. **Caching Strategy**: Avoiding redundant execution while ensuring freshness

## Proposed Minimal Prototype

### Core Components

#### 1. Code Block Extractor
```python
# extract_examples.py
import ast
import re
from pathlib import Path
from typing import List, Dict, Tuple

class DocumentationExtractor:
    def extract_code_blocks(self, file_path: str) -> List[Dict]:
        """Extract executable code blocks from documentation files"""
        pass
    
    def extract_docstring_examples(self, module_path: str) -> List[Dict]:
        """Extract examples from Python docstrings"""
        pass
```

#### 2. Example Runner
```python
# run_examples.py
import subprocess
import tempfile
from contextlib import contextmanager
from typing import Optional, Dict, Any

class ExampleRunner:
    def __init__(self, timeout: int = 30, max_memory: str = "1GB"):
        self.timeout = timeout
        self.max_memory = max_memory
    
    def execute_code(self, code: str, context: Dict[str, Any]) -> Dict:
        """Execute code block and capture output"""
        pass
    
    @contextmanager
    def isolated_environment(self):
        """Create isolated execution environment"""
        pass
```

#### 3. Output Verifier
```python
# verify_outputs.py
import difflib
from typing import List, Optional

class OutputVerifier:
    def compare_outputs(self, expected: str, actual: str) -> bool:
        """Compare expected vs actual outputs with tolerance"""
        pass
    
    def generate_diff_report(self, differences: List[Dict]) -> str:
        """Generate human-readable diff report"""
        pass
```

### Integration Options

#### Option A: Sphinx Extension
```python
# sphinx_autorun.py
from sphinx.ext import doctest
from sphinx.util.docutils import docutils_namespace

class AutoRunExamplesExtension:
    def setup(app):
        app.add_config_value('autorun_examples', True, 'html')
        app.connect('doctree-resolved', process_code_blocks)
        return {'version': '1.0', 'parallel_read_safe': True}
```

#### Option B: CI Job Integration
```yaml
# .github/workflows/docs-examples.yml
name: Validate Documentation Examples
on:
  pull_request:
    paths: ['docs/**', '**/*.py']

jobs:
  validate-examples:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r docs/requirements.txt
      - name: Extract and run examples
        run: python scripts/validate_doc_examples.py
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: example-validation-results
          path: validation_results.json
```

#### Option C: Custom nbval-style Runner
```python
# custom_runner.py
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

class DocumentationRunner:
    def __init__(self):
        self.executor = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    def run_markdown_examples(self, file_path: str) -> Dict:
        """Convert markdown examples to notebook and execute"""
        pass
    
    def validate_docstring_examples(self, module_path: str) -> Dict:
        """Run doctest-style validation on docstrings"""
        pass
```

### Implementation Plan

#### Phase 1: Prototype (2 weeks)
1. Create basic extractor for common code block patterns
2. Implement simple runner with timeout and error handling
3. Test on 10-15 representative examples from PyTorch docs
4. Measure performance impact and resource usage

#### Phase 2: Integration (4 weeks)
1. Choose optimal integration approach based on prototype results
2. Add caching and incremental execution capabilities
3. Implement comprehensive error handling and reporting
4. Create configuration system for example metadata

#### Phase 3: Deployment (2 weeks)
1. Integrate with existing CI/CD pipeline
2. Add monitoring and alerting for failed examples
3. Create contributor documentation and guidelines
4. Gradual rollout with feedback collection

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

## Community Review and Feedback Request

We seek community input on this design proposal:

### Technical Feedback Needed
- Architecture and implementation approach preferences
- Integration points with existing PyTorch documentation infrastructure
- Performance requirements and resource constraints
- Error handling and fallback strategies

### Process Feedback Needed
- Contributor workflow implications
- Testing and validation strategies
- Rollout and adoption timeline
- Maintenance and support model

### How to Provide Feedback
1. **GitHub Issues**: Comment on [Issue #6662](https://github.com/pytorch/pytorch/issues/6662)
2. **Pull Request Reviews**: Review implementation PRs when available
3. **Community Forums**: Discuss on PyTorch developer forums
4. **Direct Contact**: Reach out to the documentation team

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

**Commitment**: All development work will remain exclusively on the `docs/gen-example-autotest` branch until core reviewer approval and integration planning is complete. Regular progress updates will be provided through GitHub issues and this documentation will be maintained as the single source of truth for the project status.

## References

- [Issue #6662](https://github.com/pytorch/pytorch/issues/6662): Original feature request
- [Sphinx Documentation](https://www.sphinx-doc.org/): Primary documentation framework
- [Sphinx Doctest Extension](https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html): Code testing capabilities
- [nbsphinx Documentation](https://nbsphinx.readthedocs.io/): Jupyter integration
- [nbval](https://github.com/computationalmodelling/nbval): Notebook validation tool
- [PyTorch Documentation Guidelines](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md): Contributing standards
