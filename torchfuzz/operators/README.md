# Operators

This directory contains the operator implementations for a modular, chainable data processing system. Operators are composable components that can be connected together to form complex data processing pipelines.

## Overview of Operator Categories

### Base Infrastructure
- **Base Operator**: Core abstract base class that all operators inherit from
- **Operator Utils**: Shared utility functions and helper classes for operator implementations
- **Message Utils**: Utilities for handling message format conversions and data transformations

### Model Operators
These operators interface with various AI models and inference systems:

- **LLM Operators**: Large Language Model inference operators (Llama, GPT, etc.)
- **Embedding Operators**: Text and multimodal embedding generation
- **Classification Operators**: Model-based classification and scoring
- **Generation Operators**: Content generation using various model backends
- **Reward Model Operators**: Reward model inference for RLHF workflows

### Data Processing Operators
These operators handle data transformation and preparation:

- **Tokenizer Operators**: Tokenization using various tokenizer backends
- **Text Processing Operators**: Text cleaning, normalization, and transformation
- **Data Conversion Operators**: Format conversion between different data structures
- **Filtering Operators**: Data filtering based on various criteria
- **Aggregation Operators**: Data aggregation and statistical operations

### Template and Prompt Operators
These operators handle prompt engineering and templating:

- **Template Operators**: String template formatting with variable substitution
- **Prompt Engineering Operators**: Advanced prompt construction and optimization
- **Context Injection Operators**: Dynamic context and instruction insertion
- **Multi-turn Dialog Operators**: Conversation history management

### I/O Operators
These operators handle data input and output operations:

- **File Operators**: Reading from and writing to various file formats
- **Database Operators**: Database query and update operations
- **API Operators**: External service integration and API calls
- **Cache Operators**: Caching and memoization functionality

## Architecture Principles

### Operator Design Pattern
All operators follow a consistent design pattern:

1. **Inheritance**: All operators inherit from `BaseOperator`
2. **Chaining**: Operators can be chained using the `|` (pipe) operator
3. **DataFrame Processing**: Operators process pandas DataFrames as input/output
4. **Stateless Execution**: Operators should be stateless and reusable
5. **Dependency Management**: Operators track their dependencies through the pipeline

### Execution Model
- **Topological Sorting**: The system automatically sorts operators for correct execution order
- **Lazy Evaluation**: Operators are not executed until explicitly invoked
- **Result Caching**: Results are cached during a single execution pipeline
- **Parallel Processing**: Independent operators can be executed in parallel
- **Error Propagation**: Errors are properly propagated through the pipeline

### Data Flow Architecture
```
Input Data → Operator 1 → Operator 2 → ... → Operator N → Output Data
             ↓           ↓                   ↓
           Transform   Validate            Cache
```

## How to Add New Operators

### 1. Create the Operator Class

Create a new file in the appropriate category subdirectory following the naming convention `{operator_name}_operator.py`:

```python
import pandas as pd
from operators.base_operator import BaseOperator
from typing import Any, Dict, Optional


class YourCustomOperator(BaseOperator):
    """
    Brief description of what your operator does.
    
    This operator performs [specific functionality] on the input DataFrame
    and returns the transformed result.
    """

    def __init__(
        self,
        # Add your configuration parameters here
        param1: str,
        param2: int = 42,
        output_column: str = "result",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.param1 = param1
        self.param2 = param2
        self.output_column = output_column

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the operator logic on the input DataFrame.
        
        Args:
            df: Input DataFrame from previous operators in the pipeline
            
        Returns:
            Modified DataFrame with your operator's results added
            
        Raises:
            ValueError: If input DataFrame doesn't meet requirements
            RuntimeError: If processing fails
        """
        # Validate input
        if df.empty:
            raise ValueError("Input DataFrame cannot be empty")
            
        # Your implementation here
        result_data = []
        for _, row in df.iterrows():
            # Process each row
            processed_value = self._process_row(row)
            result_data.append(processed_value)
        
        # Add results to DataFrame
        df[self.output_column] = result_data
        return df
    
    def _process_row(self, row: pd.Series) -> Any:
        """Helper method for processing individual rows."""
        # Implement your row processing logic
        return f"{self.param1}_{row.get('input_field', '')}_{self.param2}"

    def validate_config(self) -> None:
        """Validate operator configuration."""
        super().validate_config()
        if not self.param1:
            raise ValueError("param1 cannot be empty")
```

### 2. Add to Module Exports

Add your operator to the appropriate `__init__.py` file:

```python
from .your_custom_operator import YourCustomOperator

__all__ = [
    # ... existing operators
    "YourCustomOperator",
]
```

### 3. Write Comprehensive Tests

Create a test file in the `tests/` directory:

```python
import pandas as pd
import pytest
from operators.your_category.your_custom_operator import YourCustomOperator


class TestYourCustomOperator:
    """Test suite for YourCustomOperator."""

    def test_basic_functionality(self):
        """Test basic operator functionality."""
        operator = YourCustomOperator(param1="test", param2=100)
        test_data = pd.DataFrame([
            {"input_field": "hello"},
            {"input_field": "world"}
        ])
        
        result = operator.execute(test_data.copy())
        
        assert len(result) == 2
        assert "result" in result.columns
        assert result["result"].iloc[0] == "test_hello_100"
        assert result["result"].iloc[1] == "test_world_100"

    def test_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises appropriate error."""
        operator = YourCustomOperator(param1="test")
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Input DataFrame cannot be empty"):
            operator.execute(empty_df)

    def test_chaining_with_other_operators(self):
        """Test operator chaining functionality."""
        op1 = YourCustomOperator(param1="first", output_column="step1")
        op2 = YourCustomOperator(param1="second", output_column="step2")
        
        pipeline = op1 | op2
        
        test_data = pd.DataFrame([{"input_field": "test"}])
        result = pipeline.invoke(test_data)
        
        assert "step1" in result.columns
        assert "step2" in result.columns

    def test_custom_output_column(self):
        """Test custom output column naming."""
        operator = YourCustomOperator(param1="test", output_column="custom_output")
        test_data = pd.DataFrame([{"input_field": "data"}])
        
        result = operator.execute(test_data.copy())
        
        assert "custom_output" in result.columns
        assert "result" not in result.columns
```

### 4. Consider Operator Categories

When designing your operator, consider which category it fits into:

- **Model Operators**: Interface with AI models or inference systems
- **Data Processing**: Transform, filter, or manipulate data
- **Template/Prompt**: Handle prompt engineering or text templating
- **I/O Operations**: Handle data input/output operations
- **Utility**: Provide supporting functionality for other operators

### 5. Performance Considerations

- Use vectorized pandas operations when possible
- Consider memory usage for large DataFrames
- Implement lazy evaluation where appropriate
- Add progress tracking for long-running operations

## Usage Examples

### Basic Operator Usage

```python
from operators.template.simple_template_operator import SimpleTemplateOperator
import pandas as pd

# Create an operator
template_op = SimpleTemplateOperator(template="Hello, {name}!")

# Prepare data
df = pd.DataFrame([{"name": "Alice"}, {"name": "Bob"}])

# Execute the operator
result = template_op.execute(df.copy())
print(result["output"].tolist())  # ["Hello, Alice!", "Hello, Bob!"]
```

### Chaining Operators

```python
from operators.template.simple_template_operator import SimpleTemplateOperator
from operators.processing.text_processor_operator import TextProcessorOperator

# Chain operators using the pipe operator
pipeline = (
    SimpleTemplateOperator(template="User: {message}")
    | TextProcessorOperator(operation="uppercase")
)

# Execute the chain
df = pd.DataFrame([{"message": "Hello world"}])
result = pipeline.invoke(df)
print(result["processed_text"].iloc[0])  # "USER: HELLO WORLD"
```

### Complex Pipeline Example

```python
from operators.io.file_reader_operator import FileReaderOperator
from operators.processing.data_cleaner_operator import DataCleanerOperator
from operators.model.classification_operator import ClassificationOperator
from operators.io.file_writer_operator import FileWriterOperator

# Create a complex processing pipeline
pipeline = (
    FileReaderOperator(file_path="input.csv")
    | DataCleanerOperator(remove_nulls=True, normalize_text=True)
    | ClassificationOperator(model="sentiment-analyzer")
    | FileWriterOperator(output_path="results.json")
)

# Process data through the pipeline
result = pipeline.invoke(pd.DataFrame())  # Empty DF for file input
```

### Conditional Pipeline Execution

```python
from operators.base.conditional_operator import ConditionalOperator
from operators.processing.filter_operator import FilterOperator

# Create conditional processing
pipeline = (
    ConditionalOperator(
        condition=lambda df: len(df) > 100,
        true_operator=FilterOperator(column="score", threshold=0.8),
        false_operator=FilterOperator(column="score", threshold=0.5)
    )
)
```

### Parallel Processing Example

```python
from operators.base.parallel_operator import ParallelOperator
from operators.model.embedding_operator import EmbeddingOperator
from operators.model.classification_operator import ClassificationOperator

# Create parallel processing pipeline
parallel_op = ParallelOperator([
    EmbeddingOperator(model="sentence-transformer"),
    ClassificationOperator(model="topic-classifier"),
])

# Both operators will process the same input concurrently
result = parallel_op.invoke(input_df)
```

## How to Run Tests

### Running All Tests

```bash
# Run all operator tests
pytest operators/tests/ -v

# Run with coverage
pytest operators/tests/ --cov=operators --cov-report=html
```

### Running Category-Specific Tests

```bash
# Run tests for a specific category
pytest operators/tests/model/ -v
pytest operators/tests/processing/ -v
pytest operators/tests/template/ -v
```

### Running Specific Test Files

```bash
# Run a specific test file
pytest operators/tests/model/test_llm_operator.py -v

# Run a specific test method
pytest operators/tests/model/test_llm_operator.py::TestLLMOperator::test_basic_inference -v
```

### Running Tests During Development

```bash
# Watch mode for continuous testing
pytest operators/tests/ --watch

# Run only failed tests
pytest operators/tests/ --lf

# Run tests with specific markers
pytest operators/tests/ -m "slow" -v
pytest operators/tests/ -m "not slow" -v
```

### Performance Testing

```bash
# Run performance benchmarks
pytest operators/tests/benchmarks/ --benchmark-only

# Profile memory usage
pytest operators/tests/ --memray
```

### Test Configuration

Create a `pytest.ini` file in the operators directory:

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
addopts = 
    --strict-markers
    --disable-warnings
    --tb=short
```

## Code Organization Principles

### Directory Structure
```
operators/
├── __init__.py                # Main module exports
├── README.md                  # This documentation
├── base/                      # Base classes and core functionality
│   ├── __init__.py
│   ├── base_operator.py      # Abstract base operator class
│   ├── operator_registry.py  # Operator registration system
│   └── exceptions.py         # Custom exception classes
├── model/                     # AI/ML model operators
│   ├── __init__.py
│   ├── llm_operator.py       # Large language models
│   ├── embedding_operator.py # Embedding generation
│   └── classification_operator.py
├── processing/                # Data processing operators
│   ├── __init__.py
│   ├── text_processor_operator.py
│   ├── data_cleaner_operator.py
│   └── aggregation_operator.py
├── template/                  # Template and prompt operators
│   ├── __init__.py
│   ├── simple_template_operator.py
│   └── prompt_engineering_operator.py
├── io/                        # Input/output operators
│   ├── __init__.py
│   ├── file_reader_operator.py
│   ├── file_writer_operator.py
│   └── api_client_operator.py
├── utils/                     # Utility functions and helpers
│   ├── __init__.py
│   ├── data_validation.py
│   ├── performance_utils.py
│   └── logging_utils.py
├── tests/                     # Test files
│   ├── __init__.py
│   ├── conftest.py           # Pytest configuration and fixtures
│   ├── base/
│   ├── model/
│   ├── processing/
│   ├── template/
│   ├── io/
│   └── benchmarks/           # Performance benchmarks
└── docs/                     # Additional documentation
    ├── api_reference.md
    ├── performance_guide.md
    └── examples/
```

### Naming Conventions
- **Operator Files**: `{descriptive_name}_operator.py`
- **Operator Classes**: `{DescriptiveName}Operator` (PascalCase)
- **Test Files**: `test_{operator_name}.py`
- **Test Classes**: `Test{OperatorName}`
- **Utility Modules**: `{purpose}_utils.py`

### Code Quality Standards

#### Documentation Standards
```python
class ExampleOperator(BaseOperator):
    """
    One-line summary of what the operator does.
    
    Longer description explaining the operator's purpose, use cases,
    and any important implementation details.
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2 (default: value)
        
    Attributes:
        attribute1: Description of important attributes
        
    Example:
        >>> op = ExampleOperator(param1="value")
        >>> result = op.execute(input_df)
        >>> print(result.head())
    """
```

#### Type Annotations
```python
from typing import Any, Dict, List, Optional, Union
import pandas as pd

def process_data(
    data: pd.DataFrame,
    config: Dict[str, Any],
    options: Optional[List[str]] = None
) -> Union[pd.DataFrame, None]:
    """Process data with type hints for all parameters."""
    pass
```

#### Error Handling Best Practices
```python
class CustomOperator(BaseOperator):
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Validate inputs
            self._validate_input(df)
            
            # Process data
            result = self._process_data(df)
            
            # Validate outputs
            self._validate_output(result)
            
            return result
            
        except ValidationError as e:
            raise OperatorValidationError(f"Input validation failed: {e}")
        except ProcessingError as e:
            raise OperatorExecutionError(f"Processing failed: {e}")
        except Exception as e:
            raise OperatorError(f"Unexpected error in {self.__class__.__name__}: {e}")
```

### Performance Guidelines

#### Memory Management
```python
# Good: Process data in chunks for large datasets
def execute(self, df: pd.DataFrame) -> pd.DataFrame:
    if len(df) > self.chunk_size:
        return self._process_in_chunks(df)
    return self._process_full(df)

# Good: Use vectorized operations
df['result'] = df['column'].apply(self._vectorized_function)

# Avoid: Row-by-row processing for large datasets
# for idx, row in df.iterrows():  # Slow for large DataFrames
```

#### Caching Strategy
```python
from functools import lru_cache
import hashlib

class CachedOperator(BaseOperator):
    @lru_cache(maxsize=128)
    def _expensive_computation(self, key: str) -> Any:
        """Cache expensive computations."""
        return self._compute(key)
    
    def _get_cache_key(self, data: pd.DataFrame) -> str:
        """Generate cache key from DataFrame content."""
        return hashlib.md5(data.to_string().encode()).hexdigest()
```

### Dependency Management

#### External Dependencies
- Minimize external dependencies
- Pin dependency versions in requirements files
- Use optional dependencies for non-core functionality
- Document all dependencies clearly

#### Internal Dependencies
```python
# Good: Clear internal imports
from operators.base.base_operator import BaseOperator
from operators.utils.data_validation import validate_dataframe

# Good: Lazy imports for optional features
def _get_model(self):
    try:
        import transformers
        return transformers.AutoModel.from_pretrained(self.model_name)
    except ImportError:
        raise ImportError("transformers package required for this operator")
```

### Configuration Management

#### Operator Configuration
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class OperatorConfig:
    """Configuration class for operators."""
    batch_size: int = 32
    max_retries: int = 3
    timeout: Optional[float] = None
    cache_enabled: bool = True
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")

class ConfigurableOperator(BaseOperator):
    def __init__(self, config: OperatorConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.config.validate()
```

## Contributing Guidelines

### Development Workflow
1. **Create Feature Branch**: Create a new branch for your operator
2. **Implement Operator**: Follow the patterns described above
3. **Write Tests**: Ensure >90% test coverage
4. **Update Documentation**: Update this README and add docstrings
5. **Performance Testing**: Benchmark performance-critical operators
6. **Code Review**: Submit for peer review
7. **Integration Testing**: Test with existing operators

### Code Review Checklist
- [ ] Operator follows the base class pattern
- [ ] Comprehensive test coverage (>90%)
- [ ] Clear documentation and docstrings
- [ ] Type hints for all methods
- [ ] Proper error handling and validation
- [ ] Performance considerations addressed
- [ ] Integration with existing operators tested
- [ ] No breaking changes to existing APIs

### Versioning and Compatibility
- Follow semantic versioning (MAJOR.MINOR.PATCH)
- Maintain backward compatibility when possible
- Deprecate old APIs before removing them
- Document breaking changes in changelog

### Performance Benchmarking
```python
import time
import pandas as pd
from memory_profiler import profile

@profile
def benchmark_operator():
    """Benchmark operator performance."""
    operator = YourOperator()
    test_data = pd.DataFrame({'col': range(10000)})
    
    start_time = time.time()
    result = operator.execute(test_data)
    end_time = time.time()
    
    print(f"Processing time: {end_time - start_time:.2f}s")
    print(f"Rows per second: {len(test_data) / (end_time - start_time):.0f}")
    return result
```

## Troubleshooting

### Common Issues and Solutions

#### Import Errors
```python
# Problem: Module not found
ModuleNotFoundError: No module named 'operators.model.llm_operator'

# Solution: Check __init__.py exports and Python path
# Ensure operators directory is in PYTHONPATH
import sys
sys.path.append('/path/to/operators')
```

#### Memory Issues
```python
# Problem: Out of memory with large DataFrames
# Solution: Process in chunks
def _process_in_chunks(self, df: pd.DataFrame) -> pd.DataFrame:
    chunk_size = 1000
    results = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size]
        result_chunk = self._process_chunk(chunk)
        results.append(result_chunk)
    return pd.concat(results, ignore_index=True)
```

#### Performance Issues
```python
# Problem: Slow operator execution
# Solution: Profile and optimize
import cProfile

def profile_operator():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Your operator execution
    result = operator.execute(test_data)
    
    profiler.disable()
    profiler.print_stats(sort='cumulative')
```

#### Chaining Issues
```python
# Problem: Operators not chaining correctly
# Solution: Check column name compatibility
class CompatibleOperator(BaseOperator):
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        # Ensure required columns exist
        required_cols = ['input_column']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Process and return
        return df
```

### Debugging Tips

1. **Enable Debug Logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class DebuggableOperator(BaseOperator):
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.debug(f"Input DataFrame shape: {df.shape}")
        logger.debug(f"Input columns: {df.columns.tolist()}")
        
        result = self._process(df)
        
        logger.debug(f"Output DataFrame shape: {result.shape}")
        return result
```

2. **Data Validation**:
```python
def _validate_input(self, df: pd.DataFrame) -> None:
    """Validate input DataFrame structure."""
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
    required_columns = getattr(self, 'required_columns', [])
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
```

3. **Use DataFrame.info() for debugging**:
```python
def debug_dataframe(self, df: pd.DataFrame, stage: str = "") -> None:
    """Print DataFrame information for debugging."""
    print(f"\n=== DataFrame Info {stage} ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum()} bytes")
    print(df.head())
    print("=" * 40)
```

### Getting Help

- **Documentation**: Check this README and inline docstrings
- **Examples**: Review the `examples/` directory for usage patterns
- **Tests**: Look at test files for working examples
- **Issues**: Check the issue tracker for known problems
- **Community**: Join the developer discussion channels

### Best Practices Summary

1. **Design**: Keep operators focused on single responsibilities
2. **Testing**: Write comprehensive tests with good coverage
3. **Performance**: Consider memory usage and processing time
4. **Documentation**: Write clear docstrings and examples
5. **Compatibility**: Maintain backward compatibility when possible
6. **Error Handling**: Provide helpful error messages
7. **Logging**: Add appropriate logging for debugging
8. **Validation**: Validate inputs and outputs thoroughly