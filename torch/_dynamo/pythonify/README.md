# Pythonify Module Developer Documentation

This document explains the architecture of the `pythonify` module and provides guidance
for developers who want to extend the IR with new node types or code generation visitors.

## Overview

The `pythonify` module provides infrastructure for generating explicit Python code from
`torch.compile`'s runtime machinery. When users call `torch.compile(model, pythonify="/path/to/output.py")`,
instead of producing an opaque compiled function, this module generates readable Python
code that represents all the runtime operations.

## Architecture

```
torch.compile(model, pythonify="/path/to/output.py")
         │
         ▼
┌─────────────────────────────────────┐
│         Dynamo Tracing             │
│   (FX Graph + Guards + Metadata)   │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Adapters (guards.py, inputs.py,  │
│   aot_autograd.py, cuda_graphs.py) │
│   Translate artifacts to IR nodes  │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│      RuntimeWrapperPipeline        │
│   (Structured IR Representation)   │
└─────────────────────────────────────┘
         │
         ├──────────────────┐
         ▼                  ▼
┌─────────────────┐  ┌─────────────────┐
│   gen_binary    │  │   gen_python    │
│ (Compiled Path) │  │ (Source Code)   │
└─────────────────┘  └─────────────────┘
         │                  │
         ▼                  ▼
   CompiledWrapper     Python File
   (in-memory)         (/tmp/model.py)
```

## Module Structure

```
torch/_dynamo/pythonify/
├── __init__.py              # Public API exports
├── README.md                # This documentation
├── ir.py                    # IR node definitions (IRNode, CodeGenVisitor)
├── pipeline.py              # RuntimeWrapperPipeline orchestration
├── gen_binary.py            # Binary code generation backend
├── gen_python.py            # Python code generation backend
├── kernel_serializer.py     # Kernel serialization utilities
├── context.py               # Pythonify context management
├── errors.py                # Error handling utilities
├── warnings.py              # Warning utilities for unsupported features
└── adapters/
    ├── __init__.py          # Adapter exports
    ├── guards.py            # GuardAdapter: Dynamo guards → GuardCheckNodes
    ├── aot_autograd.py      # AOTAutogradAdapter: AOT results → IR nodes
    ├── cuda_graphs.py       # CUDAGraphAdapter: CUDA config → IR nodes
    └── inputs.py            # InputAdapter: Input sources → ArgumentExtractionNodes
```

## Core Concepts

### IR Nodes

IR (Intermediate Representation) nodes are the building blocks of the pythonify system.
Each node represents a specific piece of runtime machinery:

| Node Type | Purpose |
|-----------|---------|
| `ArgumentExtractionNode` | Extract values from model attributes or frame locals |
| `GuardCheckNode` | Runtime assertion to validate input properties |
| `AOTAutogradWrapperNode` | Autograd function wrapping forward/backward graphs |
| `CUDAGraphSetupNode` | CUDA graph capture and replay configuration |
| `KernelLoadNode` | Load a serialized compiled kernel |
| `CallableInvocationNode` | Invoke the compiled callable |
| `ReturnResultNode` | Expose the result for `exec()` compatibility |

### Visitor Pattern

The IR uses the visitor pattern to support multiple code generation backends without
modifying the node classes. Each node has an `accept()` method that calls the
appropriate `visit_*` method on a `CodeGenVisitor`.

### Code Generation Backends

- **gen_binary** (`BinaryCodeGenVisitor`): Produces a `CompiledWrapper` object that
  executes the compiled function at runtime. This preserves existing `torch.compile` behavior.

- **gen_python** (`PythonCodeGenVisitor`): Emits Python source code that can be
  written to a file and executed via `exec()`.

## How to Add a New IR Node Type

Follow these steps to add a new IR node type:

### Step 1: Define the Node Class in `ir.py`

Create a new dataclass that inherits from `IRNode`:

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class MyNewNode(IRNode):
    """
    IR node for [describe what this node represents].

    This node [explain when and why this node is used].

    Attributes:
        my_attribute: [Description of attribute]
        another_attr: [Description]
    """

    my_attribute: str
    another_attr: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def accept(self, visitor: "CodeGenVisitor") -> Any:
        return visitor.visit_my_new_node(self)
```

### Step 2: Add the Visit Method to `CodeGenVisitor`

Add an abstract method to the `CodeGenVisitor` base class in `ir.py`:

```python
class CodeGenVisitor(abc.ABC):
    # ... existing methods ...

    @abc.abstractmethod
    def visit_my_new_node(self, node: MyNewNode) -> Any:
        """Process a MyNewNode."""
        pass
```

### Step 3: Implement in BinaryCodeGenVisitor (`gen_binary.py`)

Add the implementation for the binary backend:

```python
class BinaryCodeGenVisitor(CodeGenVisitor):
    # ... existing methods ...

    def visit_my_new_node(self, node: MyNewNode) -> Any:
        """
        Process MyNewNode for binary code generation.

        [Explain what this does for the binary backend]
        """
        # Implementation that updates self._wrapper or returns metadata
        return {
            "my_attribute": node.my_attribute,
            "another_attr": node.another_attr,
        }
```

### Step 4: Implement in PythonCodeGenVisitor (`gen_python.py`)

Add the implementation for the Python code generation backend:

```python
class PythonCodeGenVisitor(CodeGenVisitor):
    # ... existing methods ...

    def visit_my_new_node(self, node: MyNewNode) -> str:
        """
        Generate Python code for MyNewNode.

        Produces code like:
            # My new operation
            result = do_something(node.my_attribute)
        """
        # Emit section header if needed
        if not self._has_emitted_my_section:
            self._emitter.emit_section_header("My New Section")
            self._has_emitted_my_section = True

        # Generate the Python code
        line = f"result = do_something({format_value(node.my_attribute)})"
        self._emitter.emit_line(line)
        return line
```

### Step 5: Update the Pipeline (`pipeline.py`)

Add logic to `RuntimeWrapperPipeline` to create your new node:

```python
class RuntimeWrapperPipeline:
    def build(self) -> RuntimeWrapperIR:
        # ... existing code ...

        # Add your new node at the appropriate point
        self._build_my_new_node()

        # ... rest of build ...

    def _build_my_new_node(self) -> None:
        """Build MyNewNode if applicable."""
        assert self._ir is not None

        # Only add if some condition is met
        if self.artifacts.some_condition:
            node = MyNewNode(
                my_attribute=self.artifacts.some_value,
                another_attr=42,
            )
            self._ir.add_node(node)
```

### Step 6: Export from `__init__.py`

Add your new node to the exports:

```python
from .ir import (
    # ... existing exports ...
    MyNewNode,
)

__all__ = [
    # ... existing exports ...
    "MyNewNode",
]
```

### Step 7: Write Tests

Create tests in `test/dynamo/test_pythonify.py`:

```python
class TestMyNewNode(TestCase):
    def test_my_new_node_creation(self):
        """Test basic MyNewNode creation."""
        node = MyNewNode(
            my_attribute="test",
            another_attr=42,
        )
        self.assertEqual(node.my_attribute, "test")
        self.assertEqual(node.another_attr, 42)

    def test_my_new_node_visitor_pattern(self):
        """Test MyNewNode works with visitor pattern."""
        node = MyNewNode(my_attribute="test")
        visitor = PythonCodeGenVisitor()
        result = node.accept(visitor)
        self.assertIsInstance(result, str)

    def test_my_new_node_in_pipeline(self):
        """Test MyNewNode is created by pipeline when appropriate."""
        artifacts = CompilationArtifacts(
            some_condition=True,
            some_value="test",
            # ... other required fields ...
        )
        pipeline = RuntimeWrapperPipeline(artifacts)
        ir = pipeline.build()
        my_nodes = ir.get_nodes_by_type(MyNewNode)
        self.assertEqual(len(my_nodes), 1)
```

## How to Add a New Adapter

Adapters translate torch.compile internal data structures to IR nodes.

### Step 1: Create Adapter Module in `adapters/`

Create a new file like `adapters/my_adapter.py`:

```python
"""
Adapter for translating [source] to IR nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

from ..ir import MyNewNode


class MySourceType(Enum):
    """Enum for different source types."""
    TYPE_A = auto()
    TYPE_B = auto()


@dataclass
class MyInfo:
    """
    Intermediate representation for [source].

    Captures the essential information needed to create IR nodes.
    """
    source_type: MySourceType
    name: str
    value: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MyAdapter:
    """
    Adapter that translates [source] to IR nodes.

    Usage:
        adapter = MyAdapter()
        nodes = adapter.translate(my_source_data)
    """

    def translate(self, source_data: Any) -> list[MyNewNode]:
        """
        Translate source data to IR nodes.

        Args:
            source_data: The source data to translate

        Returns:
            List of MyNewNode objects
        """
        nodes = []
        # Translation logic here
        return nodes

    def translate_info(self, info: MyInfo) -> MyNewNode:
        """
        Translate MyInfo to a single IR node.

        Args:
            info: The intermediate representation

        Returns:
            A MyNewNode
        """
        return MyNewNode(
            my_attribute=info.name,
            another_attr=info.value,
            metadata=info.metadata.copy(),
        )


def translate_my_source(source_data: Any) -> list[MyNewNode]:
    """
    Convenience function for translating source data.

    Args:
        source_data: The source data to translate

    Returns:
        List of MyNewNode objects
    """
    adapter = MyAdapter()
    return adapter.translate(source_data)
```

### Step 2: Export from `adapters/__init__.py`

```python
from .my_adapter import (
    MyAdapter,
    MyInfo,
    MySourceType,
    translate_my_source,
)

__all__ = [
    # ... existing exports ...
    "MyAdapter",
    "MyInfo",
    "MySourceType",
    "translate_my_source",
]
```

### Step 3: Export from Main `__init__.py`

```python
from .adapters import (
    # ... existing exports ...
    MyAdapter,
    MyInfo,
    MySourceType,
    translate_my_source,
)

__all__ = [
    # ... existing exports ...
    "MyAdapter",
    "MyInfo",
    "MySourceType",
    "translate_my_source",
]
```

## Existing Adapters Reference

### GuardAdapter (`adapters/guards.py`)

Translates Dynamo guard conditions to `GuardCheckNode` objects.

Key classes:
- `DynamoGuardType`: Enum of all Dynamo guard types
- `GuardInfo`: Intermediate representation for guards
- `TensorGuardInfo`: Specialized representation for tensor guards
- `GuardAdapter`: Main adapter class

Example usage:
```python
adapter = GuardAdapter()
nodes = adapter.translate_guard_dict(guard_dict)
```

### AOTAutogradAdapter (`adapters/aot_autograd.py`)

Translates AOT Autograd compilation results to `AOTAutogradWrapperNode`.

Key classes:
- `AOTCompilationMode`: Enum for compilation modes (INFERENCE, TRAINING)
- `AOTAutogradInfo`: Comprehensive intermediate representation
- `AOTAutogradAdapter`: Main adapter class

### CUDAGraphAdapter (`adapters/cuda_graphs.py`)

Translates CUDA graph configuration to `CUDAGraphSetupNode`.

Key classes:
- `CUDACaptureMode`: Enum for capture modes (THREAD_LOCAL, RELAXED, GLOBAL)
- `CUDAGraphInfo`: Intermediate representation
- `CUDAGraphAdapter`: Main adapter class

### InputAdapter (`adapters/inputs.py`)

Translates input sources to `ArgumentExtractionNode` objects.

Key classes:
- `InputSourceType`: Enum for source types (LOCAL, GLOBAL, ATTRIBUTE, etc.)
- `InputInfo`: Intermediate representation for inputs
- `InputAdapter`: Main adapter class

## Error Handling

The `errors.py` module provides structured error handling:

```python
from .errors import (
    PythonifyError,
    PythonifyStage,
    create_pythonify_error,
)

try:
    # Some operation
    ...
except Exception as e:
    raise create_pythonify_error(
        PythonifyStage.IR_CONSTRUCTION,
        e,
        context={"step": "my new node"},
    ) from e
```

Available stages:
- `INITIALIZATION`
- `ARGUMENT_EXTRACTION`
- `GUARD_TRANSLATION`
- `AOT_AUTOGRAD`
- `CUDA_GRAPH_SETUP`
- `IR_CONSTRUCTION`
- `PYTHON_CODE_GENERATION`
- `BINARY_CODE_GENERATION`
- `FILE_WRITING`
- `KERNEL_SERIALIZATION`
- `ARTIFACT_COLLECTION`

## Testing Guidelines

1. **Unit tests for IR nodes**: Test creation, field defaults, and visitor pattern
2. **Unit tests for adapters**: Test translation logic with various inputs
3. **Integration tests**: Test full pipeline with artifacts
4. **Code generation tests**: Verify generated code is syntactically valid
5. **Execution tests**: Verify generated code executes correctly

Run tests with:
```bash
python test/dynamo/test_pythonify.py -v
```

Run specific test class:
```bash
python test/dynamo/test_pythonify.py TestMyNewNode -v
```

## Best Practices

1. **Document thoroughly**: Add docstrings to all classes and methods
2. **Use dataclasses**: IR nodes should be dataclasses for consistency
3. **Provide defaults**: Use sensible defaults for optional fields
4. **Handle edge cases**: Consider None values, empty lists, etc.
5. **Maintain backward compatibility**: Don't break existing behavior
6. **Follow the visitor pattern**: Always implement both backends
7. **Add comprehensive tests**: Cover normal cases and edge cases
8. **Use type hints**: All public APIs should have type annotations
