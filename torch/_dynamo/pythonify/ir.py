"""
RuntimeWrapper IR Node definitions for torch.compile pythonify feature.

This module defines the intermediate representation (IR) nodes that capture
the runtime machinery of torch.compile. The IR serves as a structured
representation that can be consumed by multiple code generation backends:

- gen_binary: Produces compiled artifacts (current behavior)
- gen_python: Emits explicit Python source code

The IR uses the visitor pattern to allow different backends to traverse
and process the node tree without modifying the node classes themselves.

Object ID Approach for Parameter/Buffer Access
==============================================
The pythonify feature uses Python object IDs to reference model parameters
and buffers in generated code. When `ArgumentSource.OBJECT_ID` is used, the
`object_id` field of `ArgumentExtractionNode` contains the result of calling
`id(tensor)` on the actual tensor object at compile time.

At runtime, the generated code uses ctypes to reconstruct the object:

    import ctypes
    def obj_from_id(obj_id):
        return ctypes.cast(obj_id, ctypes.py_object).value

IMPORTANT LIMITATIONS:
----------------------
1. PROCESS-LOCAL ONLY: Object IDs are memory addresses valid only within
   the Python process where they were captured. The generated file CANNOT
   be used in a different process or after a Python restart.

2. LIFETIME DEPENDENCY: The original tensor objects must remain alive for
   the generated code to work. If the model is garbage collected or modified,
   the object IDs become invalid.

3. NOT SERIALIZABLE: The generated Python file should not be saved for later
   use in a different session. It is intended for immediate execution via
   exec() within the same Python process.

4. UNSAFE IF MISUSED: Using ctypes to cast an invalid object ID will crash
   the Python interpreter or produce undefined behavior. Only use the
   generated code in the same process immediately after compilation.

The object ID approach is used because:
- It avoids needing the model variable in the exec() scope
- It guarantees we get the exact tensor that was compiled against
- It works with any model structure (nested modules, dynamic attributes)
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional, TYPE_CHECKING


if TYPE_CHECKING:
    from torch import Tensor
    from torch.fx import GraphModule


class ModelSource(Enum):
    """
    Specifies where the model object comes from for exec() compatibility.

    When generated code is executed via exec(code, f_globals, f_locals), the
    model object must be accessible. This enum specifies where to find it:

    F_LOCALS: Model is in frame locals (e.g., f_locals["model"])
        Use this when model is a local variable in the calling function.
    F_GLOBALS: Model is in frame globals (e.g., f_globals["model"])
        Use this when model is defined at module level.
    CLOSURE: Model is directly accessible as a variable (e.g., just "model")
        Use this when the variable is already in scope (passed to exec globals).
    """

    F_LOCALS = auto()
    F_GLOBALS = auto()
    CLOSURE = auto()


class ArgumentSource(Enum):
    """
    Specifies where an argument value comes from when generating code.

    MODEL_ATTRIBUTE: Attribute from the model object (e.g., model.W)
    F_LOCALS: Value from frame locals dictionary (e.g., f_locals["x"])
    F_GLOBALS: Value from frame globals dictionary (e.g., f_globals["torch"])
    CONSTANT: Literal constant value
    BUFFER: Registered buffer from nn.Module
    PARAMETER: Registered parameter from nn.Module
    OBJECT_ID: Value retrieved via object ID using ctypes. The object_id field
        of ArgumentExtractionNode contains the Python id() of the object at
        compile time. At runtime, ctypes.cast(object_id, ctypes.py_object).value
        is used to retrieve the object. This is only valid within the same
        Python process and requires the original object to still be alive.
    """

    MODEL_ATTRIBUTE = auto()
    F_LOCALS = auto()
    F_GLOBALS = auto()
    CONSTANT = auto()
    BUFFER = auto()
    PARAMETER = auto()
    OBJECT_ID = auto()


class GuardType(Enum):
    """
    Types of guards that can be checked at runtime.

    SHAPE: Guard on tensor shape (e.g., x.shape[0] == 3)
    DTYPE: Guard on tensor dtype (e.g., x.dtype == torch.float32)
    DEVICE: Guard on tensor device (e.g., x.device.type == 'cuda')
    VALUE: Guard on a constant value (e.g., x == 5)
    TENSOR_MATCH: Combined guard matching tensor properties
    IDENTITY: Guard on object identity
    TYPE: Guard on object type
    """

    SHAPE = auto()
    DTYPE = auto()
    DEVICE = auto()
    VALUE = auto()
    TENSOR_MATCH = auto()
    IDENTITY = auto()
    TYPE = auto()


class CUDAGraphPhase(Enum):
    """
    Indicates which phase of computation a CUDA graph is capturing.

    Training with CUDA graphs requires capturing forward and backward passes
    as separate graphs. The saved_tensors from forward must be managed
    carefully to be accessible during backward graph replay.

    INFERENCE: Single graph for inference-only computation (no backward pass).
        The entire forward pass is captured as one graph.
    FORWARD: Forward pass graph for training. Captures forward computation and
        establishes static buffers for saved_tensors that will be used by
        the backward graph.
    BACKWARD: Backward pass graph for training. Operates on saved_tensors from
        the forward graph and produces gradients.
    """

    INFERENCE = auto()
    FORWARD = auto()
    BACKWARD = auto()


class IRNode(abc.ABC):
    """
    Abstract base class for all RuntimeWrapper IR nodes.

    Each IR node represents a distinct piece of the runtime machinery
    that torch.compile generates. The visitor pattern is used to allow
    different code generation backends to process the IR without
    modifying the node classes.

    Subclasses must implement the accept() method to support the
    visitor pattern, enabling traversal by CodeGenVisitor implementations.
    """

    @abc.abstractmethod
    def accept(self, visitor: "CodeGenVisitor") -> Any:
        """
        Accept a visitor to process this node.

        This method should call the appropriate visit_* method on the
        visitor, passing self as the argument.

        Args:
            visitor: The CodeGenVisitor that will process this node

        Returns:
            The result from the visitor's visit method
        """
        pass


@dataclass
class ArgumentExtractionNode(IRNode):
    """
    IR node for extracting model parameters and input tensors.

    This node represents the code that retrieves values needed for
    the compiled computation. Values can come from various sources:
    - Model attributes (e.g., model.W for parameters)
    - Frame locals (e.g., f_locals["x"] for input tensors)
    - Frame globals (e.g., for accessing torch module)
    - Object ID (via ctypes, for process-local object retrieval)

    Attributes:
        name: Variable name to assign the extracted value to
        source: Where the value comes from (MODEL_ATTRIBUTE, F_LOCALS, etc.)
        access_path: Path to access the value (e.g., "W" for model.W, "x" for f_locals["x"])
        nested_path: For nested module hierarchies, list of attribute names to traverse
        object_id: Python id() of the object at compile time. When set, the generated
            code can use ctypes to retrieve the object directly via its memory address.
            This is only valid within the same Python process and requires the original
            object to still be alive. Used primarily for PARAMETER and BUFFER sources
            to avoid needing the model object in scope during exec().
    """

    name: str
    source: ArgumentSource
    access_path: str
    nested_path: list[str] = field(default_factory=list)
    object_id: Optional[int] = None

    def accept(self, visitor: "CodeGenVisitor") -> Any:
        return visitor.visit_argument_extraction(self)


@dataclass
class GuardCheckNode(IRNode):
    """
    IR node for runtime guard assertions.

    Guards ensure that the compiled code is being used with compatible
    inputs. If a guard fails, the computation should either raise an
    error or trigger recompilation.

    Attributes:
        guard_type: The type of guard (SHAPE, DTYPE, DEVICE, etc.)
        target_name: Name of the variable being guarded
        condition: The condition expression to check
        expected_value: Expected value for the guard check
        dimension: For SHAPE guards, the dimension index being checked
        error_message: Human-readable message for guard failure
        is_dynamic: If True, this dimension is marked as dynamic and guard
            is informational only (no assertion is generated). This is used
            when torch.compile is called with dynamic=True for certain dimensions.
        min_value: For dynamic shape guards, the minimum allowed value (optional)
        max_value: For dynamic shape guards, the maximum allowed value (optional)
    """

    guard_type: GuardType
    target_name: str
    condition: str
    expected_value: Any
    dimension: Optional[int] = None
    error_message: str = ""
    is_dynamic: bool = False
    min_value: Optional[int] = None
    max_value: Optional[int] = None

    def accept(self, visitor: "CodeGenVisitor") -> Any:
        return visitor.visit_guard_check(self)


@dataclass
class AOTAutogradWrapperNode(IRNode):
    """
    IR node for AOT Autograd function generation.

    This node encapsulates the torch.autograd.Function that wraps the
    Inductor-compiled forward and backward functions. It contains all
    the information needed to generate the autograd wrapper.

    Attributes:
        class_name: Name of the generated autograd.Function class
        forward_graph: The compiled forward FX graph
        backward_graph: The compiled backward FX graph (if differentiable)
        saved_tensors_indices: Indices of tensors to save for backward pass
        num_inputs: Number of input tensors to the forward function
        num_outputs: Number of outputs from the forward function
        metadata: Additional metadata from AOT Autograd compilation
        serialized_forward_code: Pre-serialized Python code for forward function.
            When provided, this code is embedded directly in the generated output
            instead of serializing forward_graph at generation time.
        serialized_backward_code: Pre-serialized Python code for backward function.
            When provided, this code is embedded directly in the generated output
            instead of serializing backward_graph at generation time.
        forward_input_names: Names of inputs to the forward graph (in order).
            Used for generating correct function signatures.
        forward_output_names: Names of outputs from the forward graph.
            Used for generating correct return statements.
        backward_input_names: Names of inputs to the backward graph (in order).
            Typically includes grad_outputs and saved tensors.
        backward_output_names: Names of outputs from the backward graph.
            These are the gradients with respect to each input.
        compiled_forward: Reference to a callable for the forward pass.
            When set, the generated code will reference this callable.
        compiled_backward: Reference to a callable for the backward pass.
            When set, the generated code will reference this callable.
    """

    class_name: str
    forward_graph: Optional["GraphModule"] = None
    backward_graph: Optional["GraphModule"] = None
    saved_tensors_indices: list[int] = field(default_factory=list)
    num_inputs: int = 0
    num_outputs: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    serialized_forward_code: Optional[str] = None
    serialized_backward_code: Optional[str] = None
    forward_input_names: list[str] = field(default_factory=list)
    forward_output_names: list[str] = field(default_factory=list)
    backward_input_names: list[str] = field(default_factory=list)
    backward_output_names: list[str] = field(default_factory=list)
    compiled_forward: Optional[Callable] = None
    compiled_backward: Optional[Callable] = None

    def accept(self, visitor: "CodeGenVisitor") -> Any:
        return visitor.visit_aot_autograd_wrapper(self)


@dataclass
class CUDAGraphSetupNode(IRNode):
    """
    IR node for CUDA graph capture and replay setup.

    When CUDA graphs are enabled, this node represents the code that
    sets up graph capture, manages streams, and handles memory allocation.

    For training mode, CUDA graphs are captured separately for forward and
    backward passes. The forward graph produces saved_tensors that must be
    stored in static buffers for the backward graph to access. This node
    can represent either:
    1. A single inference graph (phase=INFERENCE)
    2. A forward training graph (phase=FORWARD)
    3. A backward training graph (phase=BACKWARD)
    4. A paired forward+backward setup (when backward_graph_id is set)

    Attributes:
        graph_id: Unique identifier for this CUDA graph
        warmup_runs: Number of warmup runs before capture
        capture_mode: CUDA graph capture mode (e.g., "thread_local")
        stream_name: Name of the CUDA stream to use
        pool_id: Memory pool identifier for graph allocation
        static_inputs: Whether inputs are static (reused across calls)
        static_input_indices: List of indices for inputs that are static
            (parameters/buffers). Static inputs do not need to be copied
            before graph replay as their memory addresses are stable.
            Non-static (dynamic) inputs need to be copied into pre-allocated
            buffers before each replay.
        phase: Indicates whether this graph is for inference, forward, or
            backward pass. Training requires separate graphs.
        backward_graph_id: For forward graphs in training mode, the ID of the
            paired backward graph. When set, indicates this is a training setup
            with separate forward/backward graphs.
        saved_tensor_indices: For forward graphs, indices of outputs that are
            saved_tensors needed by the backward pass. For backward graphs,
            indices of inputs that are the saved_tensors from forward.
        num_forward_outputs: For forward graphs, the number of actual outputs
            (excluding saved_tensors). The forward graph may produce additional
            tensors that are saved for backward but not returned to the user.
        device_index: CUDA device index for graph capture in multi-GPU scenarios.
            When set, the generated code will set the device before capture.
        force_cudagraph_sync: Whether to synchronize after graph replay.
            Useful for debugging or when downstream operations require sync.
        skip_dynamic_graphs: Whether to skip CUDA graph capture for dynamic shapes.
            When True, the generated code will check if any input shapes differ
            from the shapes seen during graph capture and fall back to direct
            function execution instead of graph replay. CUDA graphs require
            fixed tensor shapes, so this provides graceful degradation for
            models with dynamic input dimensions.
    """

    graph_id: str
    warmup_runs: int = 1
    capture_mode: str = "thread_local"
    stream_name: str = "default"
    pool_id: Optional[str] = None
    static_inputs: bool = False
    static_input_indices: list[int] = field(default_factory=list)
    phase: "CUDAGraphPhase" = None  # type: ignore[assignment]
    backward_graph_id: Optional[str] = None
    saved_tensor_indices: list[int] = field(default_factory=list)
    num_forward_outputs: Optional[int] = None
    device_index: Optional[int] = None
    force_cudagraph_sync: bool = False
    skip_dynamic_graphs: bool = False

    def __post_init__(self) -> None:
        """Set default phase to INFERENCE if not specified."""
        if self.phase is None:
            from .ir import CUDAGraphPhase
            self.phase = CUDAGraphPhase.INFERENCE

    def is_training_setup(self) -> bool:
        """
        Check if this node represents a training setup with forward/backward.

        Returns:
            True if this is a forward graph with a paired backward graph,
            or if this is a backward graph.
        """
        return (
            self.backward_graph_id is not None
            or self.phase == CUDAGraphPhase.FORWARD
            or self.phase == CUDAGraphPhase.BACKWARD
        )

    def is_forward_graph(self) -> bool:
        """Check if this is a forward pass graph for training."""
        return self.phase == CUDAGraphPhase.FORWARD

    def is_backward_graph(self) -> bool:
        """Check if this is a backward pass graph for training."""
        return self.phase == CUDAGraphPhase.BACKWARD

    def accept(self, visitor: "CodeGenVisitor") -> Any:
        return visitor.visit_cuda_graph_setup(self)


@dataclass
class CallableInvocationNode(IRNode):
    """
    IR node for invoking the compiled callable.

    This node represents the actual call to the compiled function
    with all extracted arguments in the correct order.

    Attributes:
        callable_name: Name of the callable to invoke
        argument_names: Ordered list of argument variable names
        result_name: Variable name to store the result
        is_autograd_function: Whether callable is an autograd.Function.apply
        args_as_list: Whether to pass arguments as a single list
            (e.g., compiled_fn([arg1, arg2]) vs compiled_fn(arg1, arg2))
            Inductor's call() function expects args as a list.
        extract_first_output: Whether to extract the first element from the
            result tuple. Inductor's call() returns (output, saved_tensors...),
            so we need to extract output = result[0].
    """

    callable_name: str
    argument_names: list[str] = field(default_factory=list)
    result_name: str = "result"
    is_autograd_function: bool = False
    args_as_list: bool = False
    extract_first_output: bool = False

    def accept(self, visitor: "CodeGenVisitor") -> Any:
        return visitor.visit_callable_invocation(self)


class KernelType(Enum):
    """
    Type of compiled kernel to load.

    TRITON: GPU kernel compiled via Triton compiler (cubin, ptx)
    CPP: CPU kernel compiled via C++ compiler (.so, .dll)
    PYTHON: Pure Python wrapper
    INLINE: Kernel embedded inline as base64 in the generated code
    """

    TRITON = auto()
    CPP = auto()
    PYTHON = auto()
    INLINE = auto()


@dataclass
class KernelLoadNode(IRNode):
    """
    IR node for loading a serialized compiled kernel.

    This node represents the code needed to load a pre-compiled kernel
    from disk. Kernels can be Triton GPU kernels, C++ CPU kernels, or
    Python wrappers.

    Attributes:
        kernel_type: Type of kernel to load (TRITON, CPP, PYTHON, INLINE)
        kernel_id: Unique identifier for this kernel
        kernel_path: Path to the kernel file (relative to output directory)
        entry_point: Function/symbol name to call in the loaded kernel
        variable_name: Variable name to assign the loaded kernel to
        inline_content: Base64-encoded kernel content (for INLINE kernels)
        metadata: Additional kernel-specific metadata
    """

    kernel_type: KernelType
    kernel_id: str
    kernel_path: str = ""
    entry_point: str = "call"
    variable_name: str = "compiled_fn"
    inline_content: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def accept(self, visitor: "CodeGenVisitor") -> Any:
        return visitor.visit_kernel_load(self)


@dataclass
class ReturnResultNode(IRNode):
    """
    IR node for capturing and exposing the computation result.

    This node is essential for exec() compatibility. When code is executed
    via exec(), the result must be assigned to a variable that persists
    in the locals dictionary after exec() completes.

    Attributes:
        result_name: Name of the variable containing the result
        expose_as: Name to expose the result as in f_locals (for exec())
    """

    result_name: str = "result"
    expose_as: str = "y"

    def accept(self, visitor: "CodeGenVisitor") -> Any:
        return visitor.visit_return_result(self)


class RegionExecutionMode(Enum):
    """
    Specifies how multiple compiled regions should be executed.

    SEQUENTIAL: Regions execute in order, each passing results to the next.
        This is the typical pattern when graph breaks occur due to unsupported
        operations. The output of region N becomes an input to region N+1.

    GUARD_DISPATCH: Regions are alternatives; the first one whose guards pass
        is executed. This is used when the same code may compile differently
        based on input properties (shape, dtype, etc.).
    """

    SEQUENTIAL = auto()
    GUARD_DISPATCH = auto()


@dataclass
class CompiledRegionNode(IRNode):
    """
    IR node representing a single compiled region in a multi-region graph.

    When graph breaks occur during compilation, torch.compile generates multiple
    compiled regions. This node encapsulates all the information about one region
    including its guards, callable, inputs, and outputs.

    The region can be part of a sequential chain (graph break pattern) or one of
    several alternative implementations selected by guards (recompilation pattern).

    Attributes:
        region_id: Unique identifier for this region (e.g., "region_0")
        guards: List of guard IR nodes that must pass to use this region
        ir: The RuntimeWrapperIR containing all nodes for this region
        input_names: Names of inputs this region expects
        output_names: Names of outputs this region produces
        resume_target: For sequential execution, the next region to execute.
            If None, this is the final region.
        is_entry_region: True if this is the first region in the sequence/dispatch
        metadata: Additional region-specific metadata
    """

    region_id: str
    guards: list["GuardCheckNode"] = field(default_factory=list)
    ir: Optional["RuntimeWrapperIR"] = None
    input_names: list[str] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)
    resume_target: Optional[str] = None
    is_entry_region: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def accept(self, visitor: "CodeGenVisitor") -> Any:
        return visitor.visit_compiled_region(self)


@dataclass
class MultiRegionDispatchNode(IRNode):
    """
    IR node for dispatching between multiple compiled regions.

    This node generates the control flow logic that selects which compiled
    region to execute based on guard evaluation. There are two execution modes:

    1. SEQUENTIAL: For graph breaks, regions are chained. The first region
       runs, its output is captured, then passed to the next region.

    2. GUARD_DISPATCH: For recompilation, guards are checked in order and
       the first matching region is executed.

    Attributes:
        regions: List of CompiledRegionNode in execution/priority order
        execution_mode: How to combine regions (SEQUENTIAL or GUARD_DISPATCH)
        fallback_to_eager: If True, fall back to eager execution when no guards pass
        dispatch_table_name: Variable name for the generated dispatch function
        metadata: Additional dispatch-specific metadata
    """

    regions: list[CompiledRegionNode] = field(default_factory=list)
    execution_mode: RegionExecutionMode = RegionExecutionMode.SEQUENTIAL
    fallback_to_eager: bool = False
    dispatch_table_name: str = "_dispatch_compiled_regions"
    metadata: dict[str, Any] = field(default_factory=dict)

    def accept(self, visitor: "CodeGenVisitor") -> Any:
        return visitor.visit_multi_region_dispatch(self)


class CodeGenVisitor(abc.ABC):
    """
    Abstract base class for IR code generation visitors.

    Implementations of this visitor traverse the IR node tree and
    generate output appropriate to their backend:

    - BinaryCodeGenVisitor: Produces compiled artifacts (existing behavior)
    - PythonCodeGenVisitor: Emits Python source code

    Each visit method receives the node to process and should return
    an appropriate result (e.g., code string, compiled callable, etc.).
    """

    @abc.abstractmethod
    def visit_argument_extraction(self, node: ArgumentExtractionNode) -> Any:
        """Process an argument extraction node."""
        pass

    @abc.abstractmethod
    def visit_guard_check(self, node: GuardCheckNode) -> Any:
        """Process a guard check node."""
        pass

    @abc.abstractmethod
    def visit_aot_autograd_wrapper(self, node: AOTAutogradWrapperNode) -> Any:
        """Process an AOT Autograd wrapper node."""
        pass

    @abc.abstractmethod
    def visit_cuda_graph_setup(self, node: CUDAGraphSetupNode) -> Any:
        """Process a CUDA graph setup node."""
        pass

    @abc.abstractmethod
    def visit_callable_invocation(self, node: CallableInvocationNode) -> Any:
        """Process a callable invocation node."""
        pass

    @abc.abstractmethod
    def visit_kernel_load(self, node: KernelLoadNode) -> Any:
        """Process a kernel load node."""
        pass

    @abc.abstractmethod
    def visit_return_result(self, node: ReturnResultNode) -> Any:
        """Process a return result node."""
        pass

    @abc.abstractmethod
    def visit_compiled_region(self, node: CompiledRegionNode) -> Any:
        """Process a compiled region node."""
        pass

    @abc.abstractmethod
    def visit_multi_region_dispatch(self, node: MultiRegionDispatchNode) -> Any:
        """Process a multi-region dispatch node."""
        pass


@dataclass
class RuntimeWrapperIR:
    """
    Container for a complete RuntimeWrapper IR representation.

    This class holds an ordered list of IR nodes that together represent
    all the runtime machinery for a compiled function. The nodes should
    be processed in order to generate the complete runtime wrapper.

    Attributes:
        nodes: Ordered list of IR nodes
        metadata: Additional compilation metadata
        source_info: Information about the source function being compiled
    """

    nodes: list[IRNode] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    source_info: dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: IRNode) -> None:
        """Add a node to the IR."""
        self.nodes.append(node)

    def accept_all(self, visitor: CodeGenVisitor) -> list[Any]:
        """
        Apply a visitor to all nodes in order.

        Args:
            visitor: The visitor to apply

        Returns:
            List of results from visiting each node
        """
        return [node.accept(visitor) for node in self.nodes]

    def get_nodes_by_type(self, node_type: type) -> list[IRNode]:
        """
        Retrieve all nodes of a specific type.

        Args:
            node_type: The type of nodes to retrieve

        Returns:
            List of nodes matching the specified type
        """
        return [node for node in self.nodes if isinstance(node, node_type)]
