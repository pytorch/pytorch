"""
RuntimeWrapper IR Node definitions for torch.compile pythonify feature.

This module defines the intermediate representation (IR) nodes that capture
the runtime machinery of torch.compile. The IR serves as a structured
representation that can be consumed by multiple code generation backends:

- gen_binary: Produces compiled artifacts (current behavior)
- gen_python: Emits explicit Python source code

The IR uses the visitor pattern to allow different backends to traverse
and process the node tree without modifying the node classes themselves.

Preferred Model Reconstruction Path (Golden Path)
================================================
The pythonify IR is designed to reconstruct parameters, buffers, and
submodules directly from an in-scope nn.Module instance. Callers are
expected to supply the module (commonly ``self``) in the exec namespace and
describe where it lives via ``ModelSource``. This structured path is the
"golden path" for pythonify and should be the default for nn.Module flows.

Golden Path Usage Example
-------------------------
Below is the recommended pattern for using pythonify with an nn.Module.
The key points are: (1) configure ModelSource in CompilationArtifacts,
(2) use PARAMETER/BUFFER sources with nested_path for module attributes,
and (3) pass the model in the exec namespace when running the generated code.

::

    import torch
    import torch.nn as nn
    from torch._dynamo.pythonify import set_model_reference, get_model_source
    from torch._dynamo.pythonify.ir import ModelSource, ArgumentSource
    from torch._dynamo.pythonify.pipeline import (
        CompilationArtifacts,
        RuntimeWrapperPipeline,
    )

    # 1. Define your nn.Module
    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(10, 20)
            self.decoder = nn.Linear(20, 10)
            self.register_buffer("scale", torch.ones(10))

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x * self.scale

    model = MyModel()

    # 2. Set up pythonify context with ModelSource (default is CLOSURE)
    #    This tells codegen how the model will be referenced in exec()
    set_model_reference(model, model_source=ModelSource.CLOSURE)

    # 3. Create CompilationArtifacts with model_name and model_source
    #    The pipeline will populate PARAMETER/BUFFER sources with nested_path
    artifacts = CompilationArtifacts(
        model_name="model",
        model_source=ModelSource.CLOSURE,
        # ... other fields populated by torch.compile ...
    )

    # 4. Build IR via pipeline - this creates ArgumentExtractionNodes with:
    #    - source=PARAMETER for encoder.weight, encoder.bias, decoder.weight, etc.
    #    - source=BUFFER for scale
    #    - nested_path=["encoder", "weight"] etc. for attribute traversal
    pipeline = RuntimeWrapperPipeline(artifacts)
    ir = pipeline.build()

    # 5. Generate Python code - the code will emit module-based access:
    #    W_encoder = model.encoder.weight
    #    b_encoder = model.encoder.bias
    #    scale = model.scale
    #    (NOT obj_from_id(12345678) which is the legacy fallback)
    from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
    visitor = PythonCodeGenVisitor()
    code = visitor.generate(ir)

    # 6. Execute the generated code with model in namespace
    #    Because we used ModelSource.CLOSURE, "model" should be directly accessible
    namespace = {
        "model": model,
        "compiled_fn": compiled_callable,
        "f_locals": {"x": input_tensor},
    }
    exec(code, namespace)
    result = namespace["y"]

    # Benefits of the golden path:
    # - PROCESS-PORTABLE: Code works in any Python process
    # - LIVE ACCESS: Sees current parameter values (not stale compile-time values)
    # - SAFE: No ctypes memory access that can crash on invalid IDs
    # - NATURAL: Matches typical nn.Module attribute access patterns

Object ID Fallback (Legacy)
---------------------------
``ArgumentSource.OBJECT_ID`` remains as a fallback for non-module inputs or
ad-hoc objects that cannot be recovered from a module reference. When used,
``ArgumentExtractionNode.object_id`` holds ``id(obj)`` from compile time and
the generated code will reconstruct via ``ctypes``. This path is intentionally
process-local and should not be used for typical nn.Module reconstruction.

IMPORTANT LIMITATIONS OF OBJECT_ID (fallback path only):
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

    The pythonify "golden path" expects callers to provide an nn.Module in the
    exec namespace. ModelSource tells codegen how to reference that module:

    F_LOCALS: Model is in frame locals (e.g., f_locals["model"])
        Use this when model is a local variable in the calling function.
    F_GLOBALS: Model is in frame globals (e.g., f_globals["model"])
        Use this when model is defined at module level.
    CLOSURE: Model is directly accessible as a variable (e.g., just "model")
        Use this when the variable is already in scope (passed to exec globals).

    For nn.Module reconstruction, one of these sources should be provided so
    parameters/buffers/submodules can be recovered via attribute traversal.
    Object-id based reconstruction is a fallback and should not be the default.
    """

    F_LOCALS = auto()
    F_GLOBALS = auto()
    CLOSURE = auto()


class ArgumentSource(Enum):
    """
    Specifies where an argument value comes from when generating code.

    Preferred Sources for nn.Module Reconstruction (Golden Path)
    -------------------------------------------------------------
    For typical nn.Module pythonify flows, use these sources in combination
    with ModelSource to enable attribute-based reconstruction:

    PARAMETER: Registered parameter from nn.Module. Generated code accesses
        the parameter via module attribute traversal (e.g., ``model.layer.weight``).
        Requires ``nested_path`` to be populated with the attribute path.
        This is the preferred source for nn.Module parameters.

    BUFFER: Registered buffer from nn.Module. Generated code accesses the
        buffer via module attribute traversal (e.g., ``model.running_mean``).
        Requires ``nested_path`` to be populated with the attribute path.
        This is the preferred source for nn.Module buffers.

    MODEL_ATTRIBUTE: Generic attribute from the model object (e.g., model.config).
        Use when the attribute is not a parameter or buffer but still accessible
        from the in-scope module.

    Standard Sources
    ----------------
    F_LOCALS: Value from frame locals dictionary (e.g., f_locals["x"]).
        Use for input tensors and local variables.

    F_GLOBALS: Value from frame globals dictionary (e.g., f_globals["torch"]).
        Use for module-level globals and imports.

    CONSTANT: Literal constant value embedded directly in generated code.

    Legacy Fallback Source (Use Only When Necessary)
    -------------------------------------------------
    OBJECT_ID: Value retrieved via object ID using ctypes.

        WARNING: This is a LEGACY FALLBACK and should NOT be used for typical
        nn.Module reconstruction. It exists only for edge cases where:
        - The object cannot be accessed via module attribute traversal
        - No module context (ModelSource) is available
        - The object is a detached tensor or ad-hoc object

        When OBJECT_ID is used, ``ArgumentExtractionNode.object_id`` contains
        the Python id() of the object captured at compile time. The generated
        code uses ctypes to reconstruct the object from this memory address.

        Limitations of OBJECT_ID:
        - Process-local only: Object IDs are memory addresses, invalid across
          processes or after Python restart
        - Lifetime-dependent: Original object must remain alive
        - Not serializable: Generated code cannot be saved for later use
        - Unsafe if misused: Invalid IDs cause crashes or undefined behavior

        For nn.Module parameters/buffers, ALWAYS prefer PARAMETER or BUFFER
        sources with proper ``nested_path`` and ``ModelSource`` configuration.
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

    This node represents the code that retrieves values needed for the
    compiled computation. Values can come from various sources, and the
    choice of source determines how the generated code accesses the value.

    Golden Path (Preferred for nn.Module)
    -------------------------------------
    For nn.Module parameters and buffers, the recommended approach is:

    1. Set ``source`` to ``ArgumentSource.PARAMETER`` or ``ArgumentSource.BUFFER``
    2. Populate ``nested_path`` with the attribute path (e.g., ["layer1", "weight"])
    3. Ensure a ``ModelSource`` is configured in ``CompilationArtifacts``

    The generated code will then access the value via module attribute traversal
    (e.g., ``model.layer1.weight``), which is process-portable, supports live
    attribute access, and aligns with typical nn.Module usage.

    Available Sources
    -----------------
    - Model attributes: source=PARAMETER/BUFFER/MODEL_ATTRIBUTE with nested_path
    - Frame locals: source=F_LOCALS for input tensors (e.g., f_locals["x"])
    - Frame globals: source=F_GLOBALS for imports (e.g., f_globals["torch"])
    - Object ID: source=OBJECT_ID when module path unavailable (LEGACY FALLBACK)

    Attributes:
        name: Variable name to assign the extracted value to
        source: Where the value comes from (PARAMETER, BUFFER, F_LOCALS, etc.)
        access_path: Path to access the value (e.g., "W" for model.W, "x" for f_locals["x"])
        nested_path: For nested module hierarchies, list of attribute names to traverse.
            Required for PARAMETER/BUFFER sources to enable golden path reconstruction.
            Example: ["encoder", "layer1", "weight"] produces ``model.encoder.layer1.weight``
        object_id: Python id() of the object at compile time.

            WARNING: This field is part of the LEGACY FALLBACK path. It should only
            be set when ``source=OBJECT_ID`` and module-based reconstruction is not
            possible. When set, generated code uses ctypes to retrieve the object
            directly via its memory address.

            IMPORTANT: For nn.Module parameters/buffers, DO NOT use object_id.
            Instead, use ``source=PARAMETER`` or ``source=BUFFER`` with a populated
            ``nested_path``. The module-based approach is:
            - Process-portable (works across different Python processes)
            - Live-access (sees current parameter values, not stale compile-time values)
            - Safe (no ctypes memory access that can crash on invalid IDs)
            - Natural (matches typical nn.Module attribute access patterns)

            The object_id path remains only for edge cases: detached tensors, ad-hoc
            objects, or scenarios where no module context is available.
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


# Wrapper IR nodes
@dataclass
class EffectTokensWrapperNode(IRNode):
    """
    Injects effect tokens on input and strips them from outputs.

    Attributes:
        token_count: Number of effect tokens to prepend to args and drop from outputs.
    """

    token_count: int = 0

    def accept(self, visitor: "CodeGenVisitor") -> Any:
        return visitor.visit_effect_tokens_wrapper(self)


@dataclass
class AOTDispatchSubclassWrapperNode(IRNode):
    """
    Handles unwrapping/rewrapping tensor subclasses around compiled callables.

    Attributes capture metadata from ViewAndMutationMeta for subclass handling.
    """

    subclass_inp_meta: Any = None
    subclass_fw_graph_out_meta: Any = None
    num_fw_outs_saved_for_bw: int = 0
    maybe_subclass_meta: Any = None

    def accept(self, visitor: "CodeGenVisitor") -> Any:
        return visitor.visit_aot_dispatch_subclass_wrapper(self)


@dataclass
class FunctionalizedRngRuntimeWrapperNode(IRNode):
    """
    Manages RNG functionalization inputs/outputs and CUDA offset updates.
    """

    is_rng_op_functionalized: bool = False
    num_outputs_rng_offset: int = 0
    num_forward_returns: int = 0
    num_graphsafe_rng_states: int = 0
    graphsafe_rng_state_index: Optional[int] = None

    def accept(self, visitor: "CodeGenVisitor") -> Any:
        return visitor.visit_functionalized_rng_runtime_wrapper(self)


@dataclass
class FakifiedOutWrapperNode(IRNode):
    """
    Re-fakifies outputs using traced metadata and reported strides.
    """

    out_metas: Any = None
    fwd_output_strides: Optional[list[Any]] = None

    def accept(self, visitor: "CodeGenVisitor") -> Any:
        return visitor.visit_fakified_out_wrapper(self)


@dataclass
class RuntimeWrapperNode(IRNode):
    """
    Runtime epilogue handling detach, alias, and autocast/grad restoration.
    """

    indices_of_inps_to_detach: list[int] = field(default_factory=list)
    disable_amp: bool = False
    runtime_metadata: Any = None
    trace_joint: bool = False

    def accept(self, visitor: "CodeGenVisitor") -> Any:
        return visitor.visit_runtime_wrapper(self)


@dataclass
class AOTDedupeWrapperNode(IRNode):
    """
    Re-inserts duplicated arguments removed during compile-time deduping.
    """

    keep_arg_mask: Optional[list[bool]] = None
    add_dupe_map: Optional[list[tuple[int, int]]] = None
    needs_post_compile: bool = False
    old_input_metadata: Any = None

    def accept(self, visitor: "CodeGenVisitor") -> Any:
        return visitor.visit_aot_dedupe_wrapper(self)


@dataclass
class AOTSyntheticBaseWrapperNode(IRNode):
    """
    Reconstructs views from synthetic bases and reapplies metadata mutations.
    """

    synthetic_base_info: Any = None
    aliased_arg_idx_with_metadata_mutations: Optional[list[int]] = None
    old_input_info: Any = None
    needs_post_compile: bool = False
    trace_joint: bool = False

    def accept(self, visitor: "CodeGenVisitor") -> Any:
        return visitor.visit_aot_synthetic_base_wrapper(self)


@dataclass
class DebugAssertWrapperNode(IRNode):
    """
    Asserts requires_grad expectations at runtime when debug asserts are enabled.
    """

    flat_requires_grad: Optional[list[bool]] = None

    def accept(self, visitor: "CodeGenVisitor") -> Any:
        return visitor.visit_debug_assert_wrapper(self)


@dataclass
class AOTAutogradWrapperNode(IRNode):
    """
    IR node for AOT Autograd function generation.

    This node encapsulates the torch.autograd.Function that wraps the
    Inductor-compiled forward and backward functions. It contains all
    the information needed to generate the autograd wrapper, including
    lazy backward compilation and forward/backward stitching.

    Lazy Backward Compilation
    -------------------------
    AOTDispatchAutograd.post_compile supports lazy backward compilation, where
    the backward graph is not compiled until backward() is actually called.
    This optimization avoids compiling backward for inference-only workloads.

    When lazy backward is enabled:
    - compiled_backward is None at generation time
    - has_lazy_backward is True
    - lazy_bw_module holds the uncompiled backward GraphModule
    - lazy_bw_placeholder_list holds the placeholder nodes for backward inputs

    At runtime, when backward() is first called:
    1. The tracing and compile contexts are restored
    2. The backward module is compiled via bw_compiler(bw_module, placeholder_list)
    3. The compiled backward is cached for future calls

    Forward/Backward Stitching
    --------------------------
    AOTDispatchAutograd.post_compile stitches forward and backward callables
    into a torch.autograd.Function class. This involves:

    1. Saved tensor management: Forward outputs are sliced into user outputs
       and saved tensors/symints for backward. Slices are specified by:
       - tensors_saved_for_bw_with_vc_check_slice: tensors needing version check
       - tensors_saved_for_bw_no_vc_check_slice: tensors without version check
       - symints_saved_for_bw_slice: symbolic integers saved for backward
       - num_symints_saved_for_bw: count of saved symints

    2. RNG state pairing: When CUDA graphs with graphsafe RNG are used:
       - Forward stores RNG generator state per iteration
       - Backward restores matching RNG state from forward iteration
       - num_graphsafe_rng_states and graphsafe_rng_state_index control this

    3. Input detachment: Certain inputs are detached before backward to prevent
       gradient accumulation issues (indices_of_inps_to_detach)

    4. Subclass tangent processing: When inputs are tensor subclasses, tangents
       (gradient outputs) need special handling via maybe_subclass_meta

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
            May be None if has_lazy_backward is True.

        Lazy Backward Fields:
        has_lazy_backward: True if backward compilation is deferred until runtime.
            When True, compiled_backward is None and lazy_bw_module holds the
            uncompiled backward graph.
        lazy_bw_module: The uncompiled backward GraphModule for lazy compilation.
            Only set when has_lazy_backward is True.
        lazy_bw_placeholder_list: List of placeholder nodes for backward inputs.
            Passed to bw_compiler during lazy compilation.
        lazy_bw_saved_context: Serializable representation of TracingContext
            needed to restore tracing state during lazy backward compilation.
        lazy_bw_saved_compile_context: Serializable representation of CompileContext
            needed to restore compile state during lazy backward compilation.

        Saved Tensor Slice Fields:
        tensors_saved_for_bw_with_vc_check_slice: Slice indices for saved tensors
            that require version counter checks (detect in-place modifications).
        tensors_saved_for_bw_no_vc_check_slice: Slice indices for saved tensors
            that do not require version counter checks.
        symints_saved_for_bw_slice: Slice indices for symbolic integers saved
            for backward (e.g., dynamic shape values).
        num_symints_saved_for_bw: Count of symbolic integers saved for backward.
        dynamic_saved_tensors_idxs: Mapping from saved tensor index to set of
            dimension indices that have dynamic shapes.

        RNG State Fields:
        num_graphsafe_rng_states: Number of CUDA graph-safe RNG generator states
            to maintain for forward/backward RNG pairing.
        graphsafe_rng_state_index: Device index for CUDA RNG state management.
            When set, RNG generators are created on this device.
        is_rng_op_functionalized: Whether RNG operations were functionalized
            during AOT compilation.

        Autograd Assembly Fields:
        backward_state_indices: Indices of BackwardState objects in inputs.
            Used by compiled autograd for stateful backward passes.
        indices_of_inps_to_detach: List of input indices to detach before
            backward to prevent gradient accumulation issues.
        disable_amp: Whether autocast was disabled during compilation.
            Used to restore proper autocast context during backward.
        maybe_subclass_meta: SubclassMeta for tensor subclass handling.
            Contains grad_input_metas for processing tangent subclasses.
        fw_metadata: ViewAndMutationMeta containing runtime metadata about
            mutations, aliasing, and output structure.
        try_save_cache_entry_present: Whether a cache save callback is available
            for caching lazy-compiled backward.
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

    # Lazy backward compilation fields
    has_lazy_backward: bool = False
    lazy_bw_module: Optional["GraphModule"] = None
    lazy_bw_placeholder_list: Optional[list[Any]] = None
    lazy_bw_saved_context: Optional[dict[str, Any]] = None
    lazy_bw_saved_compile_context: Optional[dict[str, Any]] = None

    # Saved tensor slice metadata for fw/bw stitching
    tensors_saved_for_bw_with_vc_check_slice: Optional[tuple[int, int]] = None
    tensors_saved_for_bw_no_vc_check_slice: Optional[tuple[int, int]] = None
    symints_saved_for_bw_slice: Optional[tuple[int, int]] = None
    num_symints_saved_for_bw: int = 0
    dynamic_saved_tensors_idxs: dict[int, set[int]] = field(default_factory=dict)

    # RNG state pairing fields
    num_graphsafe_rng_states: int = 0
    graphsafe_rng_state_index: Optional[int] = None
    is_rng_op_functionalized: bool = False

    # Autograd assembly fields
    backward_state_indices: list[int] = field(default_factory=list)
    indices_of_inps_to_detach: list[int] = field(default_factory=list)
    disable_amp: bool = False
    maybe_subclass_meta: Optional[Any] = None
    fw_metadata: Optional[Any] = None
    try_save_cache_entry_present: bool = False

    def accept(self, visitor: "CodeGenVisitor") -> Any:
        return visitor.visit_aot_autograd_wrapper(self)

    def is_lazy_backward_enabled(self) -> bool:
        """
        Check if lazy backward compilation is enabled.

        Returns True if backward compilation is deferred to runtime.
        In this case, compiled_backward will be None and lazy_bw_module
        holds the uncompiled backward graph.
        """
        return self.has_lazy_backward and self.lazy_bw_module is not None

    def requires_rng_pairing(self) -> bool:
        """
        Check if RNG state pairing is required for forward/backward.

        Returns True if CUDA graph-safe RNG states need to be managed
        across forward/backward iterations.
        """
        return self.num_graphsafe_rng_states > 0

    def requires_saved_tensor_slicing(self) -> bool:
        """
        Check if saved tensor slicing is required.

        Returns True if forward outputs need to be sliced into user outputs
        and saved tensors for backward.
        """
        return (
            self.tensors_saved_for_bw_with_vc_check_slice is not None
            or self.tensors_saved_for_bw_no_vc_check_slice is not None
            or self.symints_saved_for_bw_slice is not None
        )


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
    def visit_effect_tokens_wrapper(self, node: EffectTokensWrapperNode) -> Any:
        """Process an effect tokens wrapper node."""
        pass

    @abc.abstractmethod
    def visit_aot_dispatch_subclass_wrapper(
        self, node: AOTDispatchSubclassWrapperNode
    ) -> Any:
        """Process an AOT subclass dispatch wrapper node."""
        pass

    @abc.abstractmethod
    def visit_functionalized_rng_runtime_wrapper(
        self, node: FunctionalizedRngRuntimeWrapperNode
    ) -> Any:
        """Process an RNG functionalization wrapper node."""
        pass

    @abc.abstractmethod
    def visit_fakified_out_wrapper(self, node: FakifiedOutWrapperNode) -> Any:
        """Process a fakified out wrapper node."""
        pass

    @abc.abstractmethod
    def visit_runtime_wrapper(self, node: RuntimeWrapperNode) -> Any:
        """Process a runtime epilogue wrapper node."""
        pass

    @abc.abstractmethod
    def visit_aot_dedupe_wrapper(self, node: AOTDedupeWrapperNode) -> Any:
        """Process an AOT dedupe wrapper node."""
        pass

    @abc.abstractmethod
    def visit_aot_synthetic_base_wrapper(
        self, node: AOTSyntheticBaseWrapperNode
    ) -> Any:
        """Process a synthetic base wrapper node."""
        pass

    @abc.abstractmethod
    def visit_debug_assert_wrapper(self, node: DebugAssertWrapperNode) -> Any:
        """Process a debug assert wrapper node."""
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


class WrapperStackSegment(Enum):
    """
    Logical segments of the post-compile wrapper stack.

    AOTAutograd applies wrappers in multiple phases. To keep ordering
    explicit in the RuntimeWrapperIR we annotate which segment each
    wrapper belongs to:

    * FORWARD_INFERENCE: Wrappers applied to the compiled forward/inference
      callable before autograd assembly. Expected order (inner-most callable
      first) is EffectTokensWrapper → AOTDispatchSubclassWrapper →
      FunctionalizedRngRuntimeWrapper → FakifiedOutWrapper.

    * AUTOGRAD_ASSEMBLY: Wrappers applied while stitching fw/bw (post_compile).
      Expected order (inner-most callable first) is AOTAutogradWrapper →
      RuntimeWrapper → DebugAssertWrapper.

    * DISPATCH: Dispatch-level wrappers added via _create_wrappers_for_dispatch
      and applied in reverse order at runtime. Expected order (inner-most
      callable first) is AOTSyntheticBaseWrapper → AOTDedupeWrapper.

    These segments allow codegen/pipeline to preserve reverse-application
    semantics without interleaving non-wrapper nodes (e.g., guards).
    """

    FORWARD_INFERENCE = auto()
    AUTOGRAD_ASSEMBLY = auto()
    DISPATCH = auto()


# Keep __all__ minimal and focused on public IR types.
__all__ = [
    "ArgumentExtractionNode",
    "ArgumentSource",
    "AOTAutogradWrapperNode",
    "AOTDedupeWrapperNode",
    "AOTDispatchSubclassWrapperNode",
    "AOTSyntheticBaseWrapperNode",
    "CallableInvocationNode",
    "CodeGenVisitor",
    "CompiledRegionNode",
    "CUDAGraphPhase",
    "CUDAGraphSetupNode",
    "DebugAssertWrapperNode",
    "EffectTokensWrapperNode",
    "FakifiedOutWrapperNode",
    "FunctionalizedRngRuntimeWrapperNode",
    "GuardCheckNode",
    "GuardType",
    "IRNode",
    "KernelLoadNode",
    "KernelType",
    "ModelSource",
    "MultiRegionDispatchNode",
    "RegionExecutionMode",
    "ReturnResultNode",
    "RuntimeWrapperIR",
    "WrapperStackSegment",
]


@dataclass
class RuntimeWrapperIR:
    """
    Container for a complete RuntimeWrapper IR representation.

    This class holds an ordered list of IR nodes that together represent
    all the runtime machinery for a compiled function. The nodes should
    be processed in order to generate the complete runtime wrapper.

    Wrapper ordering metadata
    -------------------------
    AOTAutograd applies wrappers in logical segments. To avoid interleaving
    wrapper ordering with non-wrapper nodes (guards, argument extraction), we
    track the expected wrapper stack per segment in `wrapper_stack_order`.
    The order recorded for each segment is **inner-most callable first**,
    matching the audit in wrapper_audit.md:

    - WrapperStackSegment.FORWARD_INFERENCE: EffectTokensWrapper →
      AOTDispatchSubclassWrapper → FunctionalizedRngRuntimeWrapper →
      FakifiedOutWrapper.
    - WrapperStackSegment.AUTOGRAD_ASSEMBLY: AOTAutogradWrapper →
      RuntimeWrapper → DebugAssertWrapper.
    - WrapperStackSegment.DISPATCH: AOTSyntheticBaseWrapper → AOTDedupeWrapper
      (post_compile applies these in reverse order at runtime).

    `wrapper_stack_metadata` can hold per-wrapper metadata placeholders keyed
    by wrapper name (e.g., "RuntimeWrapper" → {"indices_of_inps_to_detach": [...]})
    so pipeline/codegen can preserve reverse-application semantics without
    altering node ordering.

    Attributes:
        nodes: Ordered list of IR nodes
        metadata: Additional compilation metadata
        source_info: Information about the source function being compiled
        wrapper_stack_order: Expected wrapper order per segment (inner → outer)
        wrapper_stack_metadata: Optional metadata keyed by wrapper name
    """

    nodes: list[IRNode] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    source_info: dict[str, Any] = field(default_factory=dict)
    wrapper_stack_order: dict[WrapperStackSegment, list[str]] = field(
        default_factory=lambda: {
            WrapperStackSegment.FORWARD_INFERENCE: [],
            WrapperStackSegment.AUTOGRAD_ASSEMBLY: [],
            WrapperStackSegment.DISPATCH: [],
        }
    )
    wrapper_stack_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add_node(self, node: IRNode) -> None:
        """Add a node to the IR."""
        self.nodes.append(node)

    def record_wrapper(
        self,
        segment: WrapperStackSegment,
        wrapper_name: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Record a wrapper in a specific stack segment.

        Ordering is stored as inner-most → outer-most callable for the segment.
        Dispatch wrappers are applied in reverse at runtime; callers can use
        get_wrapper_order(..., reverse_for_application=True) to retrieve that
        ordering without mutating node sequencing.
        """

        stack = self.wrapper_stack_order.setdefault(segment, [])
        stack.append(wrapper_name)
        if metadata is not None:
            self.wrapper_stack_metadata[wrapper_name] = metadata

    def get_wrapper_order(
        self, segment: WrapperStackSegment, reverse_for_application: bool = False
    ) -> list[str]:
        """
        Retrieve wrapper order for a segment.

        Args:
            segment: WrapperStackSegment to query
            reverse_for_application: When True, reverse the ordering to match
                runtime application semantics (useful for DISPATCH).

        Returns:
            List of wrapper names for the requested segment.
        """

        ordered = list(self.wrapper_stack_order.get(segment, []))
        if reverse_for_application:
            ordered.reverse()
        return ordered

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
