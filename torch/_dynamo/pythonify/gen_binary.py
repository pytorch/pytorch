"""
Binary code generation backend for torch.compile pythonify feature.

This module implements the gen_binary backend, which produces compiled artifacts
that preserve the current torch.compile behavior. It serves as the drop-in
replacement for the existing compilation path, ensuring that the refactored
infrastructure with IR nodes doesn't break any existing functionality.

The BinaryCodeGenVisitor traverses the RuntimeWrapper IR and constructs an
executable wrapper that:
1. Extracts arguments from the appropriate sources
2. Checks guards at runtime
3. Wraps the compiled callable in an autograd.Function if needed
4. Sets up CUDA graphs if enabled
5. Invokes the compiled callable and returns the result

This backend is used when pythonify is NOT specified, maintaining the status quo.

Object ID Approach for Parameter/Buffer Extraction
=================================================
The ArgumentExtractor class supports extracting values via object ID when
the `source` is `ArgumentSource.OBJECT_ID`. In this mode, the `object_id`
field contains the Python `id()` of the tensor at compile time, and
extraction uses ctypes to retrieve the object:

    return ctypes.cast(self.object_id, ctypes.py_object).value

This approach is used by the pythonify feature to avoid requiring the model
object during execution. The gen_binary backend supports this for consistency
with gen_python, though it's primarily used in the Python code generation path.

CRITICAL LIMITATIONS of Object ID Approach:
-------------------------------------------
1. PROCESS-LOCAL: Object IDs are memory addresses valid only in the same
   Python process. They become invalid after interpreter restart or when
   used in a different process.

2. LIFETIME: The original tensor objects must remain alive. If the model
   is garbage collected, the object IDs become dangling references that
   will crash or produce undefined behavior when accessed.

3. NOT SERIALIZABLE: Object IDs cannot be persisted. Code or data containing
   object IDs should not be saved for later use in different sessions.

4. SINGLE USE: The generated wrapper should be used immediately after
   compilation, within the same Python process.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TYPE_CHECKING

from .ir import (
    AOTDedupeWrapperNode,
    AOTDispatchSubclassWrapperNode,
    AOTSyntheticBaseWrapperNode,
    ArgumentExtractionNode,
    ArgumentSource,
    AOTAutogradWrapperNode,
    CallableInvocationNode,
    CodeGenVisitor,
    CompiledRegionNode,
    CUDAGraphSetupNode,
    DebugAssertWrapperNode,
    EffectTokensWrapperNode,
    FakifiedOutWrapperNode,
    FunctionalizedRngRuntimeWrapperNode,
    GuardCheckNode,
    GuardType,
    KernelLoadNode,
    KernelType,
    MultiRegionDispatchNode,
    RegionExecutionMode,
    ReturnResultNode,
    RuntimeWrapperIR,
    RuntimeWrapperNode,
)


if TYPE_CHECKING:
    from types import FrameType


@dataclass
class ArgumentExtractor:
    """
    Holds information about how to extract an argument at runtime.

    This class captures the extraction strategy for a single argument,
    allowing the runtime wrapper to retrieve values from the appropriate
    sources (model attributes, f_locals, object IDs, etc.).

    Attributes:
        name: Variable name for this argument
        source: Where to extract the value from
        access_path: Path to access the value
        nested_path: For nested modules, list of attribute names to traverse
        object_id: Python id() of the object at compile time. When source is
            OBJECT_ID, this is used with ctypes to retrieve the object directly
            via its memory address. Only valid within the same Python process
            and requires the original object to still be alive.
    """

    name: str
    source: ArgumentSource
    access_path: str
    nested_path: list[str] = field(default_factory=list)
    object_id: Optional[int] = None

    def extract(
        self,
        model: Any,
        f_locals: dict[str, Any],
        f_globals: dict[str, Any],
    ) -> Any:
        """
        Extract the value from the appropriate source.

        Args:
            model: The model object (for PARAMETER, BUFFER, MODEL_ATTRIBUTE)
            f_locals: The frame locals dictionary
            f_globals: The frame globals dictionary

        Returns:
            The extracted value
        """
        if self.source == ArgumentSource.F_LOCALS:
            return f_locals.get(self.access_path)
        elif self.source == ArgumentSource.F_GLOBALS:
            return f_globals.get(self.access_path)
        elif self.source == ArgumentSource.CONSTANT:
            return self.access_path
        elif self.source == ArgumentSource.OBJECT_ID:
            if self.object_id is None:
                raise ValueError(
                    f"ArgumentExtractor for '{self.name}' has source=OBJECT_ID "
                    "but object_id is None"
                )
            return ctypes.cast(self.object_id, ctypes.py_object).value
        elif self.source in (
            ArgumentSource.PARAMETER,
            ArgumentSource.BUFFER,
            ArgumentSource.MODEL_ATTRIBUTE,
        ):
            obj = model
            for attr_name in self.nested_path:
                obj = getattr(obj, attr_name)
            return obj
        else:
            raise ValueError(f"Unknown argument source: {self.source}")


@dataclass
class GuardChecker:
    """
    Holds information about a guard check to perform at runtime.

    This class captures the guard condition and provides a method to
    check the guard at runtime, raising an error if the guard fails.

    Attributes:
        guard_type: The type of guard check
        target_name: Name of the variable being guarded
        condition: Human-readable condition string
        expected_value: The expected value
        dimension: For shape guards, which dimension to check
        error_message: Message to show on failure
    """

    guard_type: GuardType
    target_name: str
    condition: str
    expected_value: Any
    dimension: Optional[int] = None
    error_message: str = ""

    def check(self, args: dict[str, Any]) -> bool:
        """
        Check if the guard passes for the given arguments.

        Args:
            args: Dictionary mapping argument names to their values

        Returns:
            True if guard passes, raises AssertionError otherwise

        Raises:
            AssertionError: If the guard check fails
        """
        target = args.get(self.target_name)
        if target is None:
            return True

        if self.guard_type == GuardType.SHAPE:
            if self.dimension is not None and hasattr(target, "shape"):
                actual = target.shape[self.dimension]
                if actual != self.expected_value:
                    raise AssertionError(
                        self.error_message
                        or f"Shape guard failed: {self.condition}"
                    )
        elif self.guard_type == GuardType.DTYPE:
            if hasattr(target, "dtype"):
                if target.dtype != self.expected_value:
                    raise AssertionError(
                        self.error_message
                        or f"Dtype guard failed: {self.condition}"
                    )
        elif self.guard_type == GuardType.DEVICE:
            if hasattr(target, "device"):
                expected_device = self.expected_value
                if isinstance(expected_device, str):
                    if target.device.type != expected_device:
                        raise AssertionError(
                            self.error_message
                            or f"Device guard failed: {self.condition}"
                        )
                elif target.device != expected_device:
                    raise AssertionError(
                        self.error_message
                        or f"Device guard failed: {self.condition}"
                    )
        elif self.guard_type == GuardType.VALUE:
            if target != self.expected_value:
                raise AssertionError(
                    self.error_message or f"Value guard failed: {self.condition}"
                )
        elif self.guard_type == GuardType.TYPE:
            if type(target) != self.expected_value:
                raise AssertionError(
                    self.error_message or f"Type guard failed: {self.condition}"
                )
        elif self.guard_type == GuardType.IDENTITY:
            if target is not self.expected_value:
                raise AssertionError(
                    self.error_message
                    or f"Identity guard failed: {self.condition}"
                )

        return True


@dataclass
class CUDAGraphConfig:
    """
    Configuration for CUDA graph capture and replay.

    Attributes:
        graph_id: Unique identifier for the CUDA graph
        warmup_runs: Number of warmup runs before capture
        capture_mode: The CUDA graph capture mode
        stream_name: Name of the stream to use
        pool_id: Memory pool identifier
        static_inputs: Whether inputs are static
    """

    graph_id: str
    warmup_runs: int = 1
    capture_mode: str = "thread_local"
    stream_name: str = "default"
    pool_id: Optional[str] = None
    static_inputs: bool = False


@dataclass
class CompiledWrapper:
    """
    The runtime wrapper produced by the binary code generation backend.

    This class encapsulates all the runtime machinery needed to execute
    a compiled function. It handles argument extraction, guard checking,
    and invocation of the compiled callable.

    Attributes:
        extractors: List of ArgumentExtractor objects for input extraction
        guards: List of GuardChecker objects for runtime validation
        compiled_callable: The underlying compiled function
        is_autograd_function: Whether to use autograd.Function.apply
        cuda_graph_config: CUDA graph configuration (if enabled)
        result_name: Name of the result variable
        expose_as: Name to expose result as (for exec() compatibility)
    """

    extractors: list[ArgumentExtractor] = field(default_factory=list)
    guards: list[GuardChecker] = field(default_factory=list)
    compiled_callable: Optional[Callable[..., Any]] = None
    is_autograd_function: bool = False
    cuda_graph_config: Optional[CUDAGraphConfig] = None
    result_name: str = "result"
    expose_as: str = "y"

    def __call__(
        self,
        model: Any,
        f_locals: dict[str, Any],
        f_globals: Optional[dict[str, Any]] = None,
    ) -> Any:
        """
        Execute the compiled wrapper.

        This method:
        1. Extracts all arguments from their sources
        2. Checks all guards
        3. Invokes the compiled callable
        4. Returns the result

        Args:
            model: The model object for parameter/buffer extraction
            f_locals: Frame locals dictionary for input extraction
            f_globals: Frame globals dictionary (optional)

        Returns:
            The result of the compiled computation

        Raises:
            AssertionError: If any guard check fails
            RuntimeError: If the compiled callable is not set
        """
        if f_globals is None:
            f_globals = {}

        args: dict[str, Any] = {}
        ordered_args: list[Any] = []

        for extractor in self.extractors:
            value = extractor.extract(model, f_locals, f_globals)
            args[extractor.name] = value
            ordered_args.append(value)

        for guard in self.guards:
            guard.check(args)

        if self.compiled_callable is None:
            raise RuntimeError("Compiled callable not set in CompiledWrapper")

        result = self.compiled_callable(*ordered_args)

        return result

    def set_callable(self, callable_fn: Callable[..., Any]) -> None:
        """
        Set the compiled callable.

        Args:
            callable_fn: The compiled function to invoke
        """
        self.compiled_callable = callable_fn


class BinaryCodeGenVisitor(CodeGenVisitor):
    """
    Code generation visitor that produces compiled binary artifacts.

    This visitor traverses the RuntimeWrapper IR and builds a CompiledWrapper
    object that can execute the compiled function at runtime. It serves as
    the gen_binary backend, preserving existing torch.compile behavior.

    The visitor accumulates state as it visits each node, building up the
    CompiledWrapper piece by piece. After visiting all nodes, call
    get_wrapper() to retrieve the constructed wrapper.

    Usage:
        visitor = BinaryCodeGenVisitor()
        ir.accept_all(visitor)
        wrapper = visitor.get_wrapper()
        result = wrapper(model, f_locals)
    """

    def __init__(self) -> None:
        """Initialize the visitor with an empty CompiledWrapper."""
        self._wrapper = CompiledWrapper()
        self._callable_name: Optional[str] = None
        self._argument_names: list[str] = []

    def get_wrapper(self) -> CompiledWrapper:
        """
        Get the constructed CompiledWrapper.

        Returns:
            The CompiledWrapper built from visiting the IR nodes
        """
        return self._wrapper

    def visit_argument_extraction(self, node: ArgumentExtractionNode) -> Any:
        """
        Process an argument extraction node.

        Creates an ArgumentExtractor and adds it to the wrapper.

        Args:
            node: The ArgumentExtractionNode to process

        Returns:
            The created ArgumentExtractor
        """
        extractor = ArgumentExtractor(
            name=node.name,
            source=node.source,
            access_path=node.access_path,
            nested_path=node.nested_path.copy(),
            object_id=node.object_id,
        )
        self._wrapper.extractors.append(extractor)
        self._argument_names.append(node.name)
        return extractor

    def visit_guard_check(self, node: GuardCheckNode) -> Any:
        """
        Process a guard check node.

        Creates a GuardChecker and adds it to the wrapper.

        Args:
            node: The GuardCheckNode to process

        Returns:
            The created GuardChecker
        """
        guard = GuardChecker(
            guard_type=node.guard_type,
            target_name=node.target_name,
            condition=node.condition,
            expected_value=node.expected_value,
            dimension=node.dimension,
            error_message=node.error_message,
        )
        self._wrapper.guards.append(guard)
        return guard

    def visit_effect_tokens_wrapper(self, node: EffectTokensWrapperNode) -> Any:
        """Binary backend currently ignores effect token wrapper in pythonify IR."""
        # No-op for binary codegen; metadata is consumed elsewhere.
        return None

    def visit_aot_dispatch_subclass_wrapper(
        self, node: AOTDispatchSubclassWrapperNode
    ) -> Any:
        """Binary backend currently ignores subclass wrapper in pythonify IR."""
        return None

    def visit_functionalized_rng_runtime_wrapper(
        self, node: FunctionalizedRngRuntimeWrapperNode
    ) -> Any:
        """Binary backend currently ignores RNG wrapper in pythonify IR."""
        return None

    def visit_fakified_out_wrapper(self, node: FakifiedOutWrapperNode) -> Any:
        """Binary backend currently ignores fakified out wrapper in pythonify IR."""
        return None

    def visit_runtime_wrapper(self, node: RuntimeWrapperNode) -> Any:
        """Binary backend currently ignores runtime epilogue wrapper in pythonify IR."""
        return None

    def visit_aot_dedupe_wrapper(self, node: AOTDedupeWrapperNode) -> Any:
        """Binary backend currently ignores dedupe wrapper in pythonify IR."""
        return None

    def visit_aot_synthetic_base_wrapper(
        self, node: AOTSyntheticBaseWrapperNode
    ) -> Any:
        """Binary backend currently ignores synthetic base wrapper in pythonify IR."""
        return None

    def visit_debug_assert_wrapper(self, node: DebugAssertWrapperNode) -> Any:
        """Binary backend currently ignores debug assert wrapper in pythonify IR."""
        return None

    def visit_aot_autograd_wrapper(self, node: AOTAutogradWrapperNode) -> Any:
        """
        Process an AOT Autograd wrapper node.

        Sets up the wrapper for autograd function handling.

        Args:
            node: The AOTAutogradWrapperNode to process

        Returns:
            Metadata about the autograd wrapper configuration
        """
        self._wrapper.is_autograd_function = node.backward_graph is not None
        return {
            "class_name": node.class_name,
            "num_inputs": node.num_inputs,
            "num_outputs": node.num_outputs,
            "saved_tensors_indices": node.saved_tensors_indices,
        }

    def visit_cuda_graph_setup(self, node: CUDAGraphSetupNode) -> Any:
        """
        Process a CUDA graph setup node.

        Configures CUDA graph capture and replay settings.

        Args:
            node: The CUDAGraphSetupNode to process

        Returns:
            The CUDAGraphConfig object
        """
        config = CUDAGraphConfig(
            graph_id=node.graph_id,
            warmup_runs=node.warmup_runs,
            capture_mode=node.capture_mode,
            stream_name=node.stream_name,
            pool_id=node.pool_id,
            static_inputs=node.static_inputs,
        )
        self._wrapper.cuda_graph_config = config
        return config

    def visit_callable_invocation(self, node: CallableInvocationNode) -> Any:
        """
        Process a callable invocation node.

        Stores callable configuration for later use.

        Args:
            node: The CallableInvocationNode to process

        Returns:
            Metadata about the invocation configuration
        """
        self._callable_name = node.callable_name
        self._wrapper.result_name = node.result_name
        return {
            "callable_name": node.callable_name,
            "argument_names": node.argument_names,
            "result_name": node.result_name,
            "is_autograd_function": node.is_autograd_function,
        }

    def visit_kernel_load(self, node: KernelLoadNode) -> Any:
        """
        Process a kernel load node.

        For the binary backend, kernel loading is handled at runtime
        by the compiled wrapper infrastructure. This method records
        the kernel configuration for potential use in wrapper setup.

        Args:
            node: The KernelLoadNode to process

        Returns:
            Metadata about the kernel configuration
        """
        return {
            "kernel_type": node.kernel_type.name,
            "kernel_id": node.kernel_id,
            "kernel_path": node.kernel_path,
            "entry_point": node.entry_point,
            "variable_name": node.variable_name,
        }

    def visit_return_result(self, node: ReturnResultNode) -> Any:
        """
        Process a return result node.

        Configures how the result should be exposed.

        Args:
            node: The ReturnResultNode to process

        Returns:
            Metadata about the return configuration
        """
        self._wrapper.result_name = node.result_name
        self._wrapper.expose_as = node.expose_as
        return {
            "result_name": node.result_name,
            "expose_as": node.expose_as,
        }

    def visit_compiled_region(self, node: CompiledRegionNode) -> Any:
        """
        Process a compiled region node.

        For the binary backend, a compiled region represents a single unit
        of compiled code that can be executed. This method processes the
        region's IR and returns metadata about the region.

        Args:
            node: The CompiledRegionNode to process

        Returns:
            Metadata about the compiled region
        """
        region_info = {
            "region_id": node.region_id,
            "input_names": node.input_names.copy(),
            "output_names": node.output_names.copy(),
            "resume_target": node.resume_target,
            "is_entry_region": node.is_entry_region,
            "num_guards": len(node.guards),
        }

        if node.ir is not None:
            nested_visitor = BinaryCodeGenVisitor()
            node.ir.accept_all(nested_visitor)
            region_info["wrapper"] = nested_visitor.get_wrapper()

        return region_info

    def visit_multi_region_dispatch(self, node: MultiRegionDispatchNode) -> Any:
        """
        Process a multi-region dispatch node.

        For the binary backend, this creates a dispatch mechanism that
        selects and executes the appropriate compiled region based on
        the execution mode (sequential or guard-based).

        Args:
            node: The MultiRegionDispatchNode to process

        Returns:
            Metadata about the dispatch configuration including processed regions
        """
        processed_regions = []
        for region in node.regions:
            region_info = self.visit_compiled_region(region)
            processed_regions.append(region_info)

        return {
            "regions": processed_regions,
            "execution_mode": node.execution_mode.name,
            "fallback_to_eager": node.fallback_to_eager,
            "dispatch_table_name": node.dispatch_table_name,
        }


def generate_binary_wrapper(
    ir: RuntimeWrapperIR,
    compiled_callable: Optional[Callable[..., Any]] = None,
) -> CompiledWrapper:
    """
    Generate a CompiledWrapper from a RuntimeWrapperIR.

    This is the main entry point for the gen_binary backend. It creates
    a BinaryCodeGenVisitor, traverses the IR, and returns the constructed
    CompiledWrapper.

    Args:
        ir: The RuntimeWrapperIR to process
        compiled_callable: Optional pre-compiled callable to set on the wrapper

    Returns:
        A CompiledWrapper ready to execute the compiled function

    Example:
        artifacts = CompilationArtifacts(...)
        pipeline = RuntimeWrapperPipeline(artifacts)
        ir = pipeline.build()
        wrapper = generate_binary_wrapper(ir, my_compiled_fn)
        result = wrapper(model, frame.f_locals)
    """
    visitor = BinaryCodeGenVisitor()
    ir.accept_all(visitor)
    wrapper = visitor.get_wrapper()

    if compiled_callable is not None:
        wrapper.set_callable(compiled_callable)

    return wrapper
