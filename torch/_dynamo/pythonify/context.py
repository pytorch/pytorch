"""
Context management for pythonify feature.

This module provides thread-local context for the pythonify feature, allowing
the pythonify path to be passed through the compilation flow without modifying
every function signature in the call chain.

The context is set when entering a torch.compile region with pythonify enabled,
and is accessed after compilation to generate and write the Python file.

Model Reference and Object ID Approach
======================================
The PythonifyContext stores a reference to the model being compiled via the
`model_ref` field. This reference is used to extract parameter and buffer
tensor objects so that their object IDs can be captured for the generated code.

The object ID approach allows the generated code to retrieve tensors without
requiring the model variable in scope at execution time:

    # Generated code uses ctypes to get the tensor by its memory address:
    import ctypes
    arg2 = ctypes.cast(140234567890, ctypes.py_object).value

This is set up by calling `set_model_reference(model)` when entering the
pythonify compilation region, and accessed via `get_model_reference()` when
building CompilationArtifacts.

CRITICAL LIMITATIONS of Object ID Approach:
------------------------------------------
1. PROCESS-LOCAL ONLY: Object IDs (memory addresses) are valid only within
   the same Python process. The generated file CANNOT be used after a Python
   restart or in a different process.

2. LIFETIME DEPENDENCY: The model and its tensors must remain alive for the
   generated code to work. If the model is garbage collected, the object IDs
   become invalid.

3. NOT PERSISTABLE: The generated Python file should NOT be saved for later
   use. It is designed for immediate execution via exec() in the same session.

4. SAFETY: Accessing an invalid object ID via ctypes will crash Python or
   produce undefined behavior. The generated code should only be used
   immediately after compilation in the same process.

Why Object IDs Instead of Model Reference?
-----------------------------------------
Using object IDs instead of passing the model through exec() scope has
several advantages:
- Works regardless of variable naming in the calling context
- Guarantees the exact tensor that was compiled against is used
- Handles nested modules and complex attribute access automatically
- Simplifies the exec() invocation (no need to pass model in scope)

The trade-off is that the generated code is tied to the current process
and cannot be reused across sessions.
"""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

from .errors import (
    create_pythonify_error,
    format_code_generation_error,
    format_file_writing_error,
    PythonifyError,
    PythonifyStage,
)
from .gen_python import generate_python_code
from .ir import (
    CompiledRegionNode,
    ModelSource,
    MultiRegionDispatchNode,
    RegionExecutionMode,
    RuntimeWrapperIR,
)
from .pipeline import CompilationArtifacts, RuntimeWrapperPipeline


@dataclass
class PythonifyContext:
    """
    Thread-local context for pythonify compilation.

    Stores the pythonify path and collected compilation artifacts that will
    be used to generate the Python file after compilation completes. Wrapper
    stack ordering/metadata (wrapper_stack_order, wrapper_stack_metadata) is
    carried through untouched when present in CompilationArtifacts so callers
    can model AOTAutograd post-compile wrapper behavior without impacting
    runs that do not populate these optional fields.

    Attributes:
        pythonify_path: Path to write the generated Python file
        artifacts_list: List of CompilationArtifacts collected during compilation
        model_name: Name of the model variable in the original scope
        input_names: Names of input arguments
        inductor_outputs: List of captured Inductor compilation outputs, each
            containing source_code, graph_str, and other kernel information.
        forward_inductor_output: The forward pass Inductor output (first non-backward)
        backward_inductor_output: The backward pass Inductor output (first is_backward=True)
        model_ref: Reference to the nn.Module being compiled. This is used to
            extract parameter and buffer tensor objects so their IDs can be
            captured for ctypes-based object retrieval in generated code.
            The reference is set when the model enters the pythonify compilation
            region and cleared after code generation completes.
        model_source: Specifies where the model comes from for exec() compatibility.
            This drives how generated code accesses the model:
            - CLOSURE: Model is directly accessible as a variable (default for exec()).
                The generated code will reference the model by model_name directly.
            - F_LOCALS: Model is in frame locals (f_locals[model_name]).
            - F_GLOBALS: Model is in frame globals (f_globals[model_name]).
            Defaults to CLOSURE which is appropriate for typical exec() usage
            where the model is passed in the namespace dict.
        merged_wrapper_stack_order: Aggregated wrapper stack ordering from all
            artifacts. Keys are segment identifiers (e.g., "forward", "dispatch"),
            values are ordered lists of wrapper names. When multiple artifacts
            contribute entries for the same segment, wrapper lists are merged by
            appending new wrappers that aren't already present.
        merged_wrapper_stack_metadata: Aggregated per-wrapper metadata from all
            artifacts. Keys are wrapper names, values are wrapper-specific dicts.
            When multiple artifacts provide metadata for the same wrapper, entries
            are merged with later values taking precedence for duplicate keys.
    """

    pythonify_path: Optional[str] = None
    artifacts_list: list[CompilationArtifacts] = field(default_factory=list)
    model_name: str = "model"
    input_names: list[str] = field(default_factory=list)
    generated: bool = False
    inductor_outputs: list[dict[str, Any]] = field(default_factory=list)
    forward_inductor_output: Optional[dict[str, Any]] = None
    backward_inductor_output: Optional[dict[str, Any]] = None
    model_ref: Optional[Any] = None
    model_source: ModelSource = ModelSource.CLOSURE
    merged_wrapper_stack_order: dict[str, list[str]] = field(default_factory=dict)
    merged_wrapper_stack_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)


class _PythonifyContextManager:
    """
    Thread-local storage for pythonify context.

    Provides methods to set, get, and clear the pythonify context for the
    current thread during compilation.
    """

    def __init__(self) -> None:
        self._local = threading.local()

    def get(self) -> Optional[PythonifyContext]:
        """
        Get the current pythonify context for this thread.

        Returns:
            The current PythonifyContext, or None if not in a pythonify region
        """
        return getattr(self._local, "context", None)

    def set(self, ctx: PythonifyContext) -> None:
        """
        Set the pythonify context for this thread.

        Args:
            ctx: The PythonifyContext to set
        """
        self._local.context = ctx

    def clear(self) -> None:
        """
        Clear the pythonify context for this thread.
        """
        self._local.context = None

    def get_path(self) -> Optional[str]:
        """
        Get the pythonify path from the current context.

        Returns:
            The pythonify path, or None if not set
        """
        ctx = self.get()
        return ctx.pythonify_path if ctx else None

    def is_active(self) -> bool:
        """
        Check if pythonify is currently active.

        Returns:
            True if we're in a pythonify compilation region
        """
        ctx = self.get()
        return ctx is not None and ctx.pythonify_path is not None


_pythonify_context = _PythonifyContextManager()


def get_pythonify_context() -> Optional[PythonifyContext]:
    """
    Get the current pythonify context.

    Returns:
        The current PythonifyContext, or None if not in a pythonify region
    """
    return _pythonify_context.get()


def get_pythonify_path() -> Optional[str]:
    """
    Get the current pythonify path.

    Returns:
        The pythonify path, or None if not set
    """
    return _pythonify_context.get_path()


def is_pythonify_active() -> bool:
    """
    Check if pythonify is currently active.

    Returns:
        True if we're in a pythonify compilation region
    """
    return _pythonify_context.is_active()


def set_model_reference(
    model: Any,
    model_source: ModelSource = ModelSource.CLOSURE,
) -> None:
    """
    Set the model reference and source in the current pythonify context.

    This function should be called when the model enters the pythonify
    compilation region, typically from the OptimizeContext.__call__ or
    similar entry point. The model reference is used for the golden path
    reconstruction where parameters and buffers are accessed via module
    attribute traversal (e.g., model.layer.weight) instead of object IDs.

    The model_source specifies how the generated code should access the model
    at exec() time, enabling process-portable, live-access code that doesn't
    depend on stale object IDs.

    Args:
        model: The nn.Module or callable being compiled. If it's an nn.Module,
            its parameters and buffers will be accessible via attribute traversal.
        model_source: Specifies where the model comes from for exec() compatibility.
            - CLOSURE (default): Model is directly accessible as a variable.
                Use when passing the model in the exec() namespace dict.
            - F_LOCALS: Model is in frame locals (f_locals[model_name]).
            - F_GLOBALS: Model is in frame globals (f_globals[model_name]).
    """
    ctx = _pythonify_context.get()
    if ctx is not None:
        ctx.model_ref = model
        ctx.model_source = model_source


def get_model_reference() -> Optional[Any]:
    """
    Get the model reference from the current pythonify context.

    Returns:
        The model reference, or None if not set or pythonify is not active
    """
    ctx = _pythonify_context.get()
    if ctx is not None:
        return ctx.model_ref
    return None


def get_model_source() -> ModelSource:
    """
    Get the model source from the current pythonify context.

    The model source specifies how the generated code should access the model
    at exec() time. This is used by the pipeline to set the model_source field
    on CompilationArtifacts, which in turn drives codegen to emit the correct
    model access pattern (e.g., direct variable, f_locals["model"], etc.).

    Returns:
        The ModelSource from the current context, or CLOSURE (default) if
        pythonify is not active or no explicit source was set.
    """
    ctx = _pythonify_context.get()
    if ctx is not None:
        return ctx.model_source
    return ModelSource.CLOSURE


def get_merged_wrapper_stack_order() -> dict[str, list[str]]:
    """
    Get the merged wrapper stack ordering from the current pythonify context.

    Returns the aggregated wrapper stack order accumulated from all
    CompilationArtifacts added to the context. Each segment maps to an
    ordered list of wrapper names that should be applied in that segment.

    Returns:
        The merged wrapper stack order dict, or empty dict if not active
    """
    ctx = _pythonify_context.get()
    if ctx is not None:
        return ctx.merged_wrapper_stack_order
    return {}


def get_merged_wrapper_stack_metadata() -> dict[str, dict[str, Any]]:
    """
    Get the merged wrapper metadata from the current pythonify context.

    Returns the aggregated per-wrapper metadata accumulated from all
    CompilationArtifacts added to the context. Each wrapper name maps to
    a dict of metadata relevant for that wrapper's code generation.

    Returns:
        The merged wrapper metadata dict, or empty dict if not active
    """
    ctx = _pythonify_context.get()
    if ctx is not None:
        return ctx.merged_wrapper_stack_metadata
    return {}


@contextlib.contextmanager
def pythonify_context(pythonify_path: Optional[str]):
    """
    Context manager for pythonify compilation regions.

    Sets up the pythonify context when entering and clears it when exiting.
    After compilation, generates and writes the Python file if pythonify is set.

    Args:
        pythonify_path: Path to write the generated Python file, or None for
            normal (non-pythonify) compilation

    Yields:
        The PythonifyContext if pythonify_path is set, None otherwise
    """
    if pythonify_path is None:
        yield None
        return

    ctx = PythonifyContext(pythonify_path=pythonify_path)
    _pythonify_context.set(ctx)
    try:
        yield ctx
    finally:
        _pythonify_context.clear()


def _merge_wrapper_stack_order(
    existing: dict[str, list[str]],
    incoming: dict[str, list[str]],
) -> None:
    """
    Merge incoming wrapper stack order into existing order in-place.

    For each segment (key) in the incoming order, appends any wrapper names
    that are not already present in the existing list for that segment. This
    preserves the order of wrappers while avoiding duplicates.

    The merge strategy ensures that:
    - Existing wrappers in each segment retain their positions
    - New wrappers from incoming artifacts are appended at the end
    - Duplicate wrappers (same name in same segment) are not added twice

    Args:
        existing: The current wrapper stack order to merge into (modified in-place)
        incoming: The new wrapper stack order from an artifact
    """
    for segment, wrappers in incoming.items():
        if segment not in existing:
            existing[segment] = []
        existing_set = set(existing[segment])
        for wrapper in wrappers:
            if wrapper not in existing_set:
                existing[segment].append(wrapper)
                existing_set.add(wrapper)


def _merge_wrapper_stack_metadata(
    existing: dict[str, dict[str, Any]],
    incoming: dict[str, dict[str, Any]],
) -> None:
    """
    Merge incoming wrapper metadata into existing metadata in-place.

    For each wrapper name (key) in the incoming metadata, merges the metadata
    dict with any existing metadata for that wrapper. Later values take
    precedence for duplicate keys within a wrapper's metadata dict.

    The merge strategy ensures that:
    - Existing wrapper metadata is preserved when no new data is provided
    - New wrappers get their metadata added to the merged result
    - For overlapping keys within a wrapper's metadata, incoming values win

    Args:
        existing: The current wrapper metadata to merge into (modified in-place)
        incoming: The new wrapper metadata from an artifact
    """
    for wrapper_name, metadata in incoming.items():
        if wrapper_name not in existing:
            existing[wrapper_name] = {}
        existing[wrapper_name].update(metadata)


def validate_wrapper_metadata_merge(
    merged_order: dict[str, list[str]],
    merged_metadata: dict[str, dict[str, Any]],
    prior_order: dict[str, list[str]],
    prior_metadata: dict[str, dict[str, Any]],
) -> bool:
    """
    Validate that wrapper metadata merge preserved prior entries.

    This helper confirms that all wrappers and segments from the prior state
    are still present in the merged result. Use this after calling
    add_compilation_artifacts to verify that new artifacts did not clobber
    existing wrapper stack information.

    The validation checks:
    - All segments from prior_order exist in merged_order
    - All wrappers from each prior segment are still present (order preserved)
    - All wrapper names from prior_metadata exist in merged_metadata
    - All metadata keys from prior wrappers are still present (values may be
      overwritten by incoming data, which is expected behavior)

    Args:
        merged_order: The wrapper stack order after merge
        merged_metadata: The wrapper metadata after merge
        prior_order: The wrapper stack order before new artifact was added
        prior_metadata: The wrapper metadata before new artifact was added

    Returns:
        True if all prior entries are preserved, False otherwise

    Example:
        >>> prior_order = {"forward": ["WrapperA"]}
        >>> prior_meta = {"WrapperA": {"key1": "val1"}}
        >>> # After adding new artifact...
        >>> merged_order = {"forward": ["WrapperA", "WrapperB"]}
        >>> merged_meta = {"WrapperA": {"key1": "val1"}, "WrapperB": {"key2": "val2"}}
        >>> validate_wrapper_metadata_merge(merged_order, merged_meta, prior_order, prior_meta)
        True
    """
    for segment, prior_wrappers in prior_order.items():
        if segment not in merged_order:
            return False
        merged_wrappers = merged_order[segment]
        for wrapper in prior_wrappers:
            if wrapper not in merged_wrappers:
                return False
            prior_idx = prior_wrappers.index(wrapper)
            merged_idx = merged_wrappers.index(wrapper)
            if prior_idx != merged_idx:
                return False

    for wrapper_name, prior_meta in prior_metadata.items():
        if wrapper_name not in merged_metadata:
            return False
        merged_meta = merged_metadata[wrapper_name]
        for key in prior_meta:
            if key not in merged_meta:
                return False

    return True


def add_compilation_artifacts(artifacts: CompilationArtifacts) -> None:
    """
    Add compilation artifacts to the current pythonify context.

    Called during compilation to store artifacts that will be used to
    generate the Python file. Wrapper stack ordering/metadata stored on
    CompilationArtifacts is merged into context-level aggregated fields
    (merged_wrapper_stack_order, merged_wrapper_stack_metadata) without
    loss or overwrite. When wrapper metadata is absent on an artifact,
    behavior is unchanged and defaults are preserved.

    The merge strategy for wrapper_stack_order appends new wrappers to each
    segment while preserving existing order and avoiding duplicates. For
    wrapper_stack_metadata, per-wrapper dicts are merged with later values
    taking precedence for duplicate keys.

    Multiple calls append to the artifacts list in order without mutating
    prior entries, so wrapper_stack_order and wrapper_stack_metadata on
    individual artifacts remain intact.

    Args:
        artifacts: The compilation artifacts to add
    """
    ctx = _pythonify_context.get()
    if ctx is not None:
        ctx.artifacts_list.append(artifacts)
        _merge_wrapper_stack_order(
            ctx.merged_wrapper_stack_order,
            artifacts.wrapper_stack_order,
        )
        _merge_wrapper_stack_metadata(
            ctx.merged_wrapper_stack_metadata,
            artifacts.wrapper_stack_metadata,
        )


def add_inductor_output(inductor_output: dict[str, Any]) -> None:
    """
    Add captured Inductor compilation output to the current pythonify context.

    Called after Inductor compilation to store the generated source code and
    kernel information that will be included in the pythonify output.

    For backward pass support, this function tracks forward and backward
    Inductor outputs separately. The first non-backward output is stored as
    the forward pass, and the first is_backward=True output is stored as
    the backward pass.

    The inductor_output dictionary should contain:
        - source_code: The Python/Triton source code generated by Inductor
        - graph_str: Human-readable graph representation
        - cache_key: Unique identifier for this compiled graph
        - device_types: Set of device types used
        - constants: Dictionary of constant values
        - is_backward: True if this is backward pass compilation (optional)

    Args:
        inductor_output: Dictionary containing Inductor compilation output
    """
    ctx = _pythonify_context.get()
    if ctx is not None:
        ctx.inductor_outputs.append(inductor_output)

        is_backward = inductor_output.get("is_backward", False)
        if is_backward:
            if ctx.backward_inductor_output is None:
                ctx.backward_inductor_output = inductor_output
        else:
            if ctx.forward_inductor_output is None:
                ctx.forward_inductor_output = inductor_output


def get_latest_inductor_output() -> Optional[dict[str, Any]]:
    """
    Get the most recently captured Inductor output.

    Returns:
        The most recent Inductor output dictionary, or None if none captured
    """
    ctx = _pythonify_context.get()
    if ctx is not None and ctx.inductor_outputs:
        return ctx.inductor_outputs[-1]
    return None


def get_forward_inductor_output() -> Optional[dict[str, Any]]:
    """
    Get the forward pass Inductor output.

    Returns:
        The forward pass Inductor output dictionary, or None if none captured
    """
    ctx = _pythonify_context.get()
    if ctx is not None:
        return ctx.forward_inductor_output
    return None


def get_backward_inductor_output() -> Optional[dict[str, Any]]:
    """
    Get the backward pass Inductor output.

    Returns:
        The backward pass Inductor output dictionary, or None if none captured
    """
    ctx = _pythonify_context.get()
    if ctx is not None:
        return ctx.backward_inductor_output
    return None


def _merge_wrapper_metadata_from_inductor_output(
    artifact: CompilationArtifacts,
    inductor_output: dict[str, Any],
) -> None:
    """
    Merge wrapper metadata from an Inductor output dict into an artifact.

    Inductor outputs may contain wrapper_stack_order and wrapper_stack_metadata
    fields populated during compilation. This function merges those into the
    artifact's existing wrapper metadata without losing prior data.

    The merge strategy matches add_compilation_artifacts:
    - For wrapper_stack_order: appends new wrappers to each segment while
      avoiding duplicates and preserving existing order
    - For wrapper_stack_metadata: merges per-wrapper dicts with incoming
      values taking precedence for duplicate keys

    If the inductor output contains no wrapper metadata fields, this function
    is a no-op and the artifact's existing wrapper metadata remains untouched.

    Args:
        artifact: The CompilationArtifacts to merge wrapper metadata into
        inductor_output: The Inductor output dict that may contain wrapper metadata
    """
    incoming_order = inductor_output.get("wrapper_stack_order", {})
    incoming_metadata = inductor_output.get("wrapper_stack_metadata", {})

    if incoming_order:
        _merge_wrapper_stack_order(artifact.wrapper_stack_order, incoming_order)

    if incoming_metadata:
        _merge_wrapper_stack_metadata(artifact.wrapper_stack_metadata, incoming_metadata)


def _merge_inductor_outputs_into_artifacts(ctx: PythonifyContext) -> None:
    """
    Merge captured Inductor outputs into compilation artifacts.

    Inductor source code is captured asynchronously after artifacts are created.
    This function merges the captured outputs back into the artifacts so that
    the code generator has access to both forward and backward kernels.

    For single-artifact compilations (no graph breaks), the forward and backward
    Inductor outputs are merged into the first artifact.

    For multi-artifact compilations (with graph breaks), each artifact is matched
    with its corresponding Inductor output based on order.

    Wrapper Metadata Preservation:
    This function only updates inductor-related fields (inductor_source_code,
    inductor_graph_str, backward_inductor_source_code, backward_inductor_graph_str)
    on artifacts. Existing wrapper metadata on artifacts (wrapper_stack_order,
    wrapper_stack_metadata) is explicitly preserved and never overwritten by this
    function. If inductor outputs contain wrapper metadata, it is merged into the
    artifact's existing wrapper metadata using the same merge strategy as
    add_compilation_artifacts: new wrappers are appended to segments without
    duplicates, and per-wrapper metadata dicts are merged with incoming values
    taking precedence for duplicate keys.

    Args:
        ctx: The PythonifyContext containing artifacts and Inductor outputs
    """
    if not ctx.artifacts_list:
        return

    forward_output = ctx.forward_inductor_output
    backward_output = ctx.backward_inductor_output

    if len(ctx.artifacts_list) == 1:
        artifact = ctx.artifacts_list[0]

        if forward_output is not None and artifact.inductor_source_code is None:
            artifact.inductor_source_code = forward_output.get("source_code")
            artifact.inductor_graph_str = forward_output.get("graph_str")
            _merge_wrapper_metadata_from_inductor_output(artifact, forward_output)

        if backward_output is not None and artifact.backward_inductor_source_code is None:
            artifact.backward_inductor_source_code = backward_output.get("source_code")
            artifact.backward_inductor_graph_str = backward_output.get("graph_str")
            _merge_wrapper_metadata_from_inductor_output(artifact, backward_output)
    else:
        forward_outputs = [
            out for out in ctx.inductor_outputs
            if not out.get("is_backward", False)
        ]
        for i, artifact in enumerate(ctx.artifacts_list):
            if i < len(forward_outputs) and artifact.inductor_source_code is None:
                out = forward_outputs[i]
                artifact.inductor_source_code = out.get("source_code")
                artifact.inductor_graph_str = out.get("graph_str")
                _merge_wrapper_metadata_from_inductor_output(artifact, out)


def _uses_object_ids(artifacts_list: list[CompilationArtifacts]) -> bool:
    """
    Check if any artifacts use object IDs for parameter/buffer access.

    Object IDs are used when parameter_tensors or buffer_tensors are populated
    in the CompilationArtifacts. This indicates the generated code will use
    ctypes to retrieve objects by memory address instead of model attribute access.

    Args:
        artifacts_list: List of compilation artifacts to check

    Returns:
        True if any artifact uses object IDs, False otherwise
    """
    for artifact in artifacts_list:
        if artifact.parameter_tensors or artifact.buffer_tensors:
            return True
    return False


def _generate_object_id_warning() -> list[str]:
    """
    Generate warning lines about object ID limitations.

    This warning is included in the generated pythonify file when object IDs
    are used for parameter/buffer retrieval. It explains the critical limitations
    of the object ID approach to prevent misuse.

    Returns:
        List of warning lines to include in the generated file header
    """
    return [
        "# " + "=" * 76,
        "# WARNING: PROCESS-LOCAL FILE - DO NOT SAVE OR REUSE",
        "# " + "=" * 76,
        "#",
        "# This file uses object IDs (memory addresses) to access model parameters",
        "# and buffers. These IDs are ONLY valid within the current Python process.",
        "#",
        "# CRITICAL LIMITATIONS:",
        "#   1. NOT PERSISTABLE: Do NOT save this file for later use. The object IDs",
        "#      become invalid after the Python process exits or the model is deleted.",
        "#",
        "#   2. SAME-PROCESS ONLY: This file must be exec()'d in the same Python",
        "#      process where torch.compile() was called. Cross-process use will crash.",
        "#",
        "#   3. KEEP MODEL ALIVE: The original model must remain in memory. If the",
        "#      model is garbage collected, the object IDs become dangling references.",
        "#",
        "#   4. CRASH RISK: Invalid object IDs will cause Python to crash or produce",
        "#      undefined behavior. Never manually edit the object IDs in this file.",
        "#",
        "# This file is designed for IMMEDIATE execution via exec() in the same session.",
        "# For persistent/serializable compiled models, use torch.export() instead.",
        "# " + "=" * 76,
        "",
    ]


def generate_pythonify_output() -> Optional[str]:
    """
    Generate the pythonify output file from collected artifacts.

    Builds the IR from all collected compilation artifacts and generates
    the Python source code. Writes the code to the pythonify path.

    When multiple compilation artifacts are collected (due to graph breaks),
    generates a multi-region dispatch structure that properly chains the
    compiled regions together.

    Before generating code, this function merges any captured Inductor outputs
    (forward and backward) into the artifacts. This ensures that Inductor
    source code captured after artifact creation is included in the output.

    Returns:
        The generated Python code as a string, or None if pythonify is not active

    Raises:
        PythonifyError: If IR construction, code generation, or file writing fails
    """
    ctx = _pythonify_context.get()
    if ctx is None or ctx.pythonify_path is None:
        return None

    if ctx.generated:
        return None

    if not ctx.artifacts_list:
        return None

    _merge_inductor_outputs_into_artifacts(ctx)

    all_code_parts: list[str] = []
    all_code_parts.append('"""')
    all_code_parts.append("Generated by torch.compile(pythonify=...)")
    all_code_parts.append("")
    all_code_parts.append(
        "This file contains explicit Python code representing all runtime machinery"
    )
    all_code_parts.append(
        "that is normally embedded inside torch.compile. Execute with exec():"
    )
    all_code_parts.append("")

    uses_object_ids = _uses_object_ids(ctx.artifacts_list)
    model_name = ctx.model_name if ctx.model_name else "model"

    if uses_object_ids:
        all_code_parts.append("OBJECT ID FALLBACK MODE:")
        all_code_parts.append(
            "This file uses ctypes-based object IDs for parameter access."
        )
        all_code_parts.append(
            "Execute in the same process immediately after compilation:"
        )
        all_code_parts.append("")
        all_code_parts.append("    with open('/path/to/this/file.py') as f:")
        all_code_parts.append("        frame = inspect.currentframe()")
        all_code_parts.append("        exec(f.read(), frame.f_globals, frame.f_locals)")
    else:
        all_code_parts.append("GOLDEN PATH (RECOMMENDED):")
        all_code_parts.append(
            "This file uses module-based attribute access for parameters/buffers."
        )
        all_code_parts.append(
            f"Pass your model as '{model_name}' in the exec() namespace:"
        )
        all_code_parts.append("")
        all_code_parts.append("    with open('/path/to/this/file.py') as f:")
        all_code_parts.append("        code = f.read()")
        all_code_parts.append("    namespace = {")
        all_code_parts.append(f'        "{model_name}": your_model,')
        all_code_parts.append('        "compiled_fn": compiled_callable,')
        all_code_parts.append('        "f_locals": {"x": input_tensor},')
        all_code_parts.append("    }")
        all_code_parts.append("    exec(code, namespace)")
        all_code_parts.append('    result = namespace["y"]')
        all_code_parts.append("")
        all_code_parts.append("ADVANTAGES of Golden Path:")
        all_code_parts.append(
            "  - PROCESS-PORTABLE: Works across different Python processes"
        )
        all_code_parts.append(
            "  - LIVE ACCESS: Always sees current parameter values (no stale IDs)"
        )
        all_code_parts.append(
            "  - SAFER: No ctypes/memory address access, no crash risk"
        )
        all_code_parts.append(
            "  - PERSISTABLE: Can save this file and reuse it later"
        )

    all_code_parts.append('"""')
    all_code_parts.append("")

    # Add warning about object ID limitations if applicable
    if uses_object_ids:
        all_code_parts.extend(_generate_object_id_warning())

    all_code_parts.append("import torch")
    all_code_parts.append("")
    all_code_parts.append("# Bind f_locals to the local namespace for exec() compatibility.")
    all_code_parts.append("# When exec(code, globals, locals) is called, the locals dict is passed")
    all_code_parts.append("# as the third argument, but there's no variable named 'f_locals' in scope.")
    all_code_parts.append("# By binding f_locals = locals(), we make the locals dict accessible by name,")
    all_code_parts.append("# which allows the generated code to use f_locals['x'] and f_locals['y'] = y.")
    all_code_parts.append("f_locals = locals()")
    all_code_parts.append("")

    if len(ctx.artifacts_list) == 1:
        try:
            pipeline = RuntimeWrapperPipeline(ctx.artifacts_list[0])
            ir = pipeline.build()
        except PythonifyError:
            raise
        except Exception as e:
            raise create_pythonify_error(
                PythonifyStage.IR_CONSTRUCTION,
                e,
                context={
                    "artifact_index": 1,
                    "total_artifacts": 1,
                    "model_name": ctx.artifacts_list[0].model_name,
                    "num_inputs": len(ctx.artifacts_list[0].input_names),
                    "num_parameters": len(ctx.artifacts_list[0].parameter_names),
                },
            ) from e

        try:
            code = generate_python_code(ir)
        except PythonifyError:
            raise
        except Exception as e:
            raise create_pythonify_error(
                PythonifyStage.PYTHON_CODE_GENERATION,
                e,
                context={
                    "artifact_index": 1,
                    "total_artifacts": 1,
                    "num_ir_nodes": len(ir.nodes) if ir else 0,
                },
            ) from e

        all_code_parts.append(code)
        all_code_parts.append("")

    else:
        all_code_parts.append(f"# Multi-region execution with {len(ctx.artifacts_list)} compiled regions")
        all_code_parts.append(f"# This model has graph breaks that result in multiple compiled regions.")
        all_code_parts.append("")

        try:
            multi_region_code = _generate_multi_region_code(ctx.artifacts_list)
            all_code_parts.append(multi_region_code)
        except PythonifyError:
            raise
        except Exception as e:
            raise create_pythonify_error(
                PythonifyStage.PYTHON_CODE_GENERATION,
                e,
                context={
                    "num_regions": len(ctx.artifacts_list),
                },
            ) from e

    final_code = "\n".join(all_code_parts)

    try:
        with open(ctx.pythonify_path, "w") as f:
            f.write(final_code)
    except (PermissionError, FileNotFoundError, OSError) as e:
        raise PythonifyError(
            message=format_file_writing_error(ctx.pythonify_path, e),
            stage=PythonifyStage.FILE_WRITING,
            original_exception=e,
            context={"pythonify_path": ctx.pythonify_path},
            remedy=(
                "Ensure the output directory exists and is writable. "
                f"Path: {ctx.pythonify_path}"
            ),
        ) from e

    ctx.generated = True

    return final_code


def _generate_multi_region_code(artifacts_list: list[CompilationArtifacts]) -> str:
    """
    Generate Python code for multiple compiled regions.

    Creates a multi-region dispatch structure that chains the compiled regions
    together in sequential execution order (for graph breaks).

    Args:
        artifacts_list: List of CompilationArtifacts for each region

    Returns:
        Generated Python code for multi-region execution

    Raises:
        PythonifyError: If IR construction or code generation fails for any region
    """
    regions = []

    for i, artifacts in enumerate(artifacts_list):
        region_id = f"region_{i}"

        try:
            pipeline = RuntimeWrapperPipeline(artifacts)
            ir = pipeline.build()
        except PythonifyError:
            raise
        except Exception as e:
            raise create_pythonify_error(
                PythonifyStage.IR_CONSTRUCTION,
                e,
                context={
                    "artifact_index": i + 1,
                    "total_artifacts": len(artifacts_list),
                    "region_id": region_id,
                    "model_name": artifacts.model_name,
                    "num_inputs": len(artifacts.input_names),
                    "num_parameters": len(artifacts.parameter_names),
                },
            ) from e

        is_last = (i == len(artifacts_list) - 1)
        resume_target = None if is_last else f"region_{i + 1}"

        region = CompiledRegionNode(
            region_id=region_id,
            guards=[],
            ir=ir,
            input_names=artifacts.input_names.copy(),
            output_names=["result"],
            resume_target=resume_target,
            is_entry_region=(i == 0),
            metadata={
                "model_name": artifacts.model_name,
                "artifact_index": i,
            },
        )
        regions.append(region)

    dispatch_node = MultiRegionDispatchNode(
        regions=regions,
        execution_mode=RegionExecutionMode.SEQUENTIAL,
        fallback_to_eager=False,
        dispatch_table_name="_execute_regions",
        metadata={
            "num_regions": len(regions),
            "has_graph_breaks": len(regions) > 1,
        },
    )

    from .gen_python import PythonCodeGenVisitor

    visitor = PythonCodeGenVisitor()
    visitor.visit_multi_region_dispatch(dispatch_node)

    return visitor.get_code()
