"""
OpTracer: A __torch_dispatch__ based tracer for forward and backward operations.

This tracer uses TorchDispatchMode to intercept all tensor operations and record
them, working with FakeTensors to trace without actual computation.
"""

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode


@dataclass
class TensorPlaceholder:
    """Marker for tensor positions in traced args structure."""

    tensor_id: int


@dataclass
class TracedOp:
    """Represents a single traced operation."""

    op: str  # Full operator name (e.g., "aten::linear.default")
    op_name: str  # Short name
    input_tensor_ids: list[int]
    output_tensor_ids: list[int]
    input_shapes: list[tuple[int, ...]]
    output_shapes: list[tuple[int, ...]]
    input_dtypes: list[torch.dtype]
    output_dtypes: list[torch.dtype]
    is_backward: bool = False
    # For replay: store the actual function and non-tensor args
    func: Any = None  # The actual aten function
    args_structure: Any = None  # Structure of args with tensor positions marked
    kwargs_structure: Any = None  # Structure of kwargs

    def __repr__(self) -> str:
        direction = "BWD" if self.is_backward else "FWD"
        shapes_in = [list(s) for s in self.input_shapes]
        shapes_out = [list(s) for s in self.output_shapes]
        return f"[{direction}] {self.op_name}: {shapes_in} -> {shapes_out}"


class OpTracer(TorchDispatchMode):
    """
    A TorchDispatchMode that traces all tensor operations.

    This tracer sits on top of FakeTensorMode and records all aten operations
    that flow through __torch_dispatch__, including both forward and backward ops.

    Usage:
        fake_mode = FakeTensorMode()
        with fake_mode:
            fake_inputs = [fake_mode.from_tensor(t) for t in inputs]
            tracer = OpTracer()
            with tracer:
                output = model(*fake_inputs)
                output.backward(torch.ones_like(output))
            print(tracer.get_trace())
    """

    def __init__(self, filter_namespaces: Optional[list[str]] = None) -> None:
        """
        Args:
            filter_namespaces: If provided, only trace ops from these namespaces
                             (e.g., ["aten"]). If None, trace all ops.
        """
        super().__init__()
        self.trace: list[TracedOp] = []
        self.tensor_id_map: dict[int, int] = {}
        self.next_id: int = 0
        self.filter_namespaces = filter_namespaces
        self._in_backward = False
        self._backward_tensors: set[int] = set()

    def _get_tensor_id(self, t: torch.Tensor) -> int:
        """Assign a stable ID to a tensor for tracking data flow."""
        # Use id() for tensor identity - data_ptr is deprecated for FakeTensors
        key = id(t)

        if key not in self.tensor_id_map:
            self.tensor_id_map[key] = self.next_id
            self.next_id += 1
        return self.tensor_id_map[key]

    def _extract_tensor_info(
        self, args: tuple[Any, ...]
    ) -> tuple[list[int], list[tuple[int, ...]], list[torch.dtype]]:
        """Extract tensor IDs, shapes, and dtypes from args."""
        tensor_ids = []
        shapes = []
        dtypes = []
        for arg in pytree.tree_leaves(args):
            if isinstance(arg, torch.Tensor):
                tensor_ids.append(self._get_tensor_id(arg))
                shapes.append(tuple(arg.shape))
                dtypes.append(arg.dtype)
        return tensor_ids, shapes, dtypes

    def _create_args_structure(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[Any, Any]:
        """
        Create a structure that marks tensor positions for later replay.

        Returns args and kwargs with tensors replaced by TensorPlaceholder.
        """

        def mark_tensor(x: Any) -> Any:
            if isinstance(x, torch.Tensor):
                tid = self._get_tensor_id(x)
                return TensorPlaceholder(tensor_id=tid)
            return x

        args_structure = pytree.tree_map(mark_tensor, args)
        kwargs_structure = pytree.tree_map(mark_tensor, kwargs)
        return args_structure, kwargs_structure

    def _is_backward_op(self, func: Any, args: tuple[Any, ...]) -> bool:
        """
        Determine if this operation is part of the backward pass.

        Backward ops are detected by checking if we're inside a backward context
        or if the inputs include gradient tensors.
        """
        if self._in_backward:
            return True

        # Check if any input tensor has grad_fn (indicates it's in autograd graph)
        for arg in pytree.tree_leaves(args):
            if isinstance(arg, torch.Tensor):
                if id(arg) in self._backward_tensors:
                    return True
        return False

    def __torch_dispatch__(
        self,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: Optional[dict[str, Any]] = None,
    ) -> Any:
        kwargs = kwargs or {}

        # Filter by namespace if specified
        if self.filter_namespaces is not None:
            namespace = getattr(func, "namespace", None)
            if namespace not in self.filter_namespaces:
                return func(*args, **kwargs)

        # Record input tensor info
        input_ids, input_shapes, input_dtypes = self._extract_tensor_info(args)

        # Capture args structure for replay
        args_structure, kwargs_structure = self._create_args_structure(args, kwargs)

        # Check if this is a backward op
        is_backward = self._is_backward_op(func, args)

        # Execute the operation
        result = func(*args, **kwargs)

        # Record output tensor info
        output_ids, output_shapes, output_dtypes = self._extract_tensor_info(
            (result,) if not isinstance(result, tuple) else result
        )

        # Mark output tensors as backward tensors if this is a backward op
        if is_backward:
            for arg in pytree.tree_leaves(
                (result,) if not isinstance(result, tuple) else result
            ):
                if isinstance(arg, torch.Tensor):
                    self._backward_tensors.add(id(arg))

        # Get operation name
        if hasattr(func, "name"):
            op_name = func.name()
        elif hasattr(func, "__name__"):
            op_name = func.__name__
        else:
            op_name = str(func)

        # Record the traced op
        traced_op = TracedOp(
            op=str(func),
            op_name=op_name,
            input_tensor_ids=input_ids,
            output_tensor_ids=output_ids,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            input_dtypes=input_dtypes,
            output_dtypes=output_dtypes,
            is_backward=is_backward,
            func=func,
            args_structure=args_structure,
            kwargs_structure=kwargs_structure,
        )
        self.trace.append(traced_op)

        return result

    @contextmanager
    def mark_backward(self) -> Generator[None, None, None]:
        """Context manager to mark operations as backward ops."""
        prev = self._in_backward
        self._in_backward = True
        try:
            yield
        finally:
            self._in_backward = prev

    def get_trace(self) -> list[TracedOp]:
        """Return the list of traced operations."""
        return self.trace

    def get_forward_ops(self) -> list[TracedOp]:
        """Return only forward operations."""
        return [op for op in self.trace if not op.is_backward]

    def get_backward_ops(self) -> list[TracedOp]:
        """Return only backward operations."""
        return [op for op in self.trace if op.is_backward]

    def summary(self) -> str:
        """Return a summary of the traced operations."""
        lines = []
        lines.append(f"Total ops: {len(self.trace)}")
        lines.append(f"Forward ops: {len(self.get_forward_ops())}")
        lines.append(f"Backward ops: {len(self.get_backward_ops())}")
        lines.append("")
        lines.append("Operations:")
        for i, op in enumerate(self.trace):
            lines.append(f"  {i}: {op}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear the trace."""
        self.trace.clear()
        self.tensor_id_map.clear()
        self.next_id = 0
        self._backward_tensors.clear()


def _convert_module_to_fake(
    module: torch.nn.Module,
    fake_mode: "torch._subclasses.fake_tensor.FakeTensorMode",
) -> torch.nn.Module:
    """
    Convert all parameters and buffers of a module to FakeTensors.

    This creates a shallow copy of the module with fake parameters/buffers.
    """
    import copy

    # Create a shallow copy to avoid modifying the original
    fake_module = copy.copy(module)

    # Convert parameters
    param_dict = {}
    for name, param in module.named_parameters(recurse=False):
        fake_param = fake_mode.from_tensor(param)
        fake_param.requires_grad_(param.requires_grad)
        param_dict[name] = torch.nn.Parameter(
            fake_param, requires_grad=param.requires_grad
        )

    for name, param in param_dict.items():
        setattr(fake_module, name, param)

    # Convert buffers
    for name, buf in module.named_buffers(recurse=False):
        if buf is not None:
            fake_buf = fake_mode.from_tensor(buf)
            setattr(fake_module, name, fake_buf)

    # Recursively convert child modules
    for name, child in module.named_children():
        fake_child = _convert_module_to_fake(child, fake_mode)
        setattr(fake_module, name, fake_child)

    return fake_module


def trace_model(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    backward: bool = True,
    filter_namespaces: Optional[list[str]] = None,
) -> OpTracer:
    """
    Trace a model's forward (and optionally backward) operations using FakeTensors.

    Args:
        model: The PyTorch model to trace
        inputs: Input tensors (will be converted to FakeTensors)
        backward: Whether to also trace the backward pass
        filter_namespaces: If provided, only trace ops from these namespaces

    Returns:
        OpTracer with the recorded trace
    """
    from torch._subclasses.fake_tensor import FakeTensorMode

    fake_mode = FakeTensorMode(allow_non_fake_inputs=False)

    with fake_mode:
        # Convert model parameters and buffers to fake tensors
        fake_model = _convert_module_to_fake(model, fake_mode)

        # Convert inputs to fake tensors
        fake_inputs = []
        for t in inputs:
            fake_t = fake_mode.from_tensor(t)
            if backward and t.requires_grad:
                fake_t.requires_grad_(True)
            fake_inputs.append(fake_t)

        # Create tracer
        tracer = OpTracer(filter_namespaces=filter_namespaces)

        with tracer:
            # Forward pass
            output = fake_model(*fake_inputs)

            if backward:
                # Handle multiple outputs
                if isinstance(output, tuple):
                    # Sum all outputs for backward
                    tensors = [o.sum() for o in output if isinstance(o, torch.Tensor)]
                    if tensors:
                        loss: torch.Tensor = tensors[0]
                        for t in tensors[1:]:
                            loss = loss + t
                    else:
                        raise ValueError("No tensor outputs found for backward pass")
                else:
                    loss = output.sum()

                # Mark backward context and run backward
                with tracer.mark_backward():
                    loss.backward()

    return tracer


def execute_trace(
    trace: list[TracedOp],
    initial_tensors: dict[int, torch.Tensor],
    device: Optional[torch.device] = None,
) -> dict[int, torch.Tensor]:
    """
    Execute a trace of operations with real tensors.

    Args:
        trace: List of TracedOp from OpTracer
        initial_tensors: Dict mapping tensor IDs to initial real tensors
        device: Device to create tensors on (default: cpu)

    Returns:
        Dict mapping tensor IDs to result tensors
    """
    device = device or torch.device("cpu")
    tensor_map: dict[int, torch.Tensor] = dict(initial_tensors)

    def substitute_tensors(structure: Any) -> Any:
        """Replace TensorPlaceholder with actual tensors from tensor_map."""

        def substitute(x: Any) -> Any:
            if isinstance(x, TensorPlaceholder):
                tensor_id = x.tensor_id
                if tensor_id not in tensor_map:
                    raise KeyError(
                        f"Tensor ID {tensor_id} not found in tensor_map. "
                        f"Available IDs: {list(tensor_map.keys())}"
                    )
                return tensor_map[tensor_id]
            return x

        return pytree.tree_map(substitute, structure)

    for i, op in enumerate(trace):
        if op.func is None:
            raise ValueError(
                f"Op {i} ({op.op_name}) has no func stored. "
                "Make sure tracing captured the function."
            )

        # Substitute tensor placeholders with actual tensors
        args = substitute_tensors(op.args_structure)
        kwargs = substitute_tensors(op.kwargs_structure)

        # Execute the operation
        result = op.func(*args, **kwargs)

        # Store output tensors in the map
        if isinstance(result, tuple):
            tensor_map.update(
                {
                    tid: tensor
                    for tid, tensor in zip(op.output_tensor_ids, result)
                    if isinstance(tensor, torch.Tensor)
                }
            )
        elif isinstance(result, torch.Tensor):
            if op.output_tensor_ids:
                tensor_map[op.output_tensor_ids[0]] = result

    return tensor_map


def create_random_inputs(
    trace: list[TracedOp],
    device: Optional[torch.device] = None,
) -> dict[int, torch.Tensor]:
    """
    Create random input tensors for the first operations in a trace.

    This identifies which tensor IDs are "inputs" (used but never produced)
    and creates random tensors with the appropriate shapes/dtypes.

    Args:
        trace: List of TracedOp from OpTracer
        device: Device to create tensors on (default: cpu)

    Returns:
        Dict mapping input tensor IDs to random tensors
    """
    device = device or torch.device("cpu")

    # Find all tensor IDs that are produced by operations
    produced_ids: set[int] = set()
    for op in trace:
        produced_ids.update(op.output_tensor_ids)

    # Find input tensor IDs (used but not produced by any previous op)
    input_tensors: dict[int, torch.Tensor] = {}

    for op in trace:
        for tid, shape, dtype in zip(
            op.input_tensor_ids, op.input_shapes, op.input_dtypes
        ):
            if tid not in produced_ids and tid not in input_tensors:
                # This is an input tensor - create a random one
                if dtype.is_floating_point:
                    tensor = torch.randn(shape, dtype=dtype, device=device)
                elif dtype == torch.bool:
                    tensor = torch.randint(0, 2, shape, dtype=dtype, device=device)
                else:
                    # Integer types
                    tensor = torch.randint(0, 10, shape, dtype=dtype, device=device)
                input_tensors[tid] = tensor

    return input_tensors
