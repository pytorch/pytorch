import ast
import contextlib
import inspect
import logging
import threading
from collections.abc import Callable, Generator, Iterable
from typing import Any, Optional, Union

from torch.utils._exposed_in import exposed_in
from .custom_ops import custom_op, CustomOpDef
from .infer_schema import infer_schema


logger = logging.getLogger(__name__)

triton_ops_to_kernels: dict[str, list[object]] = {}


def get_triton_kernels_for_op(name: str) -> list[object]:
    return triton_ops_to_kernels.get(name, [])


def get_inner_triton_kernels(fn: Callable[..., Any]) -> list[object]:
    """
    Inspect the source of an arbitrary callable passed to torch._library.triton_op,
    and grab all of the triton kernels that are wrapped inside of it.

    This function traces local variable assignments to handle patterns like:
        kernel_fn = _my_kernel  # global JITFunction
        wrapped = some_wrapper(kernel_fn)
        capture_triton(wrapped)[grid](...)

    It also recursively analyzes called functions to find triton kernels hidden
    behind helper function calls.

    That said, it is best effort. There are cases (e.g., recursion > MAX_RECURSION_DEPTH)
    that are not accounted for, so keep that in mind.
    """

    # prevent infinite recursion
    MAX_RECURSION_DEPTH = 5

    def find_triton_kernels(
        fn: Callable[..., Any],
        visited_fns: set[int] | None = None,
        depth: int = 0,
    ) -> list[object]:
        try:
            from triton.runtime.autotuner import Autotuner
            from triton.runtime.jit import JITFunction
        except ImportError:
            logger.warning("Triton not available, find_triton_kernels = []")
            return []

        # init visited set and check for cycles/depth limit
        if visited_fns is None:
            visited_fns = set()

        fn_id = id(fn)
        if fn_id in visited_fns:
            return []
        if depth > MAX_RECURSION_DEPTH:
            logger.debug(
                "reached max recursion depth (%s) in find_triton_kernels",
                MAX_RECURSION_DEPTH,
            )
            return []

        visited_fns.add(fn_id)

        try:
            source = inspect.getsource(fn)
        except (OSError, TypeError):
            return []  # Source code not available

        from torch._inductor.utils import IndentedBuffer

        buffer = IndentedBuffer()
        buffer.splice(source, strip=True)
        tree = ast.parse(buffer.getrawvalue())

        # Visitor to collect function calls, assignments, and triton kernels
        class Visitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.triton_kernels: list[Any] = []
                # track local variable assignments: var_name -> list of RHS expressions
                self.assignments: dict[str, list[ast.expr]] = {}
                # track function calls
                self.called_functions: list[str] = []

            def visit_Assign(self, node: ast.Assign) -> None:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.assignments.setdefault(target.id, []).append(node.value)
                self.generic_visit(node)

            def visit_Call(self, node: ast.Call) -> None:
                triton_func_names = ("capture_triton", "wrap_triton")
                if isinstance(node.func, ast.Attribute):
                    attr = node.func
                    if isinstance(attr.value, ast.Attribute):
                        if (
                            isinstance(attr.value.value, ast.Name)
                            and attr.value.value.id == "torch"
                            and attr.value.attr == "_library"
                            and attr.attr in triton_func_names
                        ):
                            if node.args and isinstance(node.args[0], ast.Name):
                                self.triton_kernels.append(node.args[0].id)
                        elif (
                            isinstance(attr.value.value, ast.Attribute)
                            and isinstance(attr.value.value.value, ast.Name)
                            and attr.value.value.value.id == "torch"
                            and attr.value.value.attr == "ops"
                        ):
                            self.called_functions.append(
                                f"{attr.value.attr}::{attr.attr}"
                            )
                # Catch capture_triton, wrap_triton that's been
                # imported directly
                elif isinstance(node.func, ast.Name):
                    if node.func.id in triton_func_names:
                        if node.args and isinstance(node.args[0], ast.Name):
                            self.triton_kernels.append(node.args[0].id)
                    else:
                        # track regular function calls for recursive analysis
                        self.called_functions.append(node.func.id)

                self.generic_visit(node)

        collector = Visitor()
        collector.visit(tree)
        closure_vars = inspect.getclosurevars(fn)

        # build combined globals from closure and function's __globals__
        all_globals: dict[str, Any] = {}
        all_globals.update(closure_vars.globals)
        if hasattr(fn, "__globals__"):
            all_globals.update(fn.__globals__)

        def extract_names_from_expr(expr: ast.expr) -> list[str]:
            """Extract all Name references from an AST expression."""
            names: list[str] = []

            class NameExtractor(ast.NodeVisitor):
                def visit_Name(self, node: ast.Name) -> None:
                    names.append(node.id)

                def visit_Call(self, node: ast.Call) -> None:
                    # for function calls, visit the function and all args
                    self.generic_visit(node)

            NameExtractor().visit(expr)
            return names

        def resolve_to_kernel(obj: object) -> object | None:
            """Check if obj is a triton kernel or wrapper and return the kernel."""
            if isinstance(obj, (JITFunction, Autotuner)):
                return obj
            # handle wrappers that have a .fn attribute pointing to JITFunction
            if callable(obj) and hasattr(obj, "fn"):
                inner = obj.fn
                if isinstance(inner, JITFunction):
                    return inner
            return None

        def trace_to_global_kernels(
            name: str, visited: set[str] | None = None
        ) -> list[object]:
            """
            Trace a name through local assignments back to global triton kernels.

            This handles patterns like:
                kernel_fn = _my_kernel  # global
                wrapped = wrapper(kernel_fn)
                autotuned = autotune(wrapped)
                capture_triton(autotuned)  # traces back to _my_kernel
            """
            if visited is None:
                visited = set()

            if name in visited:
                return []

            visited.add(name)

            # try direct resolution from globals
            if name in all_globals:
                kernel = resolve_to_kernel(all_globals[name])
                if kernel is None:
                    logger.warning(
                        "failed to resolve all_globals[%s] to a triton kernel", name
                    )
                    return []
                return [kernel]

            # try closure nonlocals
            if name in closure_vars.nonlocals:
                kernel = resolve_to_kernel(closure_vars.nonlocals[name])
                if kernel is None:
                    logger.warning(
                        "failed to resolve closure_vars.nonlocals[%s] to a triton kernel",
                        name,
                    )
                    return []
                return [kernel]

            # try builtins (this seems unlikely, but for completeness/bc why not)
            if name in closure_vars.builtins:
                kernel = resolve_to_kernel(closure_vars.builtins[name])
                if kernel is None:
                    logger.warning(
                        "failed to resolve closure_vars.builtins[%s] to a triton kernel",
                        name,
                    )
                    return []
                return [kernel]

            # not in globals/nonlocals/builtins, check if it's a local assignment
            if name not in collector.assignments:
                logger.warning("%s not in collector.assignments", name)
                return []

            # trace through assignments - collect all names referenced in RHS
            results: list[object] = []
            for rhs_expr in collector.assignments[name]:
                referenced = extract_names_from_expr(rhs_expr)
                for ref_name in referenced:
                    traced = trace_to_global_kernels(ref_name, visited)
                    results.extend(traced)

            return results

        # resolve kernel names, tracing through local variables if needed
        resolved: list[object] = []
        seen_ids: set[int] = set()

        for name in collector.triton_kernels:
            traced_objects = trace_to_global_kernels(name)
            for obj in traced_objects:
                obj_id = id(obj)
                if obj_id not in seen_ids:
                    seen_ids.add(obj_id)
                    resolved.append(obj)

        for func_name in collector.called_functions:
            # try resolving the function from globals or closure
            func_obj = None
            if func_name in all_globals:
                func_obj = all_globals[func_name]
            elif func_name in closure_vars.nonlocals:
                func_obj = closure_vars.nonlocals[func_name]

            if func_obj is None:
                from torch._library.custom_ops import OPDEFS

                if func_name in OPDEFS:
                    func_obj = OPDEFS[func_name]._abstract_fn

            # skip if not a callable or if it's a triton kernel itself
            if func_obj is None or not callable(func_obj):
                continue

            # skip built-in functions and C extensions (they can't contain triton kernels)
            if not hasattr(func_obj, "__code__"):
                continue

            try:
                nested_kernels = find_triton_kernels(func_obj, visited_fns, depth + 1)
                for kernel in nested_kernels:
                    kernel_id = id(kernel)
                    if kernel_id not in seen_ids:
                        seen_ids.add(kernel_id)
                        resolved.append(kernel)
            except Exception:
                logger.debug(
                    "failed to analyze called function %s", func_name, exc_info=True
                )

        return resolved

    return find_triton_kernels(fn)


@exposed_in("torch.library")
def triton_op(
    name: str,
    fn: Optional[Callable] = None,
    /,
    *,
    mutates_args: Union[str, Iterable[str]],
    schema: Optional[str] = None,
) -> Callable:
    """Create a custom operator whose implementation is backed by 1+ triton kernels.

    This is a more structured way of using triton kernels with PyTorch.
    Prefer using triton kernels with no ``torch.library`` custom operator wrappers
    (like :func:`torch.library.custom_op`, :func:`torch.library.triton_op`) because
    that is simpler;
    only use :func:`torch.library.custom_op`/:func:`torch.library.triton_op` if you
    want to create an operator that behaves like PyTorch built-in operators.
    For example, you may use a ``torch.library`` wrapper API to define the
    behavior of the triton kernel when passed a tensor subclass or under
    a TorchDispatchMode.

    Use :func:`torch.library.triton_op` instead of :func:`torch.library.custom_op`
    when the implementation
    consists of 1+ triton kernels. :func:`torch.library.custom_op` treats
    custom operators as opaque (:func:`torch.compile` and
    :func:`torch.export.export` will never trace into them), but ``triton_op``
    makes the implementation visible to these subsystems, allowing them
    to optimize the triton kernel(s).

    Note that ``fn`` must only consist of calls to PyTorch-understood
    operators and triton kernels. Any triton kernels called inside ``fn``
    must be wrapped in a call to :func:`torch.library.wrap_triton`.

    Args:
        name (str): A name for the custom op that looks like "{namespace}::{name}",
            e.g. "mylib::my_linear". The name is used as the op's stable identifier
            in PyTorch subsystems (e.g. torch.export, FX graphs).
            To avoid name collisions, please use your project name as the namespace;
            e.g. all custom ops in pytorch/fbgemm use "fbgemm" as the namespace.
        mutates_args (Iterable[str] or "unknown"): The names of args that the function mutates.
            This MUST be accurate, otherwise, the behavior is undefined. If "unknown",
            it pessimistically assumes that all inputs to the operator are being mutated.
        schema (str | None): A schema string for the operator. If None
            (recommended) we'll infer a schema for the operator from its type
            annotations. We recommend letting us infer a schema unless you
            have a specific reason not to.
            Example: "(Tensor x, int y) -> (Tensor, Tensor)".

    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> import torch
        >>> from torch.library import triton_op, wrap_triton
        >>>
        >>> import triton
        >>> from triton import language as tl
        >>>
        >>> @triton.jit
        >>> def add_kernel(
        >>>     in_ptr0,
        >>>     in_ptr1,
        >>>     out_ptr,
        >>>     n_elements,
        >>>     BLOCK_SIZE: "tl.constexpr",
        >>> ):
        >>>     pid = tl.program_id(axis=0)
        >>>     block_start = pid * BLOCK_SIZE
        >>>     offsets = block_start + tl.arange(0, BLOCK_SIZE)
        >>>     mask = offsets < n_elements
        >>>     x = tl.load(in_ptr0 + offsets, mask=mask)
        >>>     y = tl.load(in_ptr1 + offsets, mask=mask)
        >>>     output = x + y
        >>>     tl.store(out_ptr + offsets, output, mask=mask)
        >>>
        >>> @triton_op("mylib::add", mutates_args={})
        >>> def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        >>>     output = torch.empty_like(x)
        >>>     n_elements = output.numel()
        >>>
        >>>     def grid(meta):
        >>>         return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        >>>
        >>>     # NB: we need to wrap the triton kernel in a call to wrap_triton
        >>>     wrap_triton(add_kernel)[grid](x, y, output, n_elements, 16)
        >>>     return output
        >>>
        >>> @torch.compile
        >>> def f(x, y):
        >>>     return add(x, y)
        >>>
        >>> x = torch.randn(3, device="cuda")
        >>> y = torch.randn(3, device="cuda")
        >>>
        >>> z = f(x, y)
        >>> assert torch.allclose(z, x + y)

    """

    def dec(fn: Callable[..., object]) -> CustomOpDef:
        def backend_fn(*args, **kwargs):  # type: ignore[no-untyped-def]
            # Optimization: we're passing regular Tensors into the triton kernel, so
            # no need to go through HOP dispatch
            with set_wrap_triton_enabled(False):
                return fn(*args, **kwargs)

        result = custom_op(
            name,
            backend_fn,
            mutates_args=mutates_args,
            schema=infer_schema(fn, mutates_args=mutates_args),
        )
        from .._subclasses.functional_tensor import FunctionalTensorMode

        # We require that the user pass us a function that is make_fx traceable,
        # so we can just register it as the Fake/meta kernel.
        result.register_fake(fn)

        # We decompose the operator when FunctionalTensorMode is active.
        # The goal is to decompose the operator in AOTDispatcher.
        # - With torch.compile, this means that the backend (usually Inductor)
        #   can see a call to the triton kernel(s) and so it can directly optimize
        #   them by inlining them into the lowering process.
        def functional_decomp(  # type: ignore[no-untyped-def]
            mode, op, types, args, kwargs
        ):
            # NOTE [Export custom triton op]
            # For torch.export (strict and non-strict), we don't do functional decomposition.
            # Instead, we preserve the custom triton ops as custom ops. This is because we want
            # the exported program to be high-level and serializable. If we decompose
            # the custom op to a functional hop and make it a node in exported program,
            # we need to figure out ways of serializing the hop and its arguments, which can be triton.jited
            # functions and triton dtypes. This is undesirable because:
            # - it can be tedious to maintain a layer that serializes the jited function (e.g. with a string) and dtypes.
            # - exported program will contain the implementation detail (e.g. triton source code) for a specific
            #   backend (GPU), which is probably at a wrong level of abstraction.
            # - changes to triton or the serialization logic for triton arguments can be BC breaking
            #
            # In the short term, we expect users to have a separate aot_compile stage that compiles the exported program
            # into a Cubin file on the same machine that users call export, which does autotuning and removes triton
            # dependency and serve the model with Cubin. This guarantees that triton changes won't break BC.
            # In the long term, we may export multiple cubins for the triton op directly
            from torch.export._trace import custom_triton_ops_decomposition_disabled

            if custom_triton_ops_decomposition_disabled():
                return mode.__torch_dispatch__(op, types, args, kwargs)
            else:
                # TODO: https://github.com/pytorch/pytorch/issues/160333
                # We should deduplicate the unrecognized_types logic.
                import torch._subclasses

                unrecognized_types = [
                    t
                    for t in types
                    if not issubclass(t, torch._subclasses.FakeTensor)
                    and t
                    not in [
                        torch.Tensor,
                        torch._subclasses.functional_tensor.FunctionalTensor,
                    ]
                ]

                if unrecognized_types:
                    return NotImplemented
                with mode:
                    return fn(*args, **kwargs)

        triton_kernels = get_inner_triton_kernels(fn)
        triton_ops_to_kernels[name] = triton_kernels
        result.register_torch_dispatch(FunctionalTensorMode, functional_decomp)
        return result

    if fn is None:
        return dec
    else:
        return dec(fn)


wrap_triton_enabled = threading.local()
wrap_triton_enabled_default = True


@contextlib.contextmanager
def set_wrap_triton_enabled(enabled: bool) -> Generator[None, None, None]:
    """If triton kernels annotated with @wrap_triton should dispatch via HOP
    or go straight to the triton kernel execution.

    We have this switch because eager-mode performance of HOP dispatch is slow
    enough to matter (~1ms) and we know that wrap_triton isn't necessary in
    some situations (eager-mode with regular Tensors)
    """
    try:
        prev = is_wrap_triton_enabled()
        wrap_triton_enabled.value = enabled
        yield
    finally:
        wrap_triton_enabled.value = prev


def is_wrap_triton_enabled() -> bool:
    return getattr(wrap_triton_enabled, "value", wrap_triton_enabled_default)


def capture_triton(triton_kernel: Callable, /) -> Any:
    """This API has been renamed to wrap_triton"""
    return wrap_triton(triton_kernel)


@exposed_in("torch.library")
def wrap_triton(triton_kernel: Callable, /) -> Any:
    """Allows capture of a triton kernel into a graph via make_fx or
    non-strict ``torch.export``.

    These technologies perform Dispatcher-based tracing (via
    ``__torch_dispatch__``) and cannot see calls to raw triton kernels.
    The ``wrap_triton`` API wraps a triton kernel into a callable that
    can actually be traced into a graph.

    Please use this API together with :func:`torch.library.triton_op`.

    Examples:

        >>> # xdoctest: +SKIP
        >>> import torch
        >>> import triton
        >>> from triton import language as tl
        >>> from torch.fx.experimental.proxy_tensor import make_fx
        >>> from torch.library import wrap_triton
        >>>
        >>> @triton.jit
        >>> def add_kernel(
        >>>     in_ptr0,
        >>>     in_ptr1,
        >>>     out_ptr,
        >>>     n_elements,
        >>>     BLOCK_SIZE: "tl.constexpr",
        >>> ):
        >>>     pid = tl.program_id(axis=0)
        >>>     block_start = pid * BLOCK_SIZE
        >>>     offsets = block_start + tl.arange(0, BLOCK_SIZE)
        >>>     mask = offsets < n_elements
        >>>     x = tl.load(in_ptr0 + offsets, mask=mask)
        >>>     y = tl.load(in_ptr1 + offsets, mask=mask)
        >>>     output = x + y
        >>>     tl.store(out_ptr + offsets, output, mask=mask)
        >>>
        >>> def add(x, y):
        >>>     output = torch.empty_like(x)
        >>>     n_elements = output.numel()
        >>>
        >>>     def grid_fn(meta):
        >>>         return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        >>>
        >>>     wrap_triton(add_kernel)[grid_fn](x, y, output, n_elements, 16)
        >>>     return output
        >>>
        >>> x = torch.randn(3, device="cuda")
        >>> y = torch.randn(3, device="cuda")
        >>> gm = make_fx(add)(x, y)
        >>> print(gm.code)
        >>> # def forward(self, x_1, y_1):
        >>> #     empty_like = torch.ops.aten.empty_like.default(x_1, pin_memory = False)
        >>> #     triton_kernel_wrapper_mutation_proxy = triton_kernel_wrapper_mutation(
        >>> #         kernel_idx = 0, constant_args_idx = 0,
        >>> #         grid = [(1, 1, 1)], kwargs = {
        >>> #             'in_ptr0': x_1, 'in_ptr1': y_1, 'out_ptr': empty_like,
        >>> #             'n_elements': 3, 'BLOCK_SIZE': 16
        >>> #         })
        >>> #     return empty_like

    """
    from triton.runtime.autotuner import Autotuner
    from triton.runtime.jit import JITFunction

    from torch._higher_order_ops.triton_kernel_wrap import TraceableTritonKernelWrapper

    if not isinstance(triton_kernel, (JITFunction, Autotuner)):
        raise RuntimeError(
            "wrap_triton only works on functions annotated with triton.jit or triton.autotune"
        )
    if not is_wrap_triton_enabled():
        return triton_kernel
    return TraceableTritonKernelWrapper(triton_kernel, None, None)
