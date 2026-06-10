import ast
import contextlib
import inspect
import logging
import textwrap
import threading
from collections.abc import Callable, Generator, Iterable
from typing import Any, cast

from torch.utils._exposed_in import exposed_in

from .custom_ops import custom_op, CustomOpDef
from .infer_schema import infer_schema


logger = logging.getLogger(__name__)

triton_ops_to_kernels: dict[str, list[object]] = {}
_TRITON_WRAPPER_NAMES = {"capture_triton", "wrap_triton"}
_MAX_KERNEL_SEARCH_DEPTH = 5


def get_triton_kernels_for_op(name: str) -> list[object]:
    return triton_ops_to_kernels.get(name, [])


def _unwrap(fn: object) -> object:
    if not callable(fn):
        return fn

    try:
        return inspect.unwrap(cast(Callable[..., Any], fn))
    except ValueError:
        return fn


def _attribute_path(node: ast.AST) -> tuple[str, ...] | None:
    names = []
    while isinstance(node, ast.Attribute):
        names.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    names.append(node.id)
    names.reverse()
    return tuple(names)


def _is_triton_wrapper_call(node: ast.Call) -> bool:
    if isinstance(node.func, ast.Name):
        return node.func.id in _TRITON_WRAPPER_NAMES

    path = _attribute_path(node.func)
    if path is None or len(path) != 3:
        return False
    return path[:2] in (("torch", "_library"), ("torch", "library")) and (
        path[2] in _TRITON_WRAPPER_NAMES
    )


def _torch_op_name(node: ast.Call) -> str | None:
    path = _attribute_path(node.func)
    if path is None or len(path) < 4 or path[:2] != ("torch", "ops"):
        return None
    return f"{path[2]}::{path[3]}"


class _TritonKernelFinder:
    def __init__(self, jit_function_cls: type[object], autotuner_cls: type[object]):
        self.kernel_types = (jit_function_cls, autotuner_cls)
        self.visited_fns: set[int] = set()
        self.seen_kernels: set[int] = set()
        self.kernels: list[object] = []

    def find(self, fn: Callable[..., Any]) -> list[object]:
        self._scan_callable(fn, 0)
        return self.kernels

    def _try_scan_callable(self, fn: object, name: str, depth: int) -> None:
        try:
            self._scan_callable(fn, depth)
        except Exception:
            logger.debug("failed to analyze called function %s", name, exc_info=True)

    def _scan_callable(self, fn: object, depth: int) -> None:
        fn = _unwrap(fn)
        if (
            not callable(fn)
            or not hasattr(fn, "__code__")
            or id(fn) in self.visited_fns
        ):
            return

        if depth > _MAX_KERNEL_SEARCH_DEPTH:
            logger.debug(
                "reached max recursion depth (%s) in get_inner_triton_kernels",
                _MAX_KERNEL_SEARCH_DEPTH,
            )
            return

        self.visited_fns.add(id(fn))

        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(fn)).strip())
        except (OSError, TypeError, SyntaxError):
            return

        body = self._body(tree)
        namespace = self._namespace(fn)
        assignments = self._assignments(body)

        for node in self._walk(body):
            if isinstance(node, ast.Call):
                if _is_triton_wrapper_call(node):
                    if node.args:
                        self._resolve_expr(node.args[0], namespace, assignments, depth)
                    continue

                op_name = _torch_op_name(node)
                if op_name is not None:
                    self._scan_custom_op(op_name, depth)
                    continue

                if isinstance(node.func, ast.Name):
                    self._scan_name(node.func.id, namespace, assignments, depth)
            elif isinstance(node, ast.Return) and node.value is not None:
                self._resolve_expr(node.value, namespace, assignments, depth)

    @staticmethod
    def _body(tree: ast.Module) -> list[ast.stmt]:
        if len(tree.body) == 1 and isinstance(
            tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            return tree.body[0].body
        return tree.body

    @staticmethod
    def _walk(nodes: list[ast.stmt]) -> Iterable[ast.AST]:
        for node in nodes:
            yield from ast.walk(node)

    @staticmethod
    def _namespace(fn: object) -> dict[str, Any]:
        if not callable(fn) or not hasattr(fn, "__code__"):
            return {}

        closure_vars = inspect.getclosurevars(fn)
        namespace: dict[str, Any] = {}
        namespace.update(closure_vars.builtins)
        namespace.update(closure_vars.globals)
        namespace.update(closure_vars.nonlocals)
        namespace.update(getattr(fn, "__globals__", {}))
        return namespace

    @staticmethod
    def _assignments(nodes: list[ast.stmt]) -> dict[str, list[ast.expr]]:
        assignments: dict[str, list[ast.expr]] = {}
        for node in _TritonKernelFinder._walk(nodes):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assignments.setdefault(target.id, []).append(node.value)
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.value is not None
            ):
                assignments.setdefault(node.target.id, []).append(node.value)
        return assignments

    def _resolve_expr(
        self,
        expr: ast.expr,
        namespace: dict[str, Any],
        assignments: dict[str, list[ast.expr]],
        depth: int,
        resolving: set[str] | None = None,
    ) -> None:
        if resolving is None:
            resolving = set()

        if isinstance(expr, ast.Name):
            self._scan_name(expr.id, namespace, assignments, depth, resolving)
            return

        if isinstance(expr, ast.Call):
            if _is_triton_wrapper_call(expr):
                if expr.args:
                    self._resolve_expr(expr.args[0], namespace, assignments, depth)
                return

            if isinstance(expr.func, ast.Name):
                self._scan_name(expr.func.id, namespace, assignments, depth, resolving)

            for arg in expr.args:
                self._resolve_expr(arg, namespace, assignments, depth, resolving)
            for kwarg in expr.keywords:
                if kwarg.value is not None:
                    self._resolve_expr(
                        kwarg.value, namespace, assignments, depth, resolving
                    )
            return

        for child in ast.iter_child_nodes(expr):
            if isinstance(child, ast.expr):
                self._resolve_expr(child, namespace, assignments, depth, resolving)

    def _scan_name(
        self,
        name: str,
        namespace: dict[str, Any],
        assignments: dict[str, list[ast.expr]],
        depth: int,
        resolving: set[str] | None = None,
    ) -> None:
        if resolving is None:
            resolving = set()
        if name in resolving:
            return

        resolving.add(name)
        try:
            if name in assignments:
                for rhs in assignments[name]:
                    self._resolve_expr(rhs, namespace, assignments, depth, resolving)
                return

            obj = namespace.get(name)
            if obj is None:
                return

            if self._add_kernel(obj):
                return

            obj = _unwrap(obj)
            if callable(obj) and hasattr(obj, "__code__"):
                self._try_scan_callable(obj, name, depth + 1)
        finally:
            resolving.remove(name)

    def _scan_custom_op(self, name: str, depth: int) -> None:
        from torch._library.custom_ops import OPDEFS

        opdef = OPDEFS.get(name)
        if opdef is not None:
            self._try_scan_callable(opdef._abstract_fn, name, depth + 1)

    def _add_kernel(self, obj: object) -> bool:
        kernel = self._to_kernel(obj)
        if kernel is None:
            return False

        kernel_id = id(kernel)
        if kernel_id not in self.seen_kernels:
            self.seen_kernels.add(kernel_id)
            self.kernels.append(kernel)
        return True

    def _to_kernel(self, obj: object) -> object | None:
        if isinstance(obj, self.kernel_types):
            return obj

        inner = getattr(obj, "fn", None)
        if isinstance(inner, self.kernel_types):
            return inner

        return None


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

    That said, it is best effort. There are cases (e.g., deep recursive calls)
    that are not accounted for, so keep that in mind.
    """
    try:
        from triton.runtime.autotuner import Autotuner
        from triton.runtime.jit import JITFunction
    except ImportError:
        logger.warning("Triton not available, get_inner_triton_kernels = []")
        return []

    return _TritonKernelFinder(JITFunction, Autotuner).find(fn)


@exposed_in("torch.library")
def triton_op(
    name: str,
    fn: Callable | None = None,
    /,
    *,
    mutates_args: str | Iterable[str],
    schema: str | None = None,
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


def _is_interpreted_triton_kernel(triton_kernel: Callable) -> bool:
    try:
        from triton.runtime.interpreter import InterpretedFunction
    except ImportError:
        return False

    return isinstance(triton_kernel, InterpretedFunction)


def _has_interpreted_triton_autotuner(triton_kernel: Callable) -> bool:
    try:
        from triton.runtime.autotuner import Autotuner, Heuristics
        from triton.runtime.interpreter import InterpretedFunction
    except ImportError:
        return False

    seen: set[int] = set()
    kernel: Any = triton_kernel
    while isinstance(kernel, (Autotuner, Heuristics)):
        kernel_id = id(kernel)
        if kernel_id in seen:
            return False
        seen.add(kernel_id)
        kernel = kernel.fn
    return isinstance(kernel, InterpretedFunction)


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

    if isinstance(triton_kernel, JITFunction):
        if not is_wrap_triton_enabled():
            return triton_kernel
        return TraceableTritonKernelWrapper(triton_kernel, None, None)

    if isinstance(triton_kernel, Autotuner):
        # TRITON_INTERPRET=1 can make @triton.autotune wrap an
        # InterpretedFunction. It is eager-launchable but lacks the JIT
        # metadata the traceable wrapper needs.
        if _has_interpreted_triton_autotuner(triton_kernel):
            return triton_kernel
        if not is_wrap_triton_enabled():
            return triton_kernel
        return TraceableTritonKernelWrapper(triton_kernel, None, None)

    # TRITON_INTERPRET=1 makes @triton.jit return InterpretedFunction.
    if _is_interpreted_triton_kernel(triton_kernel):
        return triton_kernel

    raise RuntimeError(
        "wrap_triton only works on functions annotated with triton.jit or triton.autotune"
    )
