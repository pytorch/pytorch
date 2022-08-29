import collections
import dataclasses
import warnings
from contextlib import contextmanager, nullcontext
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.fx.traceback as fx_traceback
import torch.nn as nn
import torch.utils._pytree as pytree
import torch.utils.dlpack
from torch import Tensor
from torch._subclasses import FakeTensorMode
from torch.fx import immutable_collections, Interpreter
from torch.nn.utils import stateless

from functorch import make_fx
from functorch._C import CompileCache
from functorch.experimental import functionalize
from . import config
from .decompositions import register_decomposition
from .named_members_polyfill import _named_buffers, _named_parameters
from .partitioners import default_partition

try:
    from torchdynamo import disable as disable_torchdynamo
except ImportError:

    def disable_torchdynamo(x):
        return x


pytree._register_pytree_node(
    immutable_collections.immutable_list,
    lambda x: (list(x), None),
    lambda x, c: immutable_collections.immutable_list(x),
)
pytree._register_pytree_node(
    immutable_collections.immutable_dict,
    lambda x: (list(x.values()), list(x.keys())),
    lambda x, c: immutable_collections.immutable_dict(
        {key: value for key, value in zip(c, x)}
    ),
)

aten = torch.ops.aten


@contextmanager
def preserve_rng_state():
    rng_state = torch.clone(torch.random.get_rng_state())
    if torch.cuda.is_available():
        cuda_rng_state = torch.clone(torch.cuda.get_rng_state())
    try:
        yield
    finally:
        torch.random.set_rng_state(rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_rng_state)


# Set up hooks so that during backward the fx's stack_trace is properly set
callback_set = False


def setup_stacktrace_preservation_hooks(roots: List):
    def iter_graph(roots):
        if not roots:
            return
        seen = set()
        q = collections.deque()
        for node in roots:
            if node is not None:
                seen.add(node)
                q.append(node)

        while q:
            node = q.popleft()
            for fn, _idx in node.next_functions:
                if fn in seen or fn is None:
                    continue
                seen.add(fn)
                q.append(fn)

            yield node

    def get_callback(saved_stack_):
        def callback():
            global callback_set
            fx_traceback.set_stack_trace(saved_stack_)
            callback_set = False

        return callback

    def get_prehook(stack_):
        def prehook(grad_output):
            global callback_set

            if not callback_set:
                torch.autograd.variable.Variable._execution_engine.queue_callback(
                    get_callback(fx_traceback.format_stack())
                )
                callback_set = True

            fx_traceback.set_stack_trace(stack_)

        return prehook

    def get_posthook(special_stack_):
        def posthook(grad_input, grad_output):
            fx_traceback.set_stack_trace(special_stack_)

        return posthook

    for node in iter_graph(roots):
        forward_node_stack = node.metadata.get("traceback_", [])
        node.register_prehook(get_prehook(forward_node_stack))

        special_stack = forward_node_stack.copy()
        special_stack.append(
            "Gradient addition node due to mulitple use of tensor around:"
        )
        node.register_hook(get_posthook(special_stack))


def create_joint_forward_backward(fn):
    def joint_forward_backward(
        primals: List[Any], tangents: List[Any]
    ) -> Tuple[List[Any], List[Any]]:
        # Call the forward pass
        outs = fn(*primals)
        # Get the inputs that need gradients
        grad_primals = []
        inputs_needs_grads = []
        for p in primals:
            is_grad_tensor = isinstance(p, Tensor) and p.requires_grad
            inputs_needs_grads.append(is_grad_tensor)
            if is_grad_tensor:
                grad_primals.append(p)

        # Get the outputs that need gradients
        assert len(tangents) == len(outs)
        needed_outs = []
        needed_tangents = []
        for out, tangent in zip(outs, tangents):
            if isinstance(out, Tensor) and out.requires_grad:
                needed_outs.append(out)
                needed_tangents.append(tangent)

        setup_stacktrace_preservation_hooks([out.grad_fn for out in needed_outs])

        backward_out = []
        # Call the backwards pass
        if grad_primals:
            with fx_traceback.override_stack_trace():
                backward_out = torch.autograd.grad(
                    needed_outs,
                    grad_primals,
                    grad_outputs=needed_tangents,
                    allow_unused=True,
                )
        backward_out_iter = iter(backward_out)
        return outs, [
            next(backward_out_iter) if i else None for i in inputs_needs_grads
        ]

    return joint_forward_backward


def normalize_as_list(x):
    if isinstance(x, tuple):
        return list(x)
    elif isinstance(x, list):
        return x
    return [x]


aot_autograd_decompositions = {}

# TODO: Remove these stupid decompositions
@register_decomposition(aten._reshape_alias, aot_autograd_decompositions)
def _reshape_alias(x, shape, strides):
    return aten.view(x, shape)


# This is a list since looking forward, we can have this arbitrarily nested.
graph_being_compiled: List[str] = []
nth_graph: int = 0
model_name: str = "model"


def set_model_name(name):
    global model_name
    model_name = name


def get_aot_compilation_context() -> Tuple[List[str], str, int]:
    return list(graph_being_compiled), model_name, nth_graph


def get_aot_graph_name() -> str:
    """
    Returns the name of the graph being compiled.
    """
    global model_name, graph_being_compiled, nth_graph
    return f"{model_name}_{'_'.join(graph_being_compiled)}_{nth_graph}"


get_graph_being_compiled = get_aot_graph_name


@contextmanager
def track_graph_compiling(graph_name, increment_index=False):
    global graph_being_compiled
    graph_being_compiled = [graph_name]
    yield
    if increment_index:
        global nth_graph
        nth_graph += 1
    graph_being_compiled = []


def make_boxed_func(f):
    def g(args):
        return f(*args)

    g._boxed_call = True
    return g


def make_boxed_compiler(compiler):
    @wraps(compiler)
    def f(fx_g, inps):
        out_f = compiler(fx_g, inps)
        fx_g = make_boxed_func(out_f)
        return fx_g

    return f


def call_func_with_args(f, args, steal_args=False):
    if not steal_args:
        args = list(args)
    assert isinstance(args, list)

    if hasattr(f, "_boxed_call"):
        out = normalize_as_list(f(args))
    else:
        # TODO: Please remove soon
        # https://github.com/pytorch/pytorch/pull/83137#issuecomment-1211320670
        warnings.warn(
            "Your compiler for AOTAutograd is returning a a function that doesn't take boxed arguments. "
            "Please wrap it with functorch.compile.make_boxed_func or handle the boxed arguments yourself. "
            "See https://github.com/pytorch/pytorch/pull/83137#issuecomment-1211320670 for rationale."
        )
        out = normalize_as_list(f(*args))
    return out


@dataclasses.dataclass
class AOTConfig:
    """
    Configuration for AOTDispatcher
    """

    fw_compiler: Callable
    bw_compiler: Callable
    partition_fn: Callable
    decompositions: Dict[Callable, Callable]


def aot_dispatch_base(flat_fn, flat_args: List[Tensor], aot_config: AOTConfig):
    fw_module = make_fx(flat_fn, aot_config.decompositions)(*flat_args)
    with track_graph_compiling("inference"):
        compiled_fw = aot_config.fw_compiler(fw_module, flat_args)

    @wraps(compiled_fw)
    def new_fn(args):
        fw_outs = call_func_with_args(compiled_fw, args)
        return fw_outs

    return new_fn


def aot_dispatch_autograd(flat_fn, flat_args: List[Tensor], aot_config: AOTConfig):
    joint_forward_backward = create_joint_forward_backward(flat_fn)
    out = flat_fn(*flat_args)
    out = pytree.tree_map(
        lambda x: x.detach().contiguous() if isinstance(x, Tensor) else x,
        out,
    )

    if isinstance(out, (list, tuple)):
        _num_outs = len(out)
    else:
        _num_outs = 1

    joint_inputs = (flat_args, out)
    fx_g = make_fx(joint_forward_backward, aot_config.decompositions)(*joint_inputs)

    if config.use_functionalize:
        # Functionalize the foward backward graph. First create a
        # fake fn to make functionalize happy
        def fake_fn(primals, tangents):
            return fx_g(primals, tangents)

        fx_g = make_fx(functionalize(fake_fn))(*joint_inputs)

    if config.debug_joint:
        print("====== Joint graph ======")
        fx_g.print_readable()

    with torch.no_grad():
        with track_graph_compiling("joint"):
            fw_module, bw_module = aot_config.partition_fn(fx_g, joint_inputs)

        if config.debug_graphs:
            print("====== Forward graph ======")
            fw_module.print_readable()
            print("====== Backward graph ======")
            bw_module.print_readable()

        with track_graph_compiling("forward"):
            compiled_fw_func = aot_config.fw_compiler(fw_module, flat_args)

        if config.debug_partitioner:
            fw_outs = call_func_with_args(compiled_fw_func, flat_args)
            activation_sizes = 0
            for out in fw_outs[_num_outs:]:
                if isinstance(out, torch.Tensor):
                    activation_sizes += out.storage().nbytes()
            print(f"Real Activations Stored(GB): {activation_sizes/1e9}")

    class CompiledFunction(torch.autograd.Function):
        compiled_fw = compiled_fw_func
        compiled_bw = None
        num_outs = _num_outs

        @staticmethod
        @disable_torchdynamo
        def forward(ctx, *flat_tensor_args):
            fw_outs = call_func_with_args(
                CompiledFunction.compiled_fw, flat_tensor_args
            )
            num_outs = CompiledFunction.num_outs
            ctx.save_for_backward(*fw_outs[num_outs:])
            return tuple(fw_outs[0:num_outs])

        @staticmethod
        @disable_torchdynamo
        def backward(ctx, *flat_args):
            contiguous_args = [t.contiguous() for t in flat_args]
            all_args = list(ctx.saved_tensors) + list(contiguous_args)
            if CompiledFunction.compiled_bw is None:
                with track_graph_compiling("backward", True):
                    CompiledFunction.compiled_bw = aot_config.bw_compiler(
                        bw_module, all_args
                    )
            ctx.maybe_clear_saved_tensors()
            out = call_func_with_args(
                CompiledFunction.compiled_bw, all_args, steal_args=True
            )

            return tuple(out)

    return CompiledFunction.apply


def create_aot_dispatcher_function(
    flat_fn, flat_args: List[Tensor], aot_config: AOTConfig
):
    """
    Traces the forward and backward graphs of the attr:`flat_fn` to generate a
    joint graph. The joint graph is an Fx graph with Aten ops. Please refer to
    the tracing mechanism to understand the graph capturing details.

    The joint graph is then passed through attr:`partition_fn` to isolate the
    forward and backward portions, which are then respectively compiled via the
    provided attr:`fw_compiler` and attr:`bw_compiler`.

    The resulting compiled forward and backward graphs are then wrapped up in a
    ``torch.autograd.Function`` object.
    """
    if aot_config.decompositions is None:
        aot_config.decompositions = {}

    aot_config.decompositions = {
        **aot_autograd_decompositions,
        **aot_config.decompositions,
    }
    fake_mode = FakeTensorMode.push() if config.use_fake_tensor else nullcontext()

    with preserve_rng_state(), fake_mode as mode:

        def process_inputs(flat_args):
            if mode:
                fake_flat_tensor_args = pytree.tree_map_only(
                    Tensor, mode.from_tensor, flat_args
                )
            else:
                # The detach().requires_grad_() pattern can cause some subtle bugs.
                # These will be fixed once FakeTensor is always-on for AOTAutograd.
                #
                # For models that might resize their inputs, the input tensors
                # must have allow_tensor_metadata_change() set to true.
                # detach() returns a view tensor, but with that field set to false.
                #
                # Specifically, this breaks quantized models
                # (resnet50_quantized_qat and mobilenet_v2_quantized_qat)
                # because they use a "running-mean" style op that requires
                # resizing the running counter buffers stored on the module.
                fake_flat_tensor_args = pytree.tree_map_only(
                    Tensor,
                    lambda x: x.detach().requires_grad_(x.requires_grad),
                    flat_args,
                )
            return fake_flat_tensor_args

        fake_flat_tensor_args = process_inputs(flat_args)

        needs_autograd = (
            any(
                [
                    x.requires_grad
                    for x in fake_flat_tensor_args
                    if isinstance(x, Tensor)
                ]
            )
            and torch.is_grad_enabled()
        )
        # crappy version of dispatcher
        # TODO: Do this properly
        if needs_autograd:
            return make_boxed_func(
                aot_dispatch_autograd(flat_fn, fake_flat_tensor_args, aot_config)
            )
        else:
            return aot_dispatch_base(flat_fn, fake_flat_tensor_args, aot_config)


class _CompileCache(CompileCache):
    pass


# using a C++-based pytree reduces the overhead by about 50%
compile_cache = None


# Inspired by autodidax (thanks!)
class PytreeThunk:
    spec = None
    # These are some kinda dumb microoptimizations that save about 3-4 us of overhead.
    is_simple = (
        None  # if the output spec is a tuple/list, we won't bother unflattening it.
    )
    is_really_simple = None  # if the output spec is a LeafSpec

    def set(self, spec):
        assert self.spec is None or self.spec == spec
        self.spec = spec
        if type(self.spec) in [tuple, list] and all(
            isinstance(i, pytree.LeafSpec) for i in spec.children_specs
        ):
            self.is_simple = True
        if isinstance(self.spec, pytree.LeafSpec):
            self.is_really_simple = True

    def unflatten(self, x):
        if self.is_really_simple:
            return x[0]
        if self.is_simple:
            return x
        return pytree.tree_unflatten(x, self.spec)


def filter_tensor_and_static_args(args, static_argnums):
    """
    Separate out the tensor and static args. Also, for the static args, store
    the hash.
    """
    tensor_args = []
    static_args = []
    static_args_hashed = []
    for idx, arg in enumerate(args):
        if idx not in static_argnums:
            tensor_args.append(arg)
        else:
            static_args.append(arg)
            static_args_hashed.append(arg.__hash__())
    return tensor_args, static_args, static_args_hashed


def rearrange(tensor_args, static_args, static_argnums):
    """
    Generate the args as per the original spec. static_argnums is sorted.
    """
    tensor_index = 0
    static_index = 0
    index = 0
    args = []
    assert len(static_args) == len(static_argnums)
    while tensor_index < len(tensor_args) and static_index < len(static_args):
        if index == static_argnums[static_index]:
            args.append(static_args[static_index])
            static_index += 1
        else:
            args.append(tensor_args[tensor_index])
            tensor_index += 1
        index += 1

    while tensor_index < len(tensor_args):
        args.append(tensor_args[tensor_index])
        tensor_index += 1

    while static_index < len(static_args):
        args.append(static_args[static_index])
        static_index += 1

    return args


KNOWN_TYPES = [torch.Tensor, int, str, float, bool]


def aot_function(
    fn: Callable,
    fw_compiler: Callable,
    bw_compiler: Optional[Callable] = None,
    partition_fn: Callable = default_partition,
    decompositions: Optional[Dict] = None,
    hasher_type: str = "StaticShapeHasher",
    static_argnums: Optional[Tuple[int]] = None,
) -> Callable:
    """
    Traces the forward and backward graph of :attr:`fn` using torch dispatch
    mechanism, and then compiles the generated forward and backward graphs
    through :attr:`fw_compiler` and :attr:`bw_compiler`.

    :func:`aot_function` traces the forward and backward graph ahead of time,
    and generates a joint forward and backward graph.  :attr:`partition_fn` is
    then used to separate out forward and backward graphs. The partitioner
    function can be used to perform optimizations such as recomputation. One can
    set `decompositions` dictionary to decompose the operators into a sequence
    of core or simpler operators supported by the backend compilers.

    :func:`aot_function` uses a compilation cache, based on input tensor
    properties, to detect when there is a need of recompilation. By default, its
    behavior is static, i.e., it recompiles if shape of any input tensor
    changes.

    :attr:`static_argnums` allows user to mark the arguments of the original
    :attr:`fn` as static. This is useful when an argument is a non-tensor, e.g.,
    ``int`` or ``bool``. A change in the actual value of static arg causes
    recompilation.

    .. warning::
        This API is experimental and likely to change.

    Args:
        fn (Callable): A Python function that takes one ore more arguments. Must
            return one or more Tensors.
        fw_compiler (Callable): A Python function that accepts an Fx graph with
            Aten ops and input args, and returns a Callable that semantically is
            equivalent to the input Fx graph.
        bw_compiler (Optional[Callable]): A Python function that accepts an
            Fx graph with Aten ops and input args, and returns a Callable that
            semantically is equivalent to the input Fx graph.  Default: None
            (when None, it defaults to the :attr:`fw_compiler`)
        partition_fn (Callable): A Python function that takes a joint forward
            and backward graph, and partitions it into separate forward and
            backward graphs.
        decompositions (Dict): A dictionary to define the decomposition of
            larger Aten ops into simpler or core Aten ops.
        static_argnums (Optional[Tuple[Int]]): An option tuple of ints to mark
            the arguments of the function as static.

    Returns:
        Returns a ``Callable`` that retains the eager behavior of the original
        :attr:`fn`, but with forward and backward graph compiled via
        :attr:`fw_compile` and :attr:`bw_compile`.

    A simple example usage of :func:`aot_function` is as follows. This example
    will print the forward and backward graphs of the function ``fn``

        >>> fn = lambda x : x.sin().cos()
        >>> def print_compile_fn(fx_module, args):
        >>>     print(fx_module)
        >>>     return fx_module
        >>> aot_fn = aot_function(fn, print_compile_fn)
        >>> x = torch.randn(4, 5, requires_grad=True)
        >>> aot_fn(x)

    The static argnums are used to mark the non-tensor arguments as static. An
    example is as follows where the dropout probability is as argument to the
    original function.

        >>> def fn(input, bias, residual, p: float):
        >>>     a = torch.add(input, bias)
        >>>     b = torch.nn.functional.dropout(a, p, training=True)
        >>>     c = b + residual
        >>>     return c
        >>> aot_fn = aot_function(fn, print_compile_fn, static_argnums=(3,))

    """
    global compile_cache
    if compile_cache is None:
        compile_cache = CompileCache()
    if bw_compiler is None:
        bw_compiler = fw_compiler
    aot_config = AOTConfig(
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        partition_fn=partition_fn,
        decompositions=decompositions,
    )
    cached_res = None

    fn_id = id(fn)
    fw_compiler_id = id(fw_compiler)
    bw_compiler_id = id(bw_compiler)

    if isinstance(static_argnums, int):
        static_argnums = [static_argnums]
    elif static_argnums is not None and len(static_argnums) == 0:
        static_argnums = None
    elif static_argnums is not None:
        static_argnums = list(static_argnums)
        static_argnums.sort()

    @wraps(fn)
    def returned_function(*args, **kwargs):
        global compile_cache
        nonlocal cached_res

        # Separate out static args if static_argnums is present
        tensor_args = args
        static_args = []
        # TODO - move the hashing part of static_args to C++.
        static_args_hashed = []
        if static_argnums is not None:
            (
                tensor_args,
                static_args,
                static_args_hashed,
            ) = filter_tensor_and_static_args(args, static_argnums)

        # Now flatten the tensor args
        flat_tensor_args, _ = pytree.tree_flatten((tensor_args, kwargs))

        # Check if the fn is already compiled
        num_tensor_args = len(flat_tensor_args)
        flat_args_for_cache = flat_tensor_args + static_args_hashed
        cached_res = compile_cache.at(
            fn_id,
            fw_compiler_id,
            bw_compiler_id,
            num_tensor_args,
            hasher_type,
            *flat_args_for_cache,
        )

        # Compile the function and save it in the cache
        if cached_res is None:
            # Save the args_spec for flat_tensor_args to unflatten while tracing
            _, tensor_args_spec = pytree.tree_flatten((tensor_args, kwargs))
            out_spec = PytreeThunk()

            def flat_fn(*flat_tensor_args):
                # The input are flattened tensor args. Prepare the args in the
                # order that original function expects. Add static args as well.
                # They will appear as tensor constants in the traced graph.
                nonlocal out_spec, static_args

                tensor_args, kwargs = pytree.tree_unflatten(
                    flat_tensor_args, tensor_args_spec
                )
                if static_argnums is None:
                    args = tensor_args
                else:
                    args = rearrange(tensor_args, static_args, static_argnums)
                tree_out = fn(*args, **kwargs)
                flat_out, spec = pytree.tree_flatten(tree_out)
                for i in flat_out:
                    is_known_type = False
                    for j in KNOWN_TYPES:
                        if isinstance(i, j):
                            is_known_type = True
                            break
                    if not is_known_type:
                        raise RuntimeError(
                            f"Found {type(i)} in output, which is not a known type. "
                            "If this type holds tensors, you need to register a pytree for it. "
                            "See https://github.com/pytorch/functorch/issues/475 for a brief "
                            "explanation why. If you don't need to register a pytree, please "
                            "leave a comment explaining your use case and we'll make this more "
                            "ergonomic to deal with"
                        )
                out_spec.set(spec)
                return flat_out

            compiled_fn = create_aot_dispatcher_function(
                flat_fn,
                flat_tensor_args,
                aot_config,
            )
            cached_res = (compiled_fn, out_spec)

            # Save the compiled_fn in the cache
            compile_cache.insert(
                fn_id,
                fw_compiler_id,
                bw_compiler_id,
                num_tensor_args,
                hasher_type,
                cached_res,
                *flat_args_for_cache,
            )

        cached_fn, out_spec = cached_res
        out = cached_fn(flat_tensor_args)
        return out_spec.unflatten(out)

    return returned_function


def num_of_recompilations():
    """
    Returns the numbers of recompilations since the last time cache was cleared.
    This is equivalent to the number of entries in the compilation cache.
    """
    global compile_cache
    if compile_cache is None:
        return 0
    return compile_cache.size()


def clear_compile_cache():
    """
    Clears the compilation cache.
    """
    global compile_cache
    if compile_cache is not None:
        compile_cache.clear()
        compile_cache = None


def aot_module(mod: nn.Module, *args, **kwargs) -> nn.Module:
    """
    Traces the forward and backward graph of :attr:`mod` using torch dispatch
    tracing mechanism. It is wrapper function, that underneath uses
    :func:`aot_function` to perform tracing and compilation.

    :func:`aot_module` lifts the parameters and buffers of ``nn.Module`` as inputs
    to a new callable which is then compiled through :func:`aot_function`.

    .. warning::
        This API is experimental and likely to change.

    Args:
        mod (Callable): A ``nn.Module`` module.
        args : args to be passed to :func:`aot_function`
        kwargs : kwargs to be passed to :func:`aot_function`

    Returns:
        Returns a ``nn.Module`` that retains the eager behavior of the original
        :attr:`mod`, but with forward and backward graph compiled.

    """

    def functional_call(named_params, named_buffers, *args, **kwargs):
        params_and_buffers = {**named_params, **named_buffers}
        return stateless.functional_call(mod, params_and_buffers, args, kwargs)

    compiled_f = aot_function(functional_call, *args, **kwargs)

    class AOTModule(nn.Module):
        def __init__(self):
            super(AOTModule, self).__init__()
            self.orig_module = mod

        def forward(self, *args, **kwargs):
            return compiled_f(
                dict(_named_parameters(mod, remove_duplicate=False)),
                dict(_named_buffers(mod, remove_duplicate=False)),
                *args,
                **kwargs,
            )

    return AOTModule()


def aot_module_simplified(mod: nn.Module, *top_args, **top_kwargs) -> nn.Module:
    """
    This is the simplified or low overhead version of aot_module. For frontends
    like TorchDynamo, the input functions/modules to AOT are static and have
    unpacked inputs/outputs. This gives us an opportunity to remove the
        (1) pytree overhead to parse inputs/outputs,
        (2) AOT Autograd cache,
        (3) Reading of params/buffers in every forward call

    :func:`aot_module_simplified` removes these overheads.
    """
    #########################################################

    params = {
        **dict(_named_parameters(mod, remove_duplicate=False)),
        **dict(_named_buffers(mod, remove_duplicate=False)),
    }
    params_flat, params_spec = pytree.tree_flatten(params)
    params_flat = tuple(params_flat)
    params_len = len(params_flat)

    def functional_call(*args, **kwargs):
        with stateless._reparametrize_module(
            mod, pytree.tree_unflatten(args[:params_len], params_spec)
        ):
            if isinstance(mod, torch.fx.GraphModule):
                with fx_traceback.override_stack_trace(), torch.autograd.detect_anomaly(
                    check_nan=False
                ):
                    out = Interpreter(mod).run(*args[params_len:], **kwargs)
            else:
                out = mod(*args[params_len:], **kwargs)

        if not isinstance(out, (tuple, list)):
            raise RuntimeError(
                "Graph output must be a tuple(). This is so that we can avoid "
                "pytree processing of the ouputs. Please change the module to "
                "have tuple outputs or use aot_module instead."
            )
        return out

    def aot_function_simplified(
        fn: Callable,
        fw_compiler: Callable,
        bw_compiler: Optional[Callable] = None,
        partition_fn: Callable = default_partition,
        decompositions: Optional[Dict] = None,
        hasher_type: str = "StaticShapeHasher",
        static_argnums: Optional[Tuple[int]] = None,
    ) -> Callable:
        assert static_argnums is None
        if bw_compiler is None:
            bw_compiler = fw_compiler
        aot_config = AOTConfig(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=partition_fn,
            decompositions=decompositions,
        )

        compiled_fn = None

        @wraps(fn)
        def new_func(*args):
            nonlocal compiled_fn
            if compiled_fn is None:
                compiled_fn = create_aot_dispatcher_function(
                    fn,
                    args,
                    aot_config,
                )
            return compiled_fn(args)

        return new_func

    compiled_f = aot_function_simplified(functional_call, *top_args, **top_kwargs)

    if top_kwargs:

        def forward(*args, **kwargs):
            return compiled_f(
                *params_flat,
                *args,
                **kwargs,
            )

    else:

        def forward(*args):
            return compiled_f(
                *params_flat,
                *args,
            )

    forward.zero_grad = mod.zero_grad
    return forward


compiled_function = aot_function
compiled_module = aot_module
