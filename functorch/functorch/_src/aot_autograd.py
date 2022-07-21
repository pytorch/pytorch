from contextlib import contextmanager, nullcontext
import torch
import torch.nn as nn
from torch import Tensor
from functorch import make_fx
from torch.fx import immutable_collections
from torch._subclasses import FakeTensorMode
import torch.utils._pytree as pytree
import torch.utils.dlpack
from torch.nn.utils import _stateless
from functorch._C import CompileCache
from functorch.experimental import functionalize
from . import config
from .decompositions import register_decomposition
from .partitioners import default_partition
from .named_members_polyfill import _named_parameters, _named_buffers
from typing import Callable, List, Dict, Any, Tuple, Optional
from functools import wraps

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

# TODO - move this to PyTorch core. This overrides the pytree implementation for
# dict to maintain parity with Deepmind pytree.
Context = Any


def _dict_flatten(d: Dict[Any, Any]) -> Tuple[List[Any], Context]:
    keys = sorted(d.keys())
    values = [d[key] for key in keys]
    return values, keys


def _dict_unflatten(values: List[Any], context: Context) -> Dict[Any, Any]:
    return {key: value for key, value in zip(context, values)}


pytree._register_pytree_node(dict, _dict_flatten, _dict_unflatten)

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
        backward_out = []
        # Call the backwards pass
        if grad_primals:
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


@register_decomposition(aten._reshape_alias, aot_autograd_decompositions)
def _reshape_alias(x, shape, strides):
    return aten.view(x, shape)


@register_decomposition(aten.new_zeros, aot_autograd_decompositions)
def new_zeros(inp, size, dtype=None, layout=None, device=None, pin_memory=None):
    return torch.zeros(size, dtype=inp.dtype, device=inp.device)


@register_decomposition(aten.new_full, aot_autograd_decompositions)
def new_full(inp, size, value, dtype=None, layout=None, device=None, pin_memory=None):
    return torch.full(size, value, dtype=inp.dtype, device=inp.device)


def create_aot_autograd_function(
    flat_fn, fw_compiler, bw_compiler, partition_fn, decompositions, grad_state
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
    if decompositions is None:
        decompositions = {}
    joint_forward_backward = create_joint_forward_backward(flat_fn)

    compiled_fw = None
    compiled_bw = None
    num_outs = None

    class CompiledFunction(torch.autograd.Function):
        @staticmethod
        @disable_torchdynamo
        def forward(ctx, *flat_tensor_args):
            nonlocal compiled_fw, compiled_bw, num_outs
            # Disable the JIT Autocast flag to prevent re-autocasting of jitted graph.
            # TODO - Remove when https://github.com/pytorch/functorch/pull/794 is fixed.
            old_jit_autocast_flag = torch._C._jit_set_autocast_mode(False)
            if compiled_fw is None:
                flat_tensor_args = pytree.tree_map(
                    lambda x: x.detach().requires_grad_(x.requires_grad)
                    if isinstance(x, Tensor) else x, flat_tensor_args
                )
                fake_mode = FakeTensorMode.push() if config.use_fake_tensor else nullcontext()
                with preserve_rng_state(), fake_mode as mode:
                    # Set input tensors that require grad to leaves
                    fake_flat_tensor_args = pytree.tree_map(
                        lambda x: mode.from_tensor(x) if mode else x
                        if isinstance(x, Tensor) else x, flat_tensor_args
                    )
                    with torch.set_grad_enabled(grad_state):
                        out = flat_fn(*fake_flat_tensor_args)
                    out = pytree.tree_map(
                        lambda x: x.detach().contiguous() if isinstance(x, Tensor) else x, out
                    )

                    if isinstance(out, (list, tuple)):
                        num_outs = len(out)
                    else:
                        num_outs = 1

                    joint_inputs = (fake_flat_tensor_args, out)
                    aot_decompositions = {**aot_autograd_decompositions, **decompositions}
                    with torch.set_grad_enabled(grad_state):
                        fx_g = make_fx(joint_forward_backward, aot_decompositions)(
                            *joint_inputs
                        )

                        if config.use_functionalize:
                            # Functionalize the foward backward graph. First create a
                            # fake fn to make functionalize happy
                            def fake_fn(primals, tangents):
                                return fx_g(primals, tangents)
                            fx_g = make_fx(functionalize(fake_fn))(*joint_inputs)
                fw_module, bw_module = partition_fn(fx_g, joint_inputs)
                # print(fw_module.code, bw_module.code)

                compiled_fw = fw_compiler(fw_module, flat_tensor_args)
                fw_outs = normalize_as_list(compiled_fw(*flat_tensor_args))

                bw_args = fw_outs[num_outs:] + fw_outs[0:num_outs]
                compiled_bw = bw_compiler(bw_module, bw_args)
            else:
                fw_outs = normalize_as_list(compiled_fw(*flat_tensor_args))
            torch._C._jit_set_autocast_mode(old_jit_autocast_flag)
            ctx.save_for_backward(*fw_outs[num_outs:])
            return tuple(fw_outs[0:num_outs])

        @staticmethod
        @disable_torchdynamo
        def backward(ctx, *flat_args):
            # Disable the JIT Autocast flag to prevent re-autocasting of jitted graph.
            # TODO - Remove when https://github.com/pytorch/functorch/pull/794 is fixed.
            old_jit_autocast_flag = torch._C._jit_set_autocast_mode(False)
            contiguous_args = [t.contiguous() for t in flat_args]
            # contiguous_args = [t for t in flat_args]
            out = normalize_as_list(compiled_bw(*ctx.saved_tensors, *contiguous_args))
            torch._C._jit_set_autocast_mode(old_jit_autocast_flag)
            return tuple(out)

    return CompiledFunction


class _CompileCache(CompileCache):
    pass


# using a C++-based pytree reduces the overhead by about 50%
try:
    import tree

    HAS_TREE = True
except ImportError:
    HAS_TREE = False
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
        if HAS_TREE:
            flat_tensor_args = tree.flatten((tensor_args, kwargs))
        else:
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

            compiled_fn = create_aot_autograd_function(
                flat_fn,
                fw_compiler,
                bw_compiler,
                partition_fn,
                decompositions,
                grad_state=torch.is_grad_enabled(),
            ).apply
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
        out = cached_fn(*flat_tensor_args)
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
        return _stateless.functional_call(mod, params_and_buffers, args, kwargs)

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
        with _stateless.reparametrize_module(
            mod, pytree.tree_unflatten(args[:params_len], params_spec)
        ):
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
        compiled_fn = create_aot_autograd_function(
            fn,
            fw_compiler,
            bw_compiler,
            partition_fn,
            decompositions,
            grad_state=torch.is_grad_enabled(),
        ).apply

        return compiled_fn

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
