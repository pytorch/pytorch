from functools import wraps
from typing import Callable, Dict, Optional, Tuple

import torch.utils._pytree as pytree
from torch._functorch.aot_autograd import (
    AOT_COUNTER,
    KNOWN_TYPES,
    AOTConfig,
    PytreeThunk,
    create_aot_dispatcher_function,
    default_partition,
)


def patched_aot_function(
    fn: Callable[..., object],
    fw_compiler: Callable[..., object],
    bw_compiler: Optional[Callable[..., object]] = None,
    partition_fn: Callable[..., object] = default_partition,
    decompositions: Optional[Dict[object, object]] = None,
    num_params_buffers: int = 0,
    hasher_type: object = None,  # deprecated
    static_argnums: Optional[Tuple[int]] = None,  # deprecated
    keep_inference_input_mutations: bool = False,
    pre_compile_fn: Optional[Callable[..., object]] = None,
) -> Callable[..., object]:
    """
    NOTE: rationale for patch.
        We want to do the following
            trace single device graph  --> parallelize (SPMD) ---> run graph on a shard

        But::
           - "single device graph" expects fully-sized shapes (e.g. logical shapes)
           - "parallelized graph" expects sharded shapes (e.g. physical local shapes)

        This means that we need to pass in "logical tensors" as input to the capturing step,
        but then we need to pass "physical local_shard tensors" as input to the parallelized
        graph afterwards.

        This patch allows to transform the inputs of the graph before compilation, so that
        we can capture the graph with logical shapes, and then finally after compilation,
        call into the compiled (and transformed) graph with the original sharded tensors.

        Beyond that:

            The compilation for the backwards pass doesn't follow the same pattern.
            For the backwards pass, since the compilation happens at first usage, we won't
            be able to intercept the compilation call from here. But that's fine, because
            the graph was already captured before with logical-shapes.


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
    properties, to detect when there is a need of recompilation.

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
        >>> aot_fn = patched_aot_function(fn, print_compile_fn)
        >>> x = torch.randn(4, 5, requires_grad=True)
        >>> aot_fn(x)
    """
    if static_argnums is not None:
        raise RuntimeError(
            "static_argnums has been deprecated - manually wrap your function or use torchdynamo."
        )

    if bw_compiler is None:
        bw_compiler = fw_compiler

    aot_config = AOTConfig(
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        partition_fn=partition_fn,
        # pyre-fixme
        decompositions=decompositions,  # type:ignore[arg-type]
        num_params_buffers=num_params_buffers,
        aot_id=next(AOT_COUNTER),
        keep_inference_input_mutations=keep_inference_input_mutations,
    )
    cached_res = None

    @wraps(fn)
    # pyre-fixme
    def returned_function(*args, **kwargs):
        nonlocal cached_res
        # Now flatten the tensor args
        flat_args, _ = pytree.tree_flatten((args, kwargs))

        # Compile the function and save it in the cache
        if cached_res is None:
            # Save the args_spec for flat_tensor_args to unflatten while tracing
            _, tensor_args_spec = pytree.tree_flatten((args, kwargs))
            out_spec = PytreeThunk()

            # pyre-fixme
            def flat_fn(*flat_args):
                # The input are flattened tensor args. Prepare the args in the
                # order that original function expects. Add static args as well.
                # They will appear as tensor constants in the traced graph.
                nonlocal out_spec
                args, kwargs = pytree.tree_unflatten(
                    list(flat_args),
                    tensor_args_spec,
                )
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

            compile_flat_args = (
                pre_compile_fn(flat_args)
                if pre_compile_fn is not None
                else flat_args
            )

            compiled_fn = create_aot_dispatcher_function(
                flat_fn,
                compile_flat_args,
                aot_config,
            )
            cached_res = (compiled_fn, out_spec)

        cached_fn, out_spec = cached_res
        out = cached_fn(flat_args)
        return out_spec.unflatten(out)

    return returned_function
