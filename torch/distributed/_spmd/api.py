from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from copy import copy
from dataclasses import dataclass
from functools import partial, wraps
from typing import Any, Callable, cast, Dict, List, Optional, Set, Tuple, Union

from functorch import make_fx

import torch
import torch.distributed as dist

# We need to import _functional_collectives to trigger op registration
import torch.distributed._functional_collectives
import torch.nn as nn
import torch.utils._pytree as pytree

from torch import fx
from torch._decomp.decompositions import native_layer_norm_backward

from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._spmd.data_parallel import gradients_tagging
from torch.distributed._spmd.parallel_mode import (
    DataParallel,
    DTensorExpandMode,
    ParallelMode,
)
from torch.distributed._tensor import Placement
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo, CodeGen
from torch.nn.utils import stateless
from torch.nn.utils._named_member_accessor import NamedMemberAccessor


class Override(ABC):
    r"""
    Override the tracing and transformation behavior of :meth:`~torch.distributed._spmd.compile`.
    This is useful when any part of the model is not traceable or if you prefer
    to not trace it due to any reason. More specifically, users can implement
    :meth:`torch.distributed._spmd.Override.replacement` to replace an original
    submodule with the return new submodule. The new submodule contains
    operations that users preferred to be traced, which simply be a dummy
    placeholder operator. After tracing, users can implement
    :meth:`torch.distributed._spmd.Override.transform` to transform the traced
    graph, where the dummy placeholder operator serves as an anchor to insert
    new sub-graphs.
    """

    @abstractmethod
    def replacement(self, fqn: str, orig_submodule: torch.nn.Module) -> torch.nn.Module:
        r"""
        Implement this method to return a new :class:`nn.Module` instance to
        replace the ``orig_submodule`` argument in the model. This helps if
        ``orig_submodule`` is not traceable or should not be traced.

        Args:
            fqn (str): fully quantified name of the submodule.
            orig_submodule (class:`nn.Module`): original submodule instance to replace.

        Returns:
            A new :class:`nn.Module` instance to replace the original one.
        """
        pass

    @abstractmethod
    def transform(
        self,
        gm: fx.GraphModule,
        flat_state: List[torch.Tensor],
    ) -> fx.GraphModule:
        r"""
        Given a DTensor-expanded graph and sharding schema for every node,
        conduct additional transformation for the sub-graph from the :class:`nn.Module`
        returned by :meth:`torch.distributed._spmd.Override.replacement` if
        necessary.

        Args:
            gm (:class:`fx.Graph`): a DTensor-expanded graph.
            flat_state (List[str, :class:`Tensor`]): a reference to the list of
                flattened state. The elements in ``flat_state`` map to the first
                ``len(flat_state)`` placeholders in the graph. The transformation
                can add state to or remove state from ``flat_state`` as long as
                it keeps ``flat_state`` and the placeholders consistent.

        Returns:
            The :class:`fx.Graph` after transformation.
        """
        pass


class _PyTreeCodeGenOutputsOnly(_PyTreeCodeGen):
    # pyre-ignore[3]
    def process_inputs(self, *args: Any) -> Any:
        return args

    # pyre-ignore[2, 3]
    def gen_fn_def(self, free_vars, maybe_return_annotation):
        return CodeGen.gen_fn_def(self, free_vars, maybe_return_annotation)


def _to_caller_flattened_graph_module(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Move the responsibility of flattening the input arguments from the
    graph module to the caller.

    Example:

        output = gm(my_struct)

        gm = gm(to_caller_flattened_graph_module)

        output = gm(*pytree.flatten(my_struct)[0])
    """
    # pyre-ignore[16]
    gm._graph._codegen = _PyTreeCodeGenOutputsOnly(
        pytree_info=_PyTreeInfo(
            # pyre-ignore[6]
            orig_args=None,  # type: ignore[arg-type]
            # pyre-ignore[6]
            in_spec=None,  # type: ignore[arg-type]
            # pyre-ignore[16]
            out_spec=gm._graph._codegen.pytree_info.out_spec,
        )
    )
    gm.recompile()
    return gm


# Use a dtensor expand mode for now to preserve the old behavior
# and avoid breaking existing code
dtensor_expand_mode = DTensorExpandMode()


def _override_placements(t: torch.Tensor, placements: List[Placement]):
    global dtensor_expand_mode
    dtensor_expand_mode._placements_override[id(t)] = placements


@contextmanager
def _rematerialize_optimizer(
    opt: torch.optim.Optimizer,
    named_states: Dict[str, Any],
    params: Dict[str, nn.Parameter],
):
    assert opt is not None

    # update opt.state with proxy tensors
    orig_states: Dict[str, Any] = copy(opt.state)
    for n in named_states:
        # opt.state's key type is string, but optimizer uses Parameter as keys
        opt.state[params[n]] = named_states[n]  # type: ignore[index]

    # FIXME: support multiple parameter groups
    param_group = opt.param_groups[0]
    orig_params = param_group["params"]
    param_group["params"] = params.values()

    try:
        yield
    finally:
        param_group["params"] = orig_params
        opt.state = orig_states


aten = torch.ops.aten  # pyre-ignore


@contextmanager
def _enable_compile():
    # The return value of torch._utils.is_compiling changes optimizer behavior.
    # We need that function to return True to include optimizer in the graph.
    # See: https://github.com/pytorch/pytorch/blob/a524123c91ab399c9dd6882c1189596dd77e7734/torch/optim/optimizer.py#L41
    def f_true():
        return True

    orig_is_compiling_code = torch._utils.is_compiling.__code__
    torch._utils.is_compiling.__code__ = f_true.__code__
    try:
        yield
    finally:
        torch._utils.is_compiling.__code__ = orig_is_compiling_code


def _foreach_add_decomp(self, other, alpha=1):
    self_updated = aten._foreach_add.List(self, other, alpha=alpha)
    for s, s_u in zip(self, self_updated):
        s.copy_(s_u)


def _foreach_unaop_decomp(op, self):
    self_updated = op(self)
    for s, s_u in zip(self, self_updated):
        s.copy_(s_u)


def _foreach_binop_list_decomp(op, self, other):
    self_updated = op(self, other)
    for s, s_u in zip(self, self_updated):
        s.copy_(s_u)


def _foreach_binop_scalar_decomp(op, self, scalar=1):
    self_updated = op(self, scalar)
    for s, s_u in zip(self, self_updated):
        s.copy_(s_u)


def _foreach_addcop_scalar_decomp(op, self, tensor1, tensor2, scalar=1):
    self_updated = op(self, tensor1, tensor2, scalar)
    for s, s_u in zip(self, self_updated):
        s.copy_(s_u)


def _fused_adam_decomp(
    self,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    *,
    lr=1,
    beta1=1,
    beta2=1,
    weight_decay=1,
    eps=1,
    amsgrad=True,
    maximize=True,
    grad_scale=None,
    found_inf=None,
):
    orig_tuple = (self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs)
    updated_tuple = aten._fused_adam.default(
        self,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
        eps=eps,
        amsgrad=amsgrad,
        maximize=maximize,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )

    for idx, (orig, updated) in enumerate(zip(orig_tuple, updated_tuple)):
        if idx == 1:
            # skip gradient copying as we don't need to copy gradients back
            continue
        for o, u in zip(orig, updated):
            o.copy_(u)


SPMD_DECOMP_TABLE = {
    aten._foreach_add_.List: _foreach_add_decomp,
    aten._foreach_add_.Scalar: partial(
        _foreach_binop_scalar_decomp, aten._foreach_add.Scalar
    ),
    aten._foreach_addcdiv_.Scalar: partial(
        _foreach_addcop_scalar_decomp, aten._foreach_addcdiv.Scalar
    ),
    aten._foreach_addcmul_.Scalar: partial(
        _foreach_addcop_scalar_decomp, aten._foreach_addcmul.Scalar
    ),
    aten._foreach_div_.List: partial(
        _foreach_binop_list_decomp, aten._foreach_div.List
    ),
    aten._foreach_mul_.Scalar: partial(
        _foreach_binop_scalar_decomp, aten._foreach_mul.Scalar
    ),
    aten._foreach_neg_.default: partial(
        _foreach_unaop_decomp, aten._foreach_neg.default
    ),
    aten._foreach_reciprocal_.default: partial(
        _foreach_unaop_decomp, aten._foreach_reciprocal.default
    ),
    aten._foreach_sub_.Scalar: partial(
        _foreach_binop_scalar_decomp, aten._foreach_sub.Scalar
    ),
    aten._fused_adam_.default: _fused_adam_decomp,
    aten.native_layer_norm_backward.default: native_layer_norm_backward,
}


DEDUP_TARGETS: Set[torch._ops.OpOverload] = {
    torch.ops.c10d_functional.all_reduce.default,
    torch.ops.c10d_functional.wait_tensor.default,
}


def _dedup_collectives(gm: fx.GraphModule) -> fx.GraphModule:
    args_to_node: Dict[Tuple[Any, ...], fx.Node] = {}

    for node in gm.graph.nodes:
        # replace all args with the results from the first unique comm op
        args, _ = pytree.tree_flatten(node.args)

        if node.target in DEDUP_TARGETS:
            args_key = (node.target, *args)
            unique_node = args_to_node.get(args_key, None)
            if unique_node is None:
                # first time seeing this combination, remember it
                args_to_node[args_key] = node
            else:
                # the current node is a duplicate, replace it
                node.replace_all_uses_with(unique_node)
                gm.graph.erase_node(node)

    gm.recompile()

    return gm


@dataclass
class _CompiledResult:
    gm: fx.GraphModule
    mod: nn.Module
    opt: Optional[torch.optim.Optimizer]
    flat_state: List[torch.Tensor]


def _compile(
    func: Callable,
    module_override: Optional[List[Override]],
    parallel_mode: ParallelMode,
    *args: Any,
    **kwargs: Any,
) -> _CompiledResult:
    # 1. Extract nn.Module and Optimizer from args and kwargs
    # FIXME(@mrshenli): support multiple nn.Module instances
    # FIXME(@mrshenli): support multiple Optiimzer instances
    # FIXME(@mrshenli): need to broadcast model to sync parameters
    mod, opt = None, None
    for arg in pytree.tree_flatten(list(args) + list(kwargs.values()))[0]:
        if isinstance(arg, nn.Module):
            assert mod is None, "Only support single nn.Module for now"
            mod = arg
        if isinstance(arg, torch.optim.Optimizer):
            assert opt is None, "Only support single Optimizer for now"
            opt = arg

    assert mod is not None, "Couldn't find nn.Module instances from the arguments."

    # 2. Override target submodules (e.g., MoE) with dummy replacements
    if module_override:
        accessor = NamedMemberAccessor(mod)

        def swap(fqn_prefix: str, module: torch.nn.Module) -> None:
            for override in module_override:  # type: ignore[union-attr]
                for name, child in module.named_children():
                    if len(name) == 0:
                        continue
                    fqn = fqn_prefix + "." + name if fqn_prefix != "" else name
                    new_child = override.replacement(fqn, child)
                    if id(new_child) == id(child):
                        swap(fqn, new_child)
                    else:
                        accessor.swap_submodule(fqn, new_child)

        swap("", mod)

    # 3. Trace statelss version of the train_step
    params = dict(mod.named_parameters(remove_duplicate=False))
    buffers = dict(mod.named_buffers(remove_duplicate=False))

    named_states = {}
    if opt is not None:
        # Pass named_states instead of opt.state to stateless_func, because
        # the later uses nn.Parameter as key. During tracing, we need to
        # make sure optimizers can find the states using proxy tensors.
        for n, p in params.items():
            if p in opt.state:
                # opt.state's key type is string, but optimizer uses
                # Parameter as keys
                named_states[n] = opt.state[p]  # type: ignore[index]

    is_data_parallel_mode = isinstance(parallel_mode, DataParallel)

    # Lift states and parameters as function arguments so that make_fx
    # can trace operations applied to them.
    def stateless_func(func, params, buffers, named_states, args, kwargs):
        with stateless._reparametrize_module(
            cast(nn.Module, mod), {**params, **buffers}
        ), _rematerialize_optimizer(
            opt, named_states, params
        ) if opt else nullcontext():
            # For DataParallel mode, install hooks first to tag the gradients
            with gradients_tagging(params) if is_data_parallel_mode else nullcontext():
                ret = func(*args, **kwargs)

            # make sure updated parameters are returned
            return ret, list(mod.parameters()), list(named_states.values())  # type: ignore[union-attr]

    # FIXME: Using symbolic tracing to work around in DTensor expand mode.
    # Otherwise it hits shape mismatch error, as we use local inputs to
    # trace local graph and use DTensor to expand operators, where
    # DTensor's shape is the global shape.
    tracing_mode = "fake" if is_data_parallel_mode else "symbolic"

    if is_data_parallel_mode:
        fake_mode = FakeTensorMode()
        data_parallel_mode = cast(DataParallel, parallel_mode)

        def _get_full_batch_arg(arg: torch.Tensor) -> torch.Tensor:
            # since compilation happens in the first iteration and we
            # receives mini-batch input, convert them to full batch
            # fake tensor input first for data parallel sharding
            # propagations
            fake_arg = fake_mode.from_tensor(arg)
            arg_dims = [1] * arg.ndim
            # expand the tensor to full batch size on its batch dim
            arg_dims[data_parallel_mode.input_batch_dim] *= dist.get_world_size()
            return fake_arg.repeat(arg_dims)

        args = pytree.tree_map_only(
            torch.Tensor,
            _get_full_batch_arg,
            args,
        )
        kwargs = pytree.tree_map_only(
            torch.Tensor,
            _get_full_batch_arg,
            kwargs,
        )

    with _enable_compile(), torch.autograd.detect_anomaly(check_nan=False):
        # FIXME(@mrshenli): functionalization does not work for our use
        # case yet. Use explicit decompositions for foreach ops.
        # Remove this when the following issue is addressed.
        # Issue: https://github.com/pytorch/pytorch/issues/97852
        gm = make_fx(
            partial(stateless_func, func),
            tracing_mode=tracing_mode,
            decomposition_table=SPMD_DECOMP_TABLE,
            _allow_non_fake_inputs=False,
        )(params, buffers, named_states, args, kwargs)

    params_and_buffers: Dict[str, Union[torch.Tensor, nn.Parameter]] = {
        **params,
        **buffers,
    }

    # 4. parallel mode to expand a single device graph to a distributed graph
    gm = parallel_mode.partition(
        gm,
        mod,
        opt,
        params_and_buffers,
        named_states,
        args,
        kwargs,
    )

    # 5. Move the responsibility of flattening the input arguments from the
    # graph module to the caller. This serves two purposes:
    #   - Transformations that add/remove state need to manipulate a state
    #   container that maintains the state tensors in the same order as they
    #   appear in graph placeholders.
    #   - Reduced runtime cost. The state container is only flattened once upfront.
    flat_state, _ = pytree.tree_flatten([params_and_buffers, named_states])
    gm = _to_caller_flattened_graph_module(gm)

    # 6. dedup comm operators.
    # The duplication could come from DTensor args and kwargs redistribution.
    # Suppose one operator produces a Partial gradient tensor and model
    # parameters are replicated. In this case, every optimizer operation using
    # that Partial gradient tensor would trigger an allreduce. This is becuase
    # DTensor only has local information on individual tensor/operator, which is
    # not sufficient to detect duplications in the graph. This situation can
    # also happen when inserting FSDP allgather if a parameter is used multiple
    # times in the forward method.
    # TODO(@mrshenli): @yifuwang has a suggestion of conducting expansion and
    # dedup at tracer-level to avoid multiple graph passes.
    gm = _dedup_collectives(gm)

    # 7. Replace previously inserted dummy ones with real graphs.
    if module_override:
        for override in module_override:
            gm = override.transform(gm, flat_state)

    return _CompiledResult(gm, mod, opt, flat_state)


# Note that the Python convention of __dict__ requires the key to be str.
# TODO: ensure the key is unique.
COMPILED_OBJECT_KEY = "_compiled_obj"


def compile(
    module_override: Optional[List[Override]] = None,
    gm_transformation: Optional[Callable[[fx.GraphModule], fx.GraphModule]] = None,
    parallel_mode: Optional[ParallelMode] = None,
):
    r"""
    Compile and optimize a callable, which can be a train step within a training
    loop. This method will extract :class:`nn.Module` and :class:`torch.optim.Optimizer`
    instances from the input arguments and trace operations applied to their
    parameters and states.

    Args:
        module_override (Optional[List[Override]]): a list of Override instances
            that will be applied to the module in order. The :class:`Override`
            objects provide :class:`nn.Module` replacements during tracing and a
            graph transformation function after tracing. (Default: ``None``)
        gm_transformation (Optional[Callable[fx.GraphModule, fx.GraphModule]]):
            a callback that will be called after the original callable is
            compiled and distributed (usually after the first iteration) to
            transform the compiled GraphModule into a new optimized one.
        parallel_mode (Optional[ParallelMode]): a :class:`ParallelMode` object
            that specifies how to parallelize the callable. Each ParallelMode
            would have its own strategy to partition the model and the captured
            graph (Default: ``None``)
    """

    def inner(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_train_step = kwargs.pop("last_train_step", False) if kwargs else False
            first_iter = False
            # Put the COMPILED_OBJECT_KEY in ``wrapper`` instead of ``func`` as
            # ``wrapper`` is the one that users will get.
            compiled_obj = wrapper.__dict__.get(COMPILED_OBJECT_KEY, None)
            if compiled_obj is None:
                first_iter = True
                global dtensor_expand_mode
                mode: ParallelMode = (
                    dtensor_expand_mode if parallel_mode is None else parallel_mode
                )

                compiled_obj = _compile(func, module_override, mode, *args, **kwargs)
                wrapper.__dict__[COMPILED_OBJECT_KEY] = compiled_obj

            flat_inps = compiled_obj.flat_state + pytree.tree_flatten([args, kwargs])[0]

            with torch.no_grad():
                # N.B.: we don't need autograd as backward has already been
                # captured in the graph.
                if first_iter and gm_transformation:
                    # TODO: SPMD should provid a default and configurable
                    # transformation.
                    compiled_obj.gm = gm_transformation(compiled_obj.gm)
                if not last_train_step:
                    output = compiled_obj.gm(*flat_inps)[0]
                else:
                    # This is the last train step. Call IterGraphModule.forward()
                    # with the `last_iter` argument and catch the exception in
                    # case the compiled_obj is not wrapped with IterGraphModule.
                    try:
                        output = compiled_obj.gm(*flat_inps, last_iter=last_train_step)[
                            0
                        ]
                    except TypeError as e:
                        if "last_iter" not in str(e):
                            raise e
                        output = compiled_obj.gm(*flat_inps)[0]

                return output

        return wrapper

    return inner
