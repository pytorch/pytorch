from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from copy import copy
from dataclasses import dataclass
from functools import partial, wraps
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

from functorch import make_fx

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils._pytree as pytree
from torch import fx
from torch.distributed._spmd.distribute import (
    _convert_to_distributed,
    distribute,
    Schema,
)
from torch.distributed._spmd.distributed_graph import DistributedGraph
from torch.distributed._tensor import DeviceMesh, Placement, Replicate, Shard
from torch.nn.utils import stateless
from torch.nn.utils._named_member_accessor import NamedMemberAccessor


class SPMD(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        schema: Schema,
        input_schemas: Sequence[Placement] = tuple(),
    ) -> None:
        """
        Given a non-distributed nn.Module, distribute the module and apply
        optimizations over the distributed module (fx.GraphModule).

        Args:
            module (nn.Module): The target module.
            schema (Schema): The distributed schema.
            input_schemas (Sequence[Placement]): The schemas of the inputs.
        """
        super().__init__()
        assert schema.placements == [
            Replicate()
        ], "SPMD only support Replicate() parameters for now"

        # TODO: Fix model initialization with coalescing.
        # This needs to happen post model transformation.
        # Consider an explicit model init API.
        for p in module.parameters():
            dist.broadcast(p, src=0)

        self._param_schema = schema
        self._input_schemas = input_schemas
        self._compiled_m: Optional[nn.Module] = None
        self._dist_graph = DistributedGraph(orig_module=module)

    def forward(self, *args: Tuple[object], **kwargs: Dict[str, object]) -> object:
        if self._compiled_m is None:
            self._compiled_m = distribute(
                self._dist_graph,
                self._param_schema,
                self._input_schemas,
                *args,
                **kwargs,
            )

        assert self._compiled_m is not None
        return self._compiled_m(*args, **kwargs)


class Override(ABC):
    r"""
    Override the tracing and transformation behavior of :meth:`~torch.distributed._spmd.compile`.
    This is useful when any part of the model is not traceable or if you prefer
    to not trace it due to any reason. More specifically, users can implement
    :meth:`torch.distributed._spmd.Override.replacement` to replace an original
    submodule with the return new submodule. The new submodule contrains
    operations that users preferred to be traced, which simply be a dummy
    placeholder operator. After tracing, users can implement
    :meth:`torch.distributed._spmd.Override.transform` to transform the traced
    graph, where the dummy placeholder operator serves as an anchor to insert
    new sub-graphs.
    """

    @abstractmethod
    def replacement(self, orig_submodule: torch.nn.Module) -> torch.nn.Module:
        r"""
        Implement this method to return a new :class:`nn.Module` instance to
        replace the ``orig_submodule`` argument in the model. This helps if
        ``orig_submodule`` is not traceable or should not be traced.

        Args:
            orig_submodule (class:`nn.Module`): original submodule instance to replace.

        Returns:
            A new :class:`nn.Module` instance to replace the original one.
        """
        pass

    @abstractmethod
    def transform(self, gm: fx.GraphModule, schema_map: Dict[str, Schema]) -> fx.Graph:
        r"""
        Given a DTensor-expanded graph and shardig schema for every node,
        conduct additional transformation for the sub-graph from the :class:`nn.Module`
        returned by :meth:`torch.distributed._spmd.Override.replacement` if
        necessary.

        Args:
            gm (:class:`fx.Graph`): a DTensor-expanded graph.
            schema_map (Dict[str, :class:`Schema`]): a dictionary maps from node
                name to DTensor schema.

        Returns:
            The :class:`fx.Graph` after transformation.
        """
        pass


def _dtensor_expand(
    gm: fx.GraphModule,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    named_states: Dict[str, Any],
    params_and_buffers: Dict[str, Any],
) -> Tuple[fx.GraphModule, Dict[str, Schema]]:
    flat_args, _ = pytree.tree_flatten(list(args) + list(kwargs.values()))

    mesh = DeviceMesh("cuda", torch.arange(dist.get_world_size()).cuda())
    shard_schema: Schema = Schema(mesh=mesh, placements=[Shard(0)])
    # FIXME: allow other sharding schemas
    replicate_schema: Schema = Schema(mesh=mesh, placements=[Replicate()])

    inps, schemas = [], []
    for a in flat_args:
        if isinstance(a, torch.Tensor):
            inps.append(a)
            schemas.append(shard_schema)
        elif isinstance(a, (nn.Module, torch.optim.Optimizer)):
            # nn.Module or optimizer placeholder is captured by make_fx but
            # never used in the graph
            inps.append(torch.empty(0))
            schemas.append(shard_schema)

    for o in pytree.tree_flatten(named_states)[0]:
        if isinstance(o, torch.Tensor):
            inps.append(o)
            schemas.append(replicate_schema)
        else:
            inps.append(torch.empty(0))
            schemas.append(replicate_schema)

    for p in pytree.tree_flatten(params_and_buffers)[0]:
        assert isinstance(p, torch.Tensor), f"expecting Tensor but got {type(p)}"
        inps.append(p)
        schemas.append(replicate_schema)

    return _convert_to_distributed(gm, inps, schemas, _allow_partial=False)


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
    # FIXME(@mrshenli): exclude buffers
    param_group["params"] = params.values()

    try:
        yield
    finally:
        param_group["params"] = orig_params
        opt.state.update(orig_states)


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

    for orig, updated in zip(orig_tuple, updated_tuple):
        for o, u in zip(orig, updated):
            o.copy_(u)


FOREACH_DECOMP_TABLE = {
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
}


DEDUP_TARGETS: Set[torch._ops.OpOverload] = {
    aten.all_reduce.default,
    aten.wait_tensor.default,
}


def _dedup_collectives(gm: fx.GraphModule) -> fx.GraphModule:
    # deduplicate collectives
    node_to_node: Dict[fx.Node, fx.Node] = {}
    args_to_node: Dict[Tuple[Any, ...], fx.Node] = {}

    nodes_to_erase: List[fx.Node] = []

    for node in gm.graph.nodes:
        # replace all args with the results from the first unique comm op
        args, spec = pytree.tree_flatten(node.args)
        unique_args = [node_to_node.get(a, a) for a in args]
        node.args = pytree.tree_unflatten(unique_args, spec)

        if node.target in DEDUP_TARGETS:
            args_key = (node.target, *unique_args)
            unique_node = args_to_node.get(args_key, None)
            if unique_node is None:
                # first time seeing this combination, remember it
                args_to_node[args_key] = node
            else:
                # the current node is a duplicate, replace it
                node_to_node[node] = unique_node
                nodes_to_erase.append(node)

    # erase all duplicated nodes
    for node in nodes_to_erase:
        gm.graph.erase_node(node)

    return gm


@dataclass
class _CompiledResult:
    gm: fx.GraphModule
    mod: nn.Module
    opt: Optional[torch.optim.Optimizer]
    named_states: Dict[str, torch.Tensor]
    params_and_buffers: Dict[str, torch.Tensor]


def _compile(
    func: Callable,
    module_override: Optional[Dict[Type[Any], Override]],
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

        for typ, override in module_override.items():
            for name, submodule in mod.named_modules():
                if isinstance(submodule, typ):
                    accessor.swap_submodule(name, override.replacement(submodule))

    # 3. Trace statelss version of the train_step
    params_and_buffers: Dict[str, Union[torch.Tensor, nn.Parameter]] = {
        **dict(mod.named_parameters(remove_duplicate=False)),
        **dict(mod.named_buffers(remove_duplicate=False)),
    }

    named_states = {}
    if opt is not None:
        opt_states, spec = pytree.tree_flatten(dict(opt.state))

        # Pass named_states instead of opt.state to stateless_func, because
        # the later uses nn.Parameter as key. During tracing, we need to
        # make sure optimizers can find the states using proxy tensors.
        for n, p in params_and_buffers.items():
            if p in opt.state:
                # opt.state's key type is string, but optimizer uses
                # Parameter as keys
                named_states[n] = opt.state[p]  # type: ignore[index]

    # Lift states and parameters as function arguments so that make_fx
    # can trace operations applied to them.
    def stateless_func(func, args, kwargs, named_states, params_and_buffers):
        with stateless._reparametrize_module(
            cast(nn.Module, mod), params_and_buffers
        ), _rematerialize_optimizer(
            opt, named_states, params_and_buffers
        ) if opt else nullcontext():
            ret = func(*args, **kwargs)
            # make sure updated parameters are returned
            return ret, list(mod.parameters())  # type: ignore[union-attr]

    # FIXME: Using symbolic tracing to work around. Otherwise it hits
    # shape mismatch error, as we use local inputs to trace local graph
    # and use DTensor to expand operators, where DTensor's shape is the
    # global shape.
    with _enable_compile():
        # FIXME(@mrshenli): functionalization does not work for our use
        # case yet. Use explicit decompositions for foreach ops.
        # Remove this when the following issue is addressed.
        # Issue: https://github.com/pytorch/pytorch/issues/97852
        gm = make_fx(
            partial(stateless_func, func),
            tracing_mode="symbolic",
            decomposition_table=FOREACH_DECOMP_TABLE,
            _allow_non_fake_inputs=False,
        )(args, kwargs, named_states, params_and_buffers)

    # 4. Use DTensor to insert collectives
    gm, name_to_spec = _dtensor_expand(
        gm, args, kwargs, named_states, params_and_buffers
    )

    # 5. dedup comm operators.
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

    # 6. Replace previously inserted dummy ones with real graphs.
    if module_override:
        for _, override in module_override.items():
            gm = override.transform(gm, name_to_spec)

    return _CompiledResult(gm, mod, opt, named_states, params_and_buffers)


# Note that the Python convention of __dict__ requires the key to be str.
# TODO: ensure the key is unique.
COMPILED_OBJECT_KEY = "_compiled_obj"


def compile(
    module_override: Optional[Dict[Type[Any], Override]] = None,
    gm_transformation: Optional[Callable[[fx.GraphModule], fx.GraphModule]] = None,
):
    r"""
    Compile and optimize a callable, which can be a train step within a training
    loop. This method will extract :class:`nn.Module` and :class:`torch.optim.Optimizer`
    instances from the input arguments and trace operations applied to their
    parameters and states.

    Args:
        module_override (Optional[Dict[Type[Any], Override]]): a dictionary maps
            from target :class:`nn.Module` types to :class:`Override` objects.
            The :class:`Override` objects provide :class:`nn.Module` replacements
            during tracing and a graph transformation function after tracing.
            (Default: ``None``)
        gm_transformation (Optional[Callable[fx.GraphModule, fx.GraphModule]]):
            a callback that will be called after the original callable is
            compiled and distributed (usually after the first iteration) to
            transform the compiled GraphModule into a new optimized one.
    """

    def inner(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            first_iter = False
            # Put the COMPILED_OBJECT_KEY in ``wrapper`` instead of ``func`` as
            # ``wrapper`` is the one that users will get.
            compiled_obj = wrapper.__dict__.get(COMPILED_OBJECT_KEY, None)
            if compiled_obj is None:
                first_iter = True
                compiled_obj = _compile(func, module_override, *args, **kwargs)
                wrapper.__dict__[COMPILED_OBJECT_KEY] = compiled_obj

            with torch.no_grad():
                # N.B.: we don't need autograd as backward has already been
                # captured in the graph.
                output = compiled_obj.gm(
                    args,
                    kwargs,
                    compiled_obj.named_states,
                    compiled_obj.params_and_buffers,
                )[0]
                if first_iter and gm_transformation:
                    # TODO: SPMD should provid a default and configurable
                    # transformation.
                    compiled_obj.gm = gm_transformation(compiled_obj.gm)
                return output

        return wrapper

    return inner
