from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from copy import copy
from functools import wraps, partial
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)

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
from torch.distributed._tensor import (
    DeviceMesh,
    Placement,
    Replicate,
    Shard,
)
from torch.nn.utils import stateless
from functorch import make_fx
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

    def forward(
        self, *args: Tuple[object], **kwargs: Dict[str, object]
    ) -> object:
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
    def transform(
        self, gm: fx.GraphModule, schema_map: Dict[str, Schema]
    ) -> fx.Graph:
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
        elif isinstance(a, nn.Module) or isinstance(a, torch.optim.Optimizer):
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
        assert isinstance(
            p, torch.Tensor
        ), f"expecting Tensor but got {type(p)}"
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


def compile(module_override: Optional[Dict[Type[Any], Override]] = None):
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
    """

    def inner(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 1. Extract nn.Module and Optimizer from args and kwargs
            # FIXME(@mrshenli): support multiple nn.Module instances
            # FIXME(@mrshenli): support multiple Optiimzer instances
            # FIXME(@mrshenli): need to broadcast model to sync parameters
            mod, opt = None, None
            for arg in pytree.tree_flatten(list(args) + list(kwargs.values()))[
                0
            ]:
                if isinstance(arg, nn.Module):
                    assert mod is None, "Only support single nn.Module for now"
                    mod = arg
                if isinstance(arg, torch.optim.Optimizer):
                    assert opt is None, "Only support single Optimizer for now"
                    opt = arg

            assert (
                mod is not None
            ), "Couldn't find nn.Module instances from the arguments."

            # 2. Override target submodules (e.g., MoE) with dummy replacements
            if module_override:
                accessor = NamedMemberAccessor(mod)

                for typ, override in module_override.items():
                    for name, submodule in mod.named_modules():
                        if isinstance(submodule, typ):
                            accessor.swap_submodule(
                                name, override.replacement(submodule)
                            )

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
            def stateless_func(
                func, args, kwargs, named_states, params_and_buffers
            ):
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
                # FIXME: functionalize crashes with
                # "UnsupportedFakeTensorException: meta converter nyi"
                gm = make_fx(
                    partial(stateless_func, func),
                    tracing_mode="symbolic",
                    _allow_non_fake_inputs=False,
                )(args, kwargs, named_states, params_and_buffers)

            # 4. Use DTensor to insert collectives
            gm, name_to_spec = _dtensor_expand(
                gm, args, kwargs, named_states, params_and_buffers
            )

            # 5. Replace previously inserted dummy ones with real graphs.
            if module_override:
                for _, override in module_override.items():
                    gm = override.transform(gm, name_to_spec)

            with torch.no_grad():
                # N.B.: we don't need autograd as backward has already been
                # captured in the graph.
                return gm(args, kwargs, named_states, params_and_buffers)[0]

        return wrapper

    return inner
