# mypy: allow-untyped-defs
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NoReturn,
    Sequence,
    Tuple,
    Type,
    Union,
)

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils._named_member_accessor import NamedMemberAccessor

# Utilities to make nn.Module "functional"
# In particular the goal is to be able to provide a function that takes as input
# the parameters and evaluate the nn.Module using fixed inputs.


def raise_parameter_tying_error() -> NoReturn:
    raise RuntimeError(
        "make_functional(module): we don't yet support models that "
        "do parameter tying (also sometimes known as weight sharing). "
        "Please try to rewrite your model by replacing all instances of the "
        "tied parameter with another and/or comment your support in "
        "https://github.com/pytorch/functorch/issues/446"
    )


def create_names_map(
    named_params: Union[Dict[str, Tensor], Iterable[Tuple[str, Tensor]]],
    tied_named_params: Union[Dict[str, Tensor], Iterable[Tuple[str, Tensor]]],
) -> Dict[str, List[str]]:
    """
    named_params is a dictionary of tensors: {'A': A, 'B': B}
    tied_named_params is another dictionary of tensors {'A': A, 'B': B, 'B_tied': B}
    with potentially tied (or 'duplicated') tensors

    This function creates a mapping from the names in named_params to the
    names in tied_named_params: {'A': ['A'], 'B': ['B', 'B_tied']}.
    """
    named_params = dict(named_params)
    tied_named_params = dict(tied_named_params)

    tensors_dict_keys = set(named_params.keys())
    tied_tensors_dict_keys = set(tied_named_params.keys())
    assert tensors_dict_keys.issubset(tied_tensors_dict_keys)

    tensor_to_mapping: Dict[Tensor, Tuple[str, List[str]]] = {}
    for key, tensor in named_params.items():
        tensor_to_mapping[tensor] = (key, [])
    for key, tensor in tied_named_params.items():
        assert tensor in tensor_to_mapping
        tensor_to_mapping[tensor][1].append(key)
    return dict(tensor_to_mapping.values())


def _extract_members(
    mod: nn.Module,
    named_members: Callable[..., Iterable[Tuple[str, Tensor]]],
    subclass: Callable[[Tensor], Tensor],
) -> Tuple[Tuple[Tensor, ...], Tuple[str, ...], Dict[str, List[str]]]:
    all_named_members = tuple(named_members(remove_duplicate=False))
    unique_named_members = tuple(named_members(remove_duplicate=True))
    names_map = create_names_map(unique_named_members, all_named_members)

    # Remove all the members in the model
    memo = {}
    accessor = NamedMemberAccessor(mod)
    for name, p in all_named_members:
        if p not in memo:
            memo[p] = subclass(torch.empty_like(p, device="meta"))
        replacement = memo[p]
        accessor.set_tensor(name, replacement)

    if len(unique_named_members) == 0:
        names, params = (), ()
    else:
        names, params = zip(*unique_named_members)  # type: ignore[assignment]
    return params, names, names_map


def extract_weights(
    mod: nn.Module,
) -> Tuple[Tuple[Tensor, ...], Tuple[str, ...], Dict[str, List[str]]]:
    """
    This function removes all the Parameters from the model and
    return them as a tuple as well as their original attribute names.
    The weights must be re-loaded with `load_weights` before the model
    can be used again.
    Note that this function modifies the model in place and after this
    call, mod.parameters() will be empty.
    """
    return _extract_members(mod, mod.named_parameters, nn.Parameter)


def extract_buffers(
    mod: nn.Module,
) -> Tuple[Tuple[Tensor, ...], Tuple[str, ...], Dict[str, List[str]]]:
    return _extract_members(mod, mod.named_buffers, lambda x: x)


def load_weights(
    mod: nn.Module,
    names: Sequence[str],
    params: Sequence[Tensor],
    as_params: bool = False,
) -> None:
    """
    Reload a set of weights so that `mod` can be used again to perform a forward pass.
    Note that the `params` are regular Tensors (that can have history) and so are left
    as Tensors. This means that mod.parameters() will still be empty after this call.
    """
    accessor = NamedMemberAccessor(mod)
    if as_params:
        params = [nn.Parameter(p) for p in params]
    accessor.set_tensors(names, params)


def _swap_state(
    mod: nn.Module, names_map: Dict[str, List[str]], elems: Iterable[Tensor]
) -> List[Tensor]:
    result: List[Tensor] = []
    accessor = NamedMemberAccessor(mod)
    for (_, attr_names), elem in zip(names_map.items(), elems):
        for i, attr_name in enumerate(attr_names):
            if i == 0:
                result.append(accessor.swap_tensor(attr_name, elem))
            else:
                accessor.set_tensor(attr_name, elem)
    return result


def load_buffers(
    mod: nn.Module,
    names: Sequence[str],
    buffers: Sequence[Tensor],
    as_params: bool = False,
) -> None:
    accessor = NamedMemberAccessor(mod)
    accessor.set_tensors(names, buffers)


def load_state(
    model: nn.Module,
    weights: Sequence[Tensor],
    weight_names: Sequence[str],
    buffers: Sequence[Tensor] = (),
    buffer_names: Sequence[str] = (),
) -> nn.Module:
    """load_state(model, weights, weight_names, buffers=(), buffer_names=()) -> model

    load_state takes `weights` and `buffers` and assigns them to the model.
    This is the inverse operation of `make_functional_deprecated_v1`.
    """
    assert len(weight_names) == len(weights)
    load_weights(model, weight_names, weights)
    if len(buffers) > 0:
        assert len(buffer_names) == len(buffers)
        load_buffers(model, buffer_names, buffers)
    return model


def make_functional_deprecated_v1(model: nn.Module):
    """make_functional_deprecated_v1(model) -> weights, func, weight_names

    Given an nn.Module, make_functional_deprecated_v1 extracts the state (weights)
    and returns a functional version of the model, `func`. This makes
    it so that it is possible use transforms over the parameters of
    `model`.

    `func` can be invoked as follows:
    ```
    x = torch.randn(4, 3)
    model = nn.Linear(3, 3)
    weights, func, _ = make_functional_deprecated_v1(model)
    func(weights, (x,))
    ```

    And here is an example of applying the grad transform:
    ```
    x = torch.randn(4, 3)
    model = nn.Linear(3, 3)
    weights, _, func = make_functional_deprecated_v1(model)
    grad_weights = grad(func)(weights, (x,))
    ```

    To put the state back into a model, use `load_state`.
    """
    buffers = list(model.buffers())
    if len(buffers) > 0:
        raise RuntimeError(
            "make_functional_deprecated_v1(model): `model` has buffers. Please use "
            "make_functional_with_buffers_deprecated_v1(model) instead."
        )
    weights, descriptors, _ = extract_weights(model)

    def fun(weights, data):
        mutable_model = copy.deepcopy(model)
        load_weights(mutable_model, descriptors, weights)
        return mutable_model(*data)

    return weights, fun, descriptors


def make_functional_with_buffers_deprecated_v1(model: nn.Module):
    """make_functional_with_buffers_deprecated_v1(model) -> weights, buffers, func, weight_names, buffer_names

    Given an nn.Module, make_functional_with_buffers_deprecated_v1 extracts the state (weights and buffers)
    and returns a functional version of the model, `func`.

    `func` can be invoked as follows:
    ```
    x = torch.randn(4, 3)
    model = nn.Linear(3, 3)
    weights, buffers, func, _, _ = make_functional_with_buffers_deprecated_v1(model)
    func(weights, buffers, (x,))
    ```

    And here is an example of applying the grad transform:
    ```
    x = torch.randn(4, 3)
    model = nn.Linear(3, 3)
    weights, buffers, func, _, _ = make_functional_with_buffers_deprecated_v1(model)
    func(weights, buffers, (x,))
    grad_weights = grad(func)(weights, buffers, (x,))
    ```

    To put the state back into a model, use `load_state`.
    """
    weights, weight_descriptors, _ = extract_weights(model)
    buffers, buf_descriptors, _ = extract_buffers(model)

    def fun(weights, buffers, data):
        mutable_model = copy.deepcopy(model)
        load_weights(mutable_model, weight_descriptors, weights)
        load_buffers(mutable_model, buf_descriptors, buffers)
        return mutable_model(*data)

    return weights, buffers, fun, weight_descriptors, buf_descriptors


class FunctionalModuleWithBuffers(nn.Module):
    """
    This is the callable object returned by :func:`make_functional_with_buffers`.
    """

    def __init__(
        self,
        stateless_model: nn.Module,
        param_names: Tuple[str, ...],
        buffer_names: Tuple[str, ...],
        param_names_map: Dict[str, List[str]],
        buffer_names_map: Dict[str, List[str]],
    ) -> None:
        super().__init__()
        self.stateless_model = stateless_model
        self.param_names = param_names
        self.buffer_names = buffer_names

        self.all_names_map = dict(param_names_map)
        self.all_names_map.update(buffer_names_map)

    @staticmethod
    def _create_from(
        model: nn.Module, disable_autograd_tracking: bool = False
    ) -> Tuple["FunctionalModuleWithBuffers", Tuple[Tensor, ...], Tuple[Tensor, ...]]:
        # TODO: We don't need to copy the model to create a stateless copy
        model_copy = copy.deepcopy(model)
        params, param_names, param_names_map = extract_weights(model_copy)
        buffers, buffer_names, buffer_names_map = extract_buffers(model_copy)
        if disable_autograd_tracking:
            for param in params:
                param.requires_grad_(False)
        return (
            FunctionalModuleWithBuffers(
                model_copy, param_names, buffer_names, param_names_map, buffer_names_map
            ),
            params,
            buffers,
        )

    def forward(
        self, params: Iterable[Tensor], buffers: Iterable[Tensor], *args, **kwargs
    ) -> Any:
        # Temporarily load the state back onto self.stateless_model
        old_state = _swap_state(
            self.stateless_model,
            self.all_names_map,
            tuple(params) + tuple(buffers),
        )
        try:
            return self.stateless_model(*args, **kwargs)
        finally:
            # Remove the loaded state on self.stateless_model
            _swap_state(self.stateless_model, self.all_names_map, old_state)


class FunctionalModule(nn.Module):
    """
    This is the callable object returned by :func:`make_functional`.
    """

    def __init__(
        self,
        stateless_model: nn.Module,
        param_names: Tuple[str, ...],
        names_map: Dict[str, List[str]],
    ) -> None:
        super().__init__()
        self.stateless_model = stateless_model
        self.param_names = param_names
        self.names_map = names_map

    @staticmethod
    def _create_from(
        model: nn.Module, disable_autograd_tracking: bool = False
    ) -> Tuple["FunctionalModule", Tuple[Tensor, ...]]:
        # TODO: We don't need to copy the model to create a stateless copy
        model_copy = copy.deepcopy(model)
        params, param_names, names_map = extract_weights(model_copy)
        if disable_autograd_tracking:
            for param in params:
                param.requires_grad_(False)
        return FunctionalModule(model_copy, param_names, names_map), params

    def forward(self, params: Iterable[Tensor], *args, **kwargs) -> Any:
        # Temporarily load the state back onto self.stateless_model
        old_state = _swap_state(self.stateless_model, self.names_map, params)
        try:
            return self.stateless_model(*args, **kwargs)
        finally:
            # Remove the loaded state on self.stateless_model
            _swap_state(self.stateless_model, self.names_map, old_state)


def make_functional(
    model: nn.Module, disable_autograd_tracking: bool = False
) -> Tuple[FunctionalModule, Tuple[Tensor, ...]]:
    """make_functional(model, disable_autograd_tracking=False) -> func, params

    Given a ``torch.nn.Module``, :func:`make_functional` extracts the state
    (params) and returns a functional version of the model, ``func``. This
    makes it so that it is possible use transforms over the parameters of
    ``model``.

    ``func`` can be invoked as follows:

    .. code-block:: python

        import torch
        import torch.nn as nn
        from functorch import make_functional

        x = torch.randn(4, 3)
        model = nn.Linear(3, 3)
        func, params = make_functional(model)
        func(params, x)

    And here is an example of applying the grad transform over the parameters
    of a model.

    .. code-block:: python

        import torch
        import torch.nn as nn
        from functorch import make_functional, grad

        x = torch.randn(4, 3)
        t = torch.randn(4, 3)
        model = nn.Linear(3, 3)
        func, params = make_functional(model)

        def compute_loss(params, x, t):
            y = func(params, x)
            return nn.functional.mse_loss(y, t)

        grad_weights = grad(compute_loss)(params, x, t)

    If the model has any buffers, please use :func:`make_functional_with_buffers` instead.

    Args:
        model (torch.nn.Module): Input model.
        disable_autograd_tracking (bool): Flag to disable gradients tracking for output parameters.
            The returned params are unrelated to the set of params from the original model. If False (default),
            the params will have ``requires_grad=True`` on them (aka they will be trackable with regular
            PyTorch autograd), matching the requires_grad-ness of the params from the original model.
            Otherwise, the returned params will have ``requires_grad=False``. Default, False.
            If you plan on using regular PyTorch autograd (e.g., if you want to call ``.backward()`` or
            ``torch.autograd.grad()``, then set ``disable_autograd_tracking=False``.
            Otherwise, if you're only planning on using functorch's gradient transforms,
            then please set ``disable_autograd_tracking=True`` to avoid unnecessarily tracking
            history with PyTorch autograd.

    """
    buffers = list(model.buffers())
    if len(buffers) > 0:
        raise RuntimeError(
            "make_functional(model): `model` has buffers. Please use "
            "make_functional_with_buffers(model) instead."
        )
    return FunctionalModule._create_from(
        model, disable_autograd_tracking=disable_autograd_tracking
    )


def make_functional_with_buffers(
    model: nn.Module, disable_autograd_tracking: bool = False
) -> Tuple[FunctionalModuleWithBuffers, Tuple[Tensor, ...], Tuple[Tensor, ...]]:
    """make_functional_with_buffers(model, disable_autograd_tracking=False) -> func, params, buffers

    Given a ``torch.nn.Module``, make_functional_with_buffers extracts the
    state (params and buffers) and returns a functional version of the model
    ``func`` that can be invoked like a function.

    ``func`` can be invoked as follows:

    .. code-block:: python

        import torch
        import torch.nn as nn
        from functorch import make_functional_with_buffers

        x = torch.randn(4, 3)
        model = nn.Linear(3, 3)
        func, params, buffers = make_functional_with_buffers(model)
        func(params, buffers, x)

    And here is an example of applying the grad transform over the parameters
    of a model:

    .. code-block:: python

        import torch
        import torch.nn as nn
        from functorch import make_functional_with_buffers, grad

        x = torch.randn(4, 3)
        t = torch.randn(4, 3)
        model = nn.Linear(3, 3)
        func, params, buffers = make_functional_with_buffers(model)

        def compute_loss(params, buffers, x, t):
            y = func(params, buffers, x)
            return nn.functional.mse_loss(y, t)

        grad_weights = grad(compute_loss)(params, buffers, x, t)

    Args:
        model (torch.nn.Module): Input model.
        disable_autograd_tracking (bool): Flag to disable gradients tracking for output parameters.
            The returned params are unrelated to the set of params from the original model. If False (default),
            the params will have ``requires_grad=True`` on them (aka they will be trackable with regular
            PyTorch autograd), matching the requires_grad-ness of the params from the original model.
            Otherwise, the returned params will have ``requires_grad=False``. Default, False.
            If you plan on using regular PyTorch autograd (e.g., if you want to call ``.backward()`` or
            ``torch.autograd.grad()``, then set ``disable_autograd_tracking=False``.
            Otherwise, if you're only planning on using functorch's gradient transforms,
            then please set ``disable_autograd_tracking=True`` to avoid unnecessarily tracking
            history with PyTorch autograd.

    """
    return FunctionalModuleWithBuffers._create_from(
        model, disable_autograd_tracking=disable_autograd_tracking
    )


def transpose_stack(
    tuple_of_tuple_of_tensors: Tuple[Tuple[Tensor, ...], ...]
) -> Tuple[Tensor, ...]:
    tuple_of_tuple_of_tensors = tuple(zip(*tuple_of_tuple_of_tensors))
    results = tuple(
        torch.stack(shards).detach() for shards in tuple_of_tuple_of_tensors
    )
    return results


def combine_state_for_ensemble(
    models: Sequence[nn.Module],
) -> Tuple[FunctionalModuleWithBuffers, Tuple[Tensor, ...], Tuple[Tensor, ...]]:
    """combine_state_for_ensemble(models) -> func, params, buffers

    Prepares a list of torch.nn.Modules for ensembling with :func:`vmap`.

    Given a list of ``M`` ``nn.Modules`` of the same class, stacks all of their
    parameters and buffers together to make ``params`` and ``buffers``.
    Each parameter and buffer in the result will have an additional dimension
    of size ``M``.

    :func:`combine_state_for_ensemble` also returns ``func``, a functional
    version of one of the models in :attr:`models`. One cannot directly run
    ``func(params, buffers, *args, **kwargs)`` directly, you probably want to
    use ``vmap(func, ...)(params, buffers, *args, **kwargs)``

    Here's an example of how to ensemble over a very simple model:

    .. code-block:: python

        num_models = 5
        batch_size = 64
        in_features, out_features = 3, 3
        models = [torch.nn.Linear(in_features, out_features) for i in range(num_models)]
        data = torch.randn(batch_size, 3)

        fmodel, params, buffers = combine_state_for_ensemble(models)
        output = vmap(fmodel, (0, 0, None))(params, buffers, data)

        assert output.shape == (num_models, batch_size, out_features)

    .. warning::
        All of the modules being stacked together must be the same (except for
        the values of their parameters/buffers). For example, they should be in the
        same mode (training vs eval).

        This API is subject to change -- we're investigating better ways to
        create ensembles and would love your feedback how to improve this.
    """
    if len(models) == 0:
        raise RuntimeError(
            "combine_state_for_ensemble: Expected at least one model, got 0."
        )
    if not (all(m.training for m in models) or all(not m.training for m in models)):
        raise RuntimeError(
            "combine_state_for_ensemble: Expected all models to "
            "have the same training/eval mode."
        )
    model0_typ = type(models[0])
    if not all(type(m) == model0_typ for m in models):
        raise RuntimeError(
            "combine_state_for_ensemble: Expected all models to be of the same class."
        )
    funcs, params, buffers = zip(
        *[make_functional_with_buffers(model) for model in models]
    )
    params = transpose_stack(params)
    buffers = transpose_stack(buffers)
    return funcs[0], params, buffers


def functional_init(
    model_class: Type[nn.Module],
    ensemble_shape: Union[Tuple[()], Tuple[int]] = (),
    device: torch.types.Device = "cpu",
):
    def wrapped(*args, **kwargs):
        if len(ensemble_shape) >= 2:
            raise ValueError("NYI: ensemble_shape with more than 1 element")
        if len(ensemble_shape) == 0:
            model = model_class(*args, **kwargs).to(device)
            return make_functional_deprecated_v1(model)
        num_models = ensemble_shape[0]  # type: ignore[misc]
        if num_models <= 0:
            raise ValueError(f"num_models {num_models} should be > 0")
        # NB: Not very efficient, more of a POC
        models = tuple(
            model_class(*args, **kwargs).to(device) for _ in range(num_models)
        )
        _, fn, names = make_functional_deprecated_v1(model_class(*args, **kwargs))
        weights = tuple(make_functional_deprecated_v1(model)[0] for model in models)
        weights = tuple(zip(*weights))
        weights = tuple(torch.stack(shards).detach() for shards in weights)
        return weights, fn, names

    return wrapped


def functional_init_with_buffers(
    model_class: Type[nn.Module],
    ensemble_shape: Union[Tuple[()], Tuple[int]] = (),
    device: torch.types.Device = "cpu",
):
    def wrapped(*args, **kwargs):
        if len(ensemble_shape) >= 2:
            raise ValueError("NYI: ensemble_shape with more than 1 element")
        if len(ensemble_shape) == 0:
            model = model_class(*args, **kwargs).to(device)
            return make_functional_deprecated_v1(model)
        num_models = ensemble_shape[0]  # type: ignore[misc]
        if num_models <= 0:
            raise ValueError(f"num_models {num_models} should be > 0")
        # NB: Not very efficient, more of a POC
        models = tuple(
            model_class(*args, **kwargs).to(device) for _ in range(num_models)
        )
        (
            _,
            _,
            fn,
            weight_names,
            buffer_names,
        ) = make_functional_with_buffers_deprecated_v1(model_class(*args, **kwargs))
        weights, buffers = zip(
            *tuple(
                make_functional_with_buffers_deprecated_v1(model)[:2]
                for model in models
            )
        )
        weights = tuple(zip(*weights))
        weights = tuple(torch.stack(shards).detach() for shards in weights)
        buffers = tuple(zip(*buffers))
        buffers = tuple(torch.stack(shards).detach() for shards in buffers)
        return weights, buffers, fn, weight_names, buffer_names

    return wrapped
