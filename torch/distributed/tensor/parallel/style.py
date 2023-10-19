# Copyright (c) Meta Platforms, Inc. and affiliates
import functools
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Tuple, Union

import torch
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard
from torch.distributed._tensor.placement_types import Placement
from torch.distributed.tensor.parallel._utils import (
    _deprecate_warnings,
    _prepare_input_validate,
    _prepare_output_validate,
    _PrepareInputType,
    _PrepareOutputType,
    LayoutsType,
)

__all__ = [
    "ParallelStyle",
    "RowwiseParallel",
    "ColwiseParallel",
    "PairwiseParallel",
    "PrepareModuleInput",
    "PrepareModuleOutput",
    "SequenceParallel",
    "make_input_replicate_1d",
    "make_input_reshard_replicate",
    "make_input_shard_1d",
    "make_input_shard_1d_last_dim",
    "make_sharded_output_tensor",
    "make_output_replicate_1d",
    "make_output_reshard_tensor",
    "make_output_tensor",
    "make_output_shard_1d",
]


class ParallelStyle(ABC):
    """
    The parallel style user wants the module or submodule to be parallelized.
    Users can extend this class to build their own parallel style with customized input/output preparations.

    .. warning::
        ``_prepare_input`` and ``_prepare_output`` are only for internal usage and we will
        remove them from ctor soon. Please use ``input_layouts`` and ``output_layouts`` instead.
    """

    _prepare_input: _PrepareInputType
    _prepare_output: _PrepareOutputType
    input_layouts: LayoutsType
    output_layouts: LayoutsType
    use_local_output: bool

    @abstractmethod
    def __init__(
        self,
        _prepare_input,
        _prepare_output,
        *,
        input_layouts,
        output_layouts,
        use_local_output,
    ) -> None:
        self.input_layouts = input_layouts
        self.output_layouts = output_layouts
        self.use_local_output = use_local_output
        self._prepare_input = _prepare_input  # type: ignore[assignment, misc]
        self._prepare_output = _prepare_output  # type: ignore[assignment, misc]


class PairwiseParallel(ParallelStyle):
    """
    PairwiseParallel concatenate colwise and rowwise styles as a fixed
    pair like what Megatron-LM(https://arxiv.org/abs/1909.08053) is doing.
    We assume both input and output need to be replicate DTensors.

    .. warning::
        PairwiseParallel can be decomposed into ColwiseParallel and RowwiseParallel.
        We recommend users to directly use latter instead and we are deprecating this
        style and will remove it soon.
    """

    def __init__(
        self,
        _prepare_input=None,
        _prepare_output=None,
        *,
        input_layouts=None,
        output_layouts=None,
        use_local_output=True,
    ) -> None:
        _deprecate_warnings(
            "PairwiseParallel", "Use ColwiseParallel and RowwiseParallel instead."
        )
        _prepare_input = (
            make_input_replicate_1d if _prepare_input is None else _prepare_input
        )
        _prepare_output = (
            make_output_tensor if _prepare_output is None else _prepare_output
        )
        super().__init__(
            _prepare_input,
            _prepare_output,
            input_layouts=input_layouts,
            output_layouts=output_layouts,
            use_local_output=use_local_output,
        )


class SequenceParallel(PairwiseParallel):
    """
    SequenceParallel concatenate colwise and rowwise styles as a fixed
    pair together with sequence parallel like what Megatron-LM Sequence parallel
    (https://arxiv.org/pdf/2205.05198.pdf) is doing.
    We assume both input and output need to be sharded DTensors.

    .. warning::
        SequenceParallel can be decomposed into ColwiseParallel and RowwiseParallel.
        We recommend users to directly use latter instead and we are deprecating this
        style and will remove it soon.
    """

    def __init__(
        self,
        _prepare_input=None,
        _prepare_output=None,
        *,
        input_layouts=None,
        output_layouts=None,
        use_local_output=True,
    ) -> None:
        _deprecate_warnings(
            "SequenceParallel", "Use ColwiseParallel and RowwiseParallel instead."
        )
        super().__init__(  # type: ignore[misc]
            make_input_reshard_replicate,
            make_output_reshard_tensor,
            input_layouts=input_layouts,
            output_layouts=output_layouts,
            use_local_output=use_local_output,
        )


@_prepare_input_validate  # type: ignore[arg-type] # pyre-ignore[56]
def make_input_shard_1d(
    input: Union[torch.Tensor, DTensor],
    device_mesh: Optional[DeviceMesh] = None,
    dim: int = 0,
) -> DTensor:
    """
    .. warning::
        This method was deprecated and please specify ``input_layouts`` instead.
    """
    _deprecate_warnings("make_input_shard_1d", "Specify input_layouts instead.")
    shard_spec = [Shard(dim)]
    if isinstance(input, DTensor):
        return input.redistribute(device_mesh, shard_spec)
    elif isinstance(input, torch.Tensor):
        return DTensor.from_local(input, device_mesh, shard_spec, run_check=False)
    else:
        raise RuntimeError(
            "Tensor parallel module expects torch.Tensor or DTensor input but"
            f" received {type(input)}!"
        )


@_prepare_input_validate  # type: ignore[arg-type] # pyre-ignore[56]
def make_input_shard_1d_last_dim(
    input: Union[torch.Tensor, DTensor],
    device_mesh: Optional[DeviceMesh] = None,
) -> DTensor:
    """
    .. warning::
        This method was deprecated and please specify ``input_layouts`` instead.
    """
    _deprecate_warnings(
        "make_input_shard_1d_last_dim", "Specify input_layouts instead."
    )
    return make_input_shard_1d(input, device_mesh, dim=input.dim() - 1)  # type: ignore[call-arg, misc]


@_prepare_input_validate  # type: ignore[arg-type] # pyre-ignore[56]
def make_input_reshard_replicate(
    input: torch.Tensor,
    device_mesh: DeviceMesh,
) -> DTensor:
    """
    .. warning::
        This method was deprecated and please specify ``input_layouts`` instead.
    """
    _deprecate_warnings(
        "make_input_reshard_replicate", "Specify input_layouts instead."
    )
    return make_input_replicate_1d(  # type: ignore[call-arg, misc]
        make_input_shard_1d(input, device_mesh, dim=0), device_mesh  # type: ignore[call-arg, misc]
    )


@_prepare_input_validate  # type: ignore[arg-type] # pyre-ignore[56]
def make_input_replicate_1d(
    input: Union[torch.Tensor, DTensor],
    device_mesh: Optional[DeviceMesh] = None,
) -> DTensor:
    """
    .. warning::
        This method was deprecated and please specify ``input_layouts`` instead.
    """
    _deprecate_warnings("make_input_replicate_1d", "Specify input_layouts instead.")
    replicate = [Replicate()]
    if isinstance(input, DTensor):
        return input.redistribute(device_mesh, replicate)
    elif isinstance(input, torch.Tensor):
        return DTensor.from_local(input, device_mesh, replicate, run_check=False)
    else:
        raise RuntimeError(
            "Tensor parallel module expects torch.Tensor or DTensor input but"
            f" received {type(input)}!"
        )


@_prepare_output_validate  # type: ignore[arg-type] # pyre-ignore[56]
def make_output_shard_1d(
    output: DTensor, device_mesh: Optional[DeviceMesh] = None, dim: int = 0
) -> DTensor:
    """
    .. warning::
        This method was deprecated and please specify ``output_layouts`` instead.
    """
    _deprecate_warnings("make_output_shard_1d", "Specify output_layouts instead.")
    return output.redistribute(device_mesh, [Shard(dim)])


@_prepare_output_validate  # type: ignore[arg-type] # pyre-ignore[56]
def make_output_replicate_1d(
    output: DTensor, device_mesh: Optional[DeviceMesh] = None
) -> DTensor:
    """
    .. warning::
        This method was deprecated and please specify ``output_layouts`` instead.
    """
    _deprecate_warnings("make_output_replicate_1d", "Specify output_layouts instead.")
    return output.redistribute(device_mesh, [Replicate()])


@_prepare_output_validate  # type: ignore[arg-type] # pyre-ignore[56]
def make_output_tensor(
    output: DTensor, device_mesh: Optional[DeviceMesh] = None
) -> torch.Tensor:
    """
    .. warning::
        This method was deprecated and please specify ``output_layouts`` instead.
    """
    _deprecate_warnings("make_output_tensor", "Specify output_layouts instead.")
    return make_output_replicate_1d(  # type: ignore[attr-defined, misc]
        output, device_mesh
    ).to_local()  # type: ignore[call-arg]


@_prepare_output_validate  # type: ignore[arg-type] # pyre-ignore[56]
def make_sharded_output_tensor(
    output: DTensor, _device_mesh: Optional[DeviceMesh] = None
) -> torch.Tensor:
    """
    .. warning::
        This method was deprecated and please specify ``output_layouts`` instead.
    """
    _deprecate_warnings("make_sharded_output_tensor", "Specify output_layouts instead.")
    return output.to_local()  # type: ignore[call-arg]


@_prepare_output_validate  # type: ignore[arg-type] # pyre-ignore[56]
def make_output_reshard_tensor(
    output: DTensor,
    device_mesh: Optional[DeviceMesh] = None,
) -> torch.Tensor:
    """
    .. warning::
        This method was deprecated and please specify ``output_layouts`` instead.
    """
    _deprecate_warnings("make_output_reshard_tensor", "Specify output_layouts instead.")
    return make_output_shard_1d(output, device_mesh).to_local()  # type: ignore[call-arg, attr-defined, misc]


def _needs_redistribute(
    dst_placements: Tuple[Placement, ...], dtensor: DTensor
) -> bool:
    """
    Check DTensor placements to decide whether the DTensor redistribute
    is needed to be called or not. If not, we can directly early return
    and save CPU overhead.
    """
    return dtensor._spec.placements == dst_placements


def _get_prepare_input(
    input_layouts: LayoutsType, output_layouts: LayoutsType
) -> Callable[[Any], Any]:
    """
    Get the prepare input function for this parallel style.
    """

    def _redistribute_per_both_layouts(t, input_layout, output_layout, device_mesh):
        dst_placements = (output_layout,)
        if isinstance(t, DTensor):
            return (
                t
                if _needs_redistribute(dst_placements, t)
                else t.redistribute(device_mesh, dst_placements)
            )
        elif isinstance(t, torch.Tensor):
            dtensor = DTensor.from_local(
                t, device_mesh, [input_layout], run_check=False
            )
            return (
                dtensor
                if _needs_redistribute(dst_placements, dtensor)
                else dtensor.redistribute(device_mesh, dst_placements)
            )
        else:
            if input_layout is not None:
                raise RuntimeError(
                    "Tensor parallel module expects DTensor or tensor"
                    f" when layout specified but received {type(t)}!"
                )
            else:
                return t

    def make_input_redistribute_1d(
        input_layouts: LayoutsType,
        output_layouts: LayoutsType,
        inputs: Tuple[Any, ...],
        device_mesh: Optional[DeviceMesh] = None,
    ) -> Optional[Any]:
        """
        Redistribute input tensor over an 1-D device mesh. This function will be used in ParallelStyle.

        Args:
            input (Union[:class:`torch.Tensor`, :class:`DTensor`]):
                This input tensor will be replicated over the 1-D :class:`DeviceMesh`.
            device_mesh (:class:`DeviceMesh`, optional):
                The 1-D device mesh where ``input`` will be replicated.
                If no :class:`DeviceMesh` is passed and ``input`` is a :class:`DTensor`,
                ``input.device_mesh`` will be used.
                If :class:`DeviceMesh` is not 1-D, an exception will be thrown.
                Default: ``None``

        Returns:
            A :class:`DTensor` replicated over ``device_mesh``.
        """
        # Early return to save CPU overhead when there is only one input.
        if not isinstance(inputs, tuple):
            return _redistribute_per_both_layouts(
                inputs, input_layouts, output_layouts, device_mesh
            )

        if not isinstance(input_layouts, tuple):
            input_layouts = (input_layouts,)  # type: ignore[assignment]
            output_layouts = (output_layouts,)  # type: ignore[assignment]
        results = []
        for input, input_layout, output_layout in zip(
            inputs, input_layouts, output_layouts  # type: ignore[arg-type]
        ):
            results.append(
                _redistribute_per_both_layouts(
                    input, input_layout, output_layout, device_mesh
                )
            )
        return tuple(results)

    return functools.partial(make_input_redistribute_1d, input_layouts, output_layouts)


def _get_prepare_output(
    output_layouts: LayoutsType, use_local_output: bool
) -> Callable[[Any], Any]:
    """
    Get the prepare input function for this parallel style.
    """

    def _redistribute_per_layout(t, layout, device_mesh, use_local_output):
        dst_placements = (layout,)
        if isinstance(t, DTensor):
            dtensor = (
                t
                if _needs_redistribute(dst_placements, t)
                else t.redistribute(device_mesh, dst_placements)
            )
            return dtensor.to_local() if use_local_output else dtensor
        else:
            if layout is not None:
                raise RuntimeError(
                    "Tensor parallel module expects DTensor or tensor"
                    f" when layout specified but received {type(t)}!"
                )
            else:
                return t

    def make_output_redistribute_1d(
        output_layouts: LayoutsType,
        use_local_output: bool,
        outputs: Tuple[Any, ...],
        device_mesh: Optional[DeviceMesh] = None,
    ) -> Optional[Any]:
        """
        Redistribute input tensor over an 1-D device mesh. This function will be used in ParallelStyle.

        Args:
            input (Union[:class:`torch.Tensor`, :class:`DTensor`]):
                This input tensor will be replicated over the 1-D :class:`DeviceMesh`.
            device_mesh (:class:`DeviceMesh`, optional):
                The 1-D device mesh where ``input`` will be replicated.
                If no :class:`DeviceMesh` is passed and ``input`` is a :class:`DTensor`,
                ``input.device_mesh`` will be used.
                If :class:`DeviceMesh` is not 1-D, an exception will be thrown.
                Default: ``None``

        Returns:
            A :class:`DTensor` replicated over ``device_mesh``.
        """
        # Early return to save CPU overhead when there is only one output.
        if not isinstance(outputs, tuple):
            return _redistribute_per_layout(
                outputs, output_layouts, device_mesh, use_local_output
            )

        if not isinstance(output_layouts, tuple):
            output_layouts = (output_layouts,)  # type: ignore[assignment]
        results = []
        for output, output_layout in zip(outputs, output_layouts):  # type: ignore[arg-type]
            results.append(
                _redistribute_per_layout(
                    output, output_layout, device_mesh, use_local_output
                )
            )
        return tuple(results)

    return functools.partial(
        make_output_redistribute_1d, output_layouts, use_local_output
    )


class RowwiseParallel(ParallelStyle):
    """
    Partitioning the row of a module.
    We assume the input to be a sharded :class:`DTensor` and output to be a :class:`torch.Tensor`.

    Args:
        input_layouts (Union[Placement, Tuple[Placement, ...]]):
            The layout of input tensor(s) which DTensor will be created upon.
        output_layouts (Union[Placement, Tuple[Placement, ...]]):
            The layout of input tensor(s) which created DTensor will be redistributed to.
        use_local_output (bool):
            Whether to convert the DTensor to local :class:`torch.Tensor`.

    Returns:
        None.

    .. warning::
        RowwiseParallel now only support ``nn.Linear``. Users can compose it with ColwiseParallel
        to achieve the sharding of more complicated modules.

    .. warning::
        ``_prepare_input`` and ``_prepare_output`` are only for internal usage and we will
        remove them from ctor soon. Please use ``input_layouts`` and ``output_layouts`` instead.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, PrepareModuleInput
        >>> ...
        >>> parallelize_plan = {
        >>>     "wo": RowwiseParallel(),   # The input of Linear will be converted to Sharded DTensor
        >>>                                # and we will return a replicate :class:`torch.Tensor` as output.
        >>>     ...
        >>> }
        >>> parallelize_module(
        >>>     module=block, # this can be a submodule or module
        >>>     ...,
        >>>     parallelize_plan=parallelize_plan,
        >>> )
        >>> ...
    """

    def __init__(
        self,
        _prepare_input=None,
        _prepare_output=None,
        *,
        input_layouts=Shard(-1),
        output_layouts=Replicate(),
        use_local_output=True,
    ) -> None:
        if isinstance(input_layouts, tuple) or isinstance(output_layouts, tuple):
            raise NotImplementedError(
                "RowwiseParallel only supports single input/output."
            )

        if _prepare_input is not None:
            prepare_input_fn = _prepare_input
        if input_layouts == Shard(-1):
            prepare_input_fn = make_input_shard_1d_last_dim
        else:
            prepare_input_fn = _get_prepare_input(
                input_layouts,
                Shard(-1),
            )

        if _prepare_output is not None:
            prepare_output_fn = _prepare_output
        elif output_layouts == Replicate():
            prepare_output_fn = make_output_tensor
        else:
            prepare_output_fn = _get_prepare_output(
                output_layouts,
                use_local_output,
            )

        super().__init__(
            input_layouts=input_layouts,
            output_layouts=output_layouts,
            use_local_output=use_local_output,
            _prepare_input=prepare_input_fn,
            _prepare_output=prepare_output_fn,
        )


class ColwiseParallel(ParallelStyle):
    """
    Partitioning the column of a tensor or module.
    We assume the input to be a replicated :class:`DTensor` and output to be a sharded :class:`torch.Tensor`.

    Args:
        input_layouts (Union[Placement, Tuple[Placement, ...]]):
            The layout of input tensor(s) which DTensor will be created upon.
        output_layouts (Union[Placement, Tuple[Placement, ...]]):
            The layout of input tensor(s) which created DTensor will be redistributed to.
        use_local_output (bool):
            Whether to convert the DTensor to local :class:`torch.Tensor`.

    Returns:
        None.

    .. warning::
        ColwiseParallel now only support ``nn.Linear`` and ``nn.Embedding``. Users can compose
        it with RowwiseParallel to achieve the sharding of more complicated modules.

    .. warning::
        ``_prepare_input`` and ``_prepare_output`` are only for internal usage and we will
        remove them from ctor soon. Please use ``input_layouts`` and ``output_layouts`` instead.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, PrepareModuleInput
        >>> ...
        >>> parallelize_plan = {
        >>>     "w1": ColwiseParallel(),   # The input of Linear will be converted to Replicated DTensor
        >>>                                # and we will return a sharded :class:`torch.Tensor` as output.
        >>>     ...
        >>> }
        >>> parallelize_module(
        >>>     module=block, # this can be a submodule or module
        >>>     ...,
        >>>     parallelize_plan=parallelize_plan,
        >>> )
        >>> ...
    """

    def __init__(
        self,
        _prepare_input=None,
        _prepare_output=None,
        *,
        input_layouts=Replicate(),
        output_layouts=Shard(-1),
        use_local_output=True,
    ) -> None:
        """

        """
        if isinstance(input_layouts, tuple) or isinstance(output_layouts, tuple):
            raise NotImplementedError(
                "ColwiseParallel only supports single input/output."
            )

        if _prepare_input is not None:
            prepare_input_fn = _prepare_input
        if input_layouts == Replicate():
            prepare_input_fn = make_input_replicate_1d
        else:
            prepare_input_fn = _get_prepare_input(
                input_layouts,
                Replicate(),
            )

        if _prepare_output is not None:
            prepare_output_fn = _prepare_output
        elif output_layouts == Shard(-1):
            prepare_output_fn = make_sharded_output_tensor
        else:
            prepare_output_fn = _get_prepare_output(
                output_layouts,
                use_local_output,
            )

        super().__init__(
            input_layouts=input_layouts,
            output_layouts=output_layouts,
            use_local_output=use_local_output,
            _prepare_input=prepare_input_fn,
            _prepare_output=prepare_output_fn,
        )


class PrepareModuleInput(ParallelStyle):
    """
    :class:`PrepareModuleInput` enables users to annotate :class:`torch.Tensor` or :class:`DTensor`
    inputs with ``input_layouts`` and ``output_layouts`` so that each input can be converted to
    :class:`DTensor` based on the annotation. Specifically, a DTensor will be created
    from the input Tensor based on ``input_layouts`` and then redistributed to another
    DTensor based on ``output_layouts``.

    When the input is not a :class:`torch.Tensor` or :class:`DTensor`, if no layout is
    specified, it will be a no-op. Otherwise, it will throw an error.
    """

    def __init__(
        self,
        input_layouts: LayoutsType = Shard(0),
        output_layouts: LayoutsType = Replicate(),
        use_local_output: bool = False,
    ) -> None:
        """
        Args:
            input_layouts (Union[Placement, Tuple[Placement, ...]]):
                The layout of input tensor(s) which DTensor will be created upon.
            output_layouts (Union[Placement, Tuple[Placement, ...]]):
                The layout of input tensor(s) which created DTensor will be redistributed to.
            use_local_output (bool):
                Whether to convert the DTensor to local :class:`torch.Tensor`.

        Returns:
            None.

        Example::
            >>> # xdoctest: +SKIP(failing)
            >>> from torch.distributed.tensor.parallel import parallelize_module, PrepareModuleInput
            >>> ...
            >>> parallelize_plan = {
            >>>     "attn": PrepareModuleInput(),   # The input of attn will be converted to Sharded DTensor
            >>>                                     # and and redistributed to Replicated DTensor.
            >>>     ...
            >>> }
            >>> parallelize_module(
            >>>     module=block, # this can be a submodule or module
            >>>     ...,
            >>>     parallelize_plan=parallelize_plan,
            >>> )
            >>> ...
        """
        super().__init__(
            input_layouts=input_layouts,
            output_layouts=output_layouts,
            use_local_output=use_local_output,
            _prepare_input=_get_prepare_input(
                input_layouts,
                output_layouts,
            ),
            _prepare_output=None,
        )


class PrepareModuleOutput(ParallelStyle):
    """
    :class:`PrepareModuleOutput` enables users to annotate :class:`DTensor` outputs
    with ``output_layouts`` and ``use_local_output`` so that each output can be converted to
    :class:`DTensor` or :class:`torch.Tensor` based on the annotation. Specifically, a DTensor
    will be redistributed to another DTensor based on ``output_layouts`` and the flag ``use_local_output``
    to decide whether to convert the DTensor to local :class:`torch.Tensor`.

    When the output is not a :class:`DTensor`, if no layout is specified, it will be
    a no-op. Otherwise, it will throw an error.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, PrepareModuleOutput
        >>> ...
        >>> parallelize_plan = {
        >>>     "submodule": PrepareModuleOutput(),   # The output of submodule will be converted to Replicated DTensor
        >>>                                           # if it's not a DTensor, then redistributed to Sharded local tensor
        >>>     ...
        >>> }
        >>> parallelize_module(
        >>>     module=block, # this can be a submodule or module
        >>>     ...,
        >>>     parallelize_plan=parallelize_plan,
        >>> )
        >>> ...
    """

    def __init__(
        self,
        input_layouts: LayoutsType = Replicate(),
        output_layouts: LayoutsType = Shard(0),
        use_local_output: bool = True,
    ) -> None:
        super().__init__(
            input_layouts=input_layouts,
            output_layouts=output_layouts,
            use_local_output=use_local_output,
            _prepare_input=None,
            _prepare_output=_get_prepare_output(output_layouts, use_local_output),
        )
