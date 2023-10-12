# Copyright (c) Meta Platforms, Inc. and affiliates
import functools
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Tuple, Union

import torch
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard
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
    """

    _prepare_input: _PrepareInputType
    _prepare_output: _PrepareOutputType
    input_layouts: LayoutsType
    output_layouts: LayoutsType
    use_local: bool

    @abstractmethod
    def __init__(
        self, input_layouts, output_layouts, use_local, _prepare_input, _prepare_output
    ) -> None:
        self.input_layouts = input_layouts
        self.output_layouts = output_layouts
        self.use_local = use_local
        self._prepare_input = _prepare_input  # type: ignore[assignment, misc]
        self._prepare_output = _prepare_output  # type: ignore[assignment, misc]


class PairwiseParallel(ParallelStyle):
    """
    PairwiseParallel concatenate colwise and rowwise styles as a fixed
    pair like what Megatron-LM(https://arxiv.org/abs/1909.08053) is doing.
    We assume both input and output need to be replicate DTensors.

    .. warning::
        PairwiseParallel does not support ``nn.MultiheadAttention``,
        ``nn.Transformer`` well at this moment. One workaround is to apply
        ``ColwiseParallel`` and ``RowwiseParallel`` to the components of
        transformer. We recommend to use ``PairwiseParallel`` only
        for even-number-layer MLP for now.
    """
    @_deprecate_warnings("Use ColwiseParallel and RowwiseParallel instead.")  # type: ignore[misc]
    def __init__(
        self,
        input_layouts=None,
        output_layouts=None,
        use_local=True,
        _prepare_input=None,
        _prepare_output=None,
    ) -> None:
        _prepare_input = (
            make_input_replicate_1d if _prepare_input is None else _prepare_input
        )
        _prepare_output = (
            make_output_tensor if _prepare_output is None else _prepare_output
        )
        super().__init__(
            input_layouts, output_layouts, use_local, _prepare_input, _prepare_output
        )


class SequenceParallel(PairwiseParallel):
    """
    SequenceParallel concatenate colwise and rowwise styles as a fixed
    pair together with sequence parallel like what Megatron-LM Sequence parallel
    (https://arxiv.org/pdf/2205.05198.pdf) is doing.
    We assume both input and output need to be sharded DTensors.

    .. warning::
        SequenceParallel does not support ``nn.MultiheadAttention``,
        ``nn.Transformer`` well at this moment. One workaround is to apply
        ``ColwiseParallel`` and ``RowwiseParallel`` to the components of
        transformer. We recommend to use ``SequenceParallel`` only
        for even-number-layer MLP for now.
    """
    @_deprecate_warnings("Use ColwiseParallel and RowwiseParallel instead.")  # type: ignore[misc]
    def __init__(
        self,
        input_layouts=None,
        output_layouts=None,
        use_local=True,
        _prepare_input=None,
        _prepare_output=None,
    ) -> None:
        super().__init__(  # type: ignore[misc]
            input_layouts,
            output_layouts,
            use_local,
            make_input_reshard_replicate,
            make_output_reshard_tensor,
        )


@_deprecate_warnings("Specify input_layouts instead.")  # type: ignore[arg-type]
@_prepare_input_validate  # type: ignore[arg-type] # pyre-ignore[56]
def make_input_shard_1d(
    input: Union[torch.Tensor, DTensor],
    device_mesh: Optional[DeviceMesh] = None,
    dim: int = 0,
) -> DTensor:
    """
    Shard input tensor on ``dim`` over an 1-D device mesh. This function will be used in ParallelStyle.

    Args:
        input (Union[:class:`torch.Tensor`, :class:`DTensor`]):
            Single tensor will be sharded on dimension ``dim``
            over the 1-D :class:`DeviceMesh`.
        device_mesh (:class:`DeviceMesh`, optional):
            The 1-D device mesh where ``input`` will be sharded.
            If no :class:`DeviceMesh` is passed and ``input`` is a :class:`DTensor`,
            `input.device_mesh` will be used.
            If :class:`DeviceMesh` is not 1-D, an exception will be thrown.
            Default: ``None``
        dim (int, optional): The sharding dimension of ``input`` tensor.
            Default: 0

    Returns:
        A :class:`DTensor` sharded on dimension ``dim`` over ``device_mesh``.
    """
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


@_deprecate_warnings("Specify input_layouts instead.")  # type: ignore[arg-type]
@_prepare_input_validate  # type: ignore[arg-type] # pyre-ignore[56]
def make_input_shard_1d_last_dim(
    input: Union[torch.Tensor, DTensor],
    device_mesh: Optional[DeviceMesh] = None,
) -> DTensor:
    """
    Wrapper func of ``make_input_shard_1d`` with ``dim`` = -1.

    Args:
        input (Union[:class:`torch.Tensor`, :class:`DTensor`]):
            This single tensor will be sharded on the last dimension
            over the 1-D :class:`DeviceMesh`.
        device_mesh (:class:`DeviceMesh`, optional):
            The 1-D device mesh where ``input`` will be sharded.
            If no :class:`DeviceMesh` is passed and ``input`` is a :class:`DTensor`,
            `input.device_mesh` will be used.
            If :class:`DeviceMesh` is not 1-D, an exception will be thrown.
            Default: ``None``

    Returns:
        A :class:`DTensor` sharded on the last dimension over ``device_mesh``.
    """
    return make_input_shard_1d(input, device_mesh, dim=input.dim() - 1)  # type: ignore[call-arg, misc]


@_deprecate_warnings("Specify input_layouts instead.")  # type: ignore[arg-type]
@_prepare_input_validate  # type: ignore[arg-type] # pyre-ignore[56]
def make_input_reshard_replicate(
    input: torch.Tensor,
    device_mesh: DeviceMesh,
) -> DTensor:
    """
    To construct a Sharded DTensor from a tensor on different ranks
    and then convert to a replicate DTensor.

    Args:
        input (:class:`torch.Tensor`):
            The input tensor on each rank which consists of a global DTensor
            sharded on dimension ``0`` over the 1-D :class:`DeviceMesh`
            and then the sharded DTensor is converted to a replicate DTensor.
        device_mesh (:class:`DeviceMesh`, optional):
            The 1-D device mesh where ``input`` will be sharded.
            If :class:`DeviceMesh` is not 1-D, an exception will be thrown.
            Default: ``None``

    Returns:
        A :class:`DTensor` sharded on dimension ``0`` over ``device_mesh``
            and then converted to replicate.
    """
    return make_input_replicate_1d(  # type: ignore[call-arg, misc]
        make_input_shard_1d(input, device_mesh, dim=0), device_mesh  # type: ignore[call-arg, misc]
    )


@_deprecate_warnings("Specify input_layouts instead.")  # type: ignore[arg-type]
@_prepare_input_validate  # type: ignore[arg-type] # pyre-ignore[56]
def make_input_replicate_1d(
    input: Union[torch.Tensor, DTensor],
    device_mesh: Optional[DeviceMesh] = None,
) -> DTensor:
    """
    Replicate input tensor over an 1-D device mesh. This function will be used in ParallelStyle.

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


@_deprecate_warnings("Specify output_layouts instead.")  # type: ignore[arg-type]
@_prepare_output_validate  # type: ignore[arg-type] # pyre-ignore[56]
def make_output_shard_1d(
    output: DTensor, device_mesh: Optional[DeviceMesh] = None, dim: int = 0
) -> DTensor:
    """
    Convert Output DTensor to a sharded DTensor. This will be used in ParallelStyle.

    Args:
        output (:class:`DTensor`):
            Output of module to be converted.
        device_mesh (:class:`DeviceMesh`, optional):
            Object needed to shard the output and it needs to be a 1D ``device_mesh``
            and we will throw exceptions if a non-1D ``device_mesh`` is passed in.
            If no ``device_mesh`` is passed in, we will reuse the one from output.
            Default: ``None``
        dim (int): Sharding dim for output. Default: 0

    Return:
        A :class:`DTensor` object sharded on the given dim.
    """

    return output.redistribute(device_mesh, [Shard(dim)])


@_deprecate_warnings("Specify output_layouts instead.")  # type: ignore[arg-type]
@_prepare_output_validate  # type: ignore[arg-type] # pyre-ignore[56]
def make_output_replicate_1d(
    output: DTensor, device_mesh: Optional[DeviceMesh] = None
) -> DTensor:
    """
    Convert Output DTensor to a replicated DTensor. This will be used in ParallelStyle.

    Args:
        output (:class:`DTensor`):
            Output of module to be converted.
        device_mesh (:class:`DeviceMesh`, optional):
            Object needed to replicate the output and it needs to be a 1D ``device_mesh``
            and we will throw exceptions if a non-1D ``device_mesh`` is passed in.
            If no ``device_mesh`` is passed in, we will reuse the one from output.
            Default: ``None``

    Return:
        A :class:`DTensor` object made replicate.
    """

    return output.redistribute(device_mesh, [Replicate()])


@_deprecate_warnings("Specify output_layouts instead.")  # type: ignore[arg-type]
@_prepare_output_validate  # type: ignore[arg-type] # pyre-ignore[56]
def make_output_tensor(
    output: DTensor, device_mesh: Optional[DeviceMesh] = None
) -> torch.Tensor:
    """
    Convert Output DTensor to a replicated DTensor first and then convert it to Tensor.

    Args:
        output (:class:`DTensor`):
            Output of module to be converted.
        device_mesh (:class:`DeviceMesh`, optional):
            Object which is needed to replicate the output and it needs to be
            a 1D ``device_mesh`` and we will throw exceptions if a non-1D
            ``device_mesh`` is passed in. If no ``device_mesh`` is passed in,
            we will reuse the one from output.
            Default: ``None``

    Return:
        A :class:`torch.Tensor` object converted from output DTensor.
    """

    return make_output_replicate_1d(  # type: ignore[attr-defined, misc]
        output, device_mesh
    ).to_local()  # type: ignore[call-arg]


@_deprecate_warnings("Specify output_layouts instead.")  # type: ignore[arg-type]
@_prepare_output_validate  # type: ignore[arg-type] # pyre-ignore[56]
def make_sharded_output_tensor(
    output: DTensor, _device_mesh: Optional[DeviceMesh] = None
) -> torch.Tensor:
    """
    Convert sharded Output DTensor to torch.Tensor.

    Args:
        output (:class:`DTensor`):
            Output of module to be converted.

    Return:
        A :class:`torch.Tensor` object converted from output DTensor.

    ``_device_mesh`` is not needed and is just kept to match with
        the signature in its callsite in ``distribute_module``.
    """

    return output.to_local()  # type: ignore[call-arg]


@_deprecate_warnings("Specify output_layouts instead.")  # type: ignore[arg-type]
@_prepare_output_validate  # type: ignore[arg-type] # pyre-ignore[56]
def make_output_reshard_tensor(
    output: DTensor,
    device_mesh: Optional[DeviceMesh] = None,
) -> torch.Tensor:
    """
    Convert Output DTensor to a sharded DTensor and return the local tensor.

    Args:
        output (:class:`DTensor`):
            Output of module to be converted.
        device_mesh (:class:`DeviceMesh`, optional):
            Object needed to shard the output and it needs to be a 1D ``device_mesh``
            and we will throw exceptions if a non-1D ``device_mesh`` is passed in.
            If no ``device_mesh`` is passed in, we will reuse the one from output.
            Default: ``None``

    Return:
        A :class:`torch.Tensor` object converted from output DTensor.
    """

    return make_output_shard_1d(output, device_mesh).to_local()  # type: ignore[call-arg, attr-defined, misc]


def _get_prepare_input(
    input_layouts: LayoutsType, output_layouts: LayoutsType
) -> Callable[[Any], Any]:
    """
    Get the prepare input function for this parallel style.
    """

    def _redistribute_per_both_layouts(t, input_layout, output_layout, device_mesh):
        if not isinstance(t, (DTensor, torch.Tensor)):
            if input_layout is not None:
                raise RuntimeError(
                    "Tensor parallel module expects DTensor or tensor"
                    f" when layout specified but received {type(t)}!"
                )
            else:
                return t
        elif isinstance(t, DTensor):
            return t.redistribute(device_mesh, [output_layout])
        else:  # t is torch.Tensor.
            dtensor = DTensor.from_local(
                t, device_mesh, [input_layout], run_check=False
            )
            return dtensor.redistribute(device_mesh, [output_layout])

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
        if not isinstance(inputs, tuple):
            return _redistribute_per_both_layouts(
                inputs, input_layouts, output_layouts, device_mesh
            )

        if not isinstance(input_layouts, tuple):
            input_layouts = [  # type: ignore[assignment]
                input_layouts,
            ]
            output_layouts = [  # type: ignore[assignment]
                output_layouts,
            ]
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
    output_layouts: LayoutsType, use_local: bool
) -> Callable[[Any], Any]:
    """
    Get the prepare input function for this parallel style.
    """

    def _redistribute_per_layout(t, layout, device_mesh, use_local):
        if not isinstance(t, DTensor):
            if layout is not None:
                raise RuntimeError(
                    "Tensor parallel module expects DTensor or tensor"
                    f" when layout specified but received {type(t)}!"
                )
            else:
                return t
        else:  # t is DTensor.
            dtensor = t.redistribute(device_mesh, [layout])
            return dtensor.to_local() if use_local else dtensor

    def make_output_redistribute_1d(
        output_layouts: LayoutsType,
        use_local: bool,
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
        if not isinstance(outputs, tuple):
            return _redistribute_per_layout(
                outputs, output_layouts, device_mesh, use_local
            )

        if not isinstance(output_layouts, tuple):
            output_layouts = [  # type: ignore[assignment]
                output_layouts,
            ]
        results = []
        for output, output_layout in zip(outputs, output_layouts):  # type: ignore[arg-type]
            results.append(
                _redistribute_per_layout(output, output_layout, device_mesh, use_local)
            )
        return tuple(results)

    return functools.partial(make_output_redistribute_1d, output_layouts, use_local)


class RowwiseParallel(ParallelStyle):
    """
    Partitioning the row of a module.
    We assume the input to be a sharded :class:`DTensor` and output to be a :class:`torch.Tensor`.
    """

    def __init__(
        self,
        input_layouts=Shard(-1),
        output_layouts=Replicate(),
        use_local=True,
        _prepare_input=None,
        _prepare_output=None,
    ) -> None:
        super().__init__(
            input_layouts=input_layouts,
            output_layouts=output_layouts,
            use_local=use_local,
            _prepare_input=_prepare_input
            if _prepare_input is not None
            else _get_prepare_input(
                input_layouts,
                [Shard(-1)] * len(input_layouts)  # type: ignore[arg-type]
                if isinstance(input_layouts, tuple)
                else Shard(-1),
            ),
            _prepare_output=_prepare_output
            if _prepare_output is not None
            else _get_prepare_output(output_layouts, use_local),
        )


class ColwiseParallel(ParallelStyle):
    """
    Partitioning the column of a tensor or module.
    We assume the input to be a replicated :class:`DTensor` and output to be a sharded :class:`torch.Tensor`.
    """

    def __init__(
        self,
        input_layouts=Replicate(),
        output_layouts=Shard(-1),
        use_local=True,
        _prepare_input=None,
        _prepare_output=None,
    ) -> None:
        super().__init__(
            input_layouts=input_layouts,
            output_layouts=output_layouts,
            use_local=use_local,
            _prepare_input=_prepare_input
            if _prepare_input is not None
            else _get_prepare_input(
                input_layouts,
                [Replicate()] * len(input_layouts)  # type: ignore[arg-type]
                if isinstance(input_layouts, tuple)
                else Replicate(),
            ),
            _prepare_output=_prepare_output
            if _prepare_output is not None
            else _get_prepare_output(output_layouts, use_local),
        )
