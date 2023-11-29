# Copyright (c) Meta Platforms, Inc. and affiliates
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple

import torch.nn as nn
from torch.distributed._tensor import DeviceMesh, DTensor, Placement, Replicate, Shard, distribute_tensor, distribute_module


__all__ = [
    "ParallelStyle",
    "RowwiseParallel",
    "ColwiseParallel",
    "PrepareModuleInput",
    "PrepareModuleOutput",
]


class ParallelStyle(ABC):
    """
    The parallel style contract defines how the module or submodule should be parallelized.

    It only defines the `apply` method for `parallelize_module` to use, this allows maximum
    flexibility for different kind of style implementations.
    """

    @abstractmethod
    def apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        ...


class ColwiseParallel(ParallelStyle):
    """
    Partition the column of a nn.Module and allow the parallelized module to run colwise sharded computation.
    Users can compose it with RowwiseParallel to achieve the sharding of more complicated modules.

    Keyword Args:
        input_layouts (Placement, optional):
            The DTensor layout of input tensor for the nn.Module, this is used to annotate the input tensor to
            become a DTensor. If not specified, we assume the input tensor to be replicated.
        output_layouts (Placement, optional):
            The DTensor layout of the output for the nn.Module, this is used to ensure the output of the nn.Module
            with the desired layout. If not specified, we assume the output tensor to be sharded on the last dimension.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of `DTensor` for the module output, default: True.
    Returns:
        A ParallelStyle object that represents Colwise sharding of the nn.Module.

    .. warning::
        ColwiseParallel now only support ``nn.Linear`` and ``nn.Embedding``.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel
        >>> ...
        >>> # By default, the input of the "w1" Linear will be annotated to Replicated DTensor
        >>> # and the output of "w1" will return :class:`torch.Tensor` that shards on the last dim.
        >>>>
        >>> parallelize_module(
        >>>     module=block, # this can be a submodule or module
        >>>     ...,
        >>>     parallelize_plan={"w1": ColwiseParallel()},
        >>> )
        >>> ...
    """

    def __init__(
        self,
        *,
        input_layouts: Optional[Placement] = None,
        output_layouts: Optional[Placement] = None,
        use_local_output: bool = True
    ) -> None:
        super().__init__()
        self.input_layouts = (input_layouts or Replicate(), )
        self.output_layouts = (output_layouts or Shard(-1), )
        # colwise linear runtime sharding (desired sharding):
        # 1. requires replicate input
        # 2. shard output on last dim
        self.desired_input_layouts = (Replicate(), )
        self.desired_output_layouts = (Shard(-1), )
        self.use_local_output = use_local_output

    def _prepare_input_fn(self, inputs, device_mesh):
        # annotate module input placements/sharding with input_layouts
        dtensor = DTensor.from_local(inputs[0], device_mesh, self.input_layouts)
        # transform the input layouts to the desired layouts of ColwiseParallel
        desired_sharding = (Replicate(), )
        if self.input_layouts != desired_sharding:
            dtensor = dtensor.redistribute(placements=self.desired_input_layouts)
        return dtensor

    def _partition_fn(self, name, module, device_mesh):
        if isinstance(module, nn.Linear):
            # colwise shard weight/bias to Shard(0), weight be Shard(0)
            # means Colwise as Linear is input * weight^T + bias, where
            # weight would become Shard(1)
            for name, param in module.named_parameters():
                dist_param = nn.Parameter(
                    distribute_tensor(param, device_mesh, [Shard(0)])
                )
                module.register_parameter(name, dist_param)
        elif isinstance(module, nn.Embedding):
            # colwise shard embedding.weight is straight forward as Shard(1)
            for name, param in module.named_parameters():
                dist_param = nn.Parameter(
                    distribute_tensor(param, device_mesh, [Shard(1)])
                )
                module.register_parameter(name, dist_param)
        else:
            raise NotImplementedError(
                "ColwiseParallel only supports nn.Linear"
                f"and nn.Embedding for now, but found {type(module)}!"
            )

    def _prepare_output_fn(self, outputs, device_mesh):
        # outputs is a shard on last dimension DTensor, i.e. Shard(-1)
        if outputs.placements != self.desired_output_layouts:
            outputs = outputs.redistribute(placements=self.desired_output_layouts)
        # back to local tensor
        return outputs.to_local() if self.use_local_output else outputs

    def apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._partition_fn,
            self._prepare_input_fn,
            self._prepare_output_fn
        )


class RowwiseParallel(ParallelStyle):
    """
    Partition the row of a nn.Module and allow the parallelized module to run rowwise sharded computation.
    Users can compose it with ColwiseParallel to achieve the sharding of more complicated modules.

    Keyword Args:
        input_layouts (Placement, optional):
            The DTensor layout of input tensor for the nn.Module, this is used to annotate the input tensor to
            become a DTensor. If not specified, we assume the input tensor to be sharded on the last dimension.
        output_layouts (Placement, optional):
            The DTensor layout of the output for the nn.Module, this is used to ensure the output of the nn.Module
            with the desired layout. If not specified, we assume the output tensor to be replicated.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of `DTensor` for the module output, default: True.
    Returns:
        A ParallelStyle object that represents Rowwise sharding of the nn.Module.

    .. warning::
        RowwiseParallel now only support ``nn.Linear``.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, RowwiseParallel
        >>> ...
        >>> # By default, the input of the "w2" Linear will be annotated to DTensor that shards on the last dim
        >>> # and the output of "w2" will return a replicated :class:`torch.Tensor`.
        >>>
        >>> parallelize_module(
        >>>     module=block, # this can be a submodule or module
        >>>     ...,
        >>>     parallelize_plan={"w2": RownwiseParallel()},
        >>> )
        >>> ...
    """

    def __init__(
        self,
        *,
        input_layouts: Optional[Placement] = None,
        output_layouts: Optional[Placement] = None,
        use_local_output: bool = True
    ) -> None:
        super().__init__()
        self.input_layouts = (input_layouts or Shard(-1), )
        self.output_layouts = (output_layouts or Replicate(), )
        # rowwise linear runtime sharding:
        # 1. shard input on last dim
        # 2. partial output, to replicate -> allreduce
        self.desired_input_layouts = (Shard(-1), )
        self.desired_output_layouts = (Replicate(), )
        self.use_local_output = use_local_output

    def _prepare_input_fn(self, inputs, device_mesh):
        dtensor = DTensor.from_local(inputs[0], device_mesh, self.input_layouts)
        if self.input_layouts != self.desired_input_layouts:
            dtensor = dtensor.redistribute(placements=self.desired_input_layouts)
        return dtensor

    def _partition_fn(self, name, module, device_mesh):
        assert isinstance(module, nn.Linear), "Only support nn.Linear"
        # Rowwise shard weight to Shard(1), bias to Replicate(), weight be Shard(1)
        # means Rowwise as Linear is input * weight^T + bias, where
        # weight would become Shard(0)
        module.register_parameter("weight", nn.Parameter(
            distribute_tensor(module.weight, device_mesh, [Shard(1)])
        ))
        if module.bias is not None:
            module.register_parameter("bias", nn.Parameter(
                distribute_tensor(module.bias, device_mesh, [Replicate()])
            ))

    def _prepare_output_fn(self, outputs, device_mesh):
        if outputs.placements != self.desired_output_layouts:
            outputs = outputs.redistribute(placements=self.desired_output_layouts)
        # back to local tensor if use_local_output is True
        return outputs.to_local() if self.use_local_output else outputs

    def apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._partition_fn,
            self._prepare_input_fn,
            self._prepare_output_fn
        )


class PrepareModuleInput(ParallelStyle):
    """
    Annotate the input tensors of nn.Module to DTensors according to input_layouts, and perform layout redistribution
    according to the desired_input_layouts.

    Keyword Args:
        input_layouts (Union[Placement, Tuple[Placement]]):
            The DTensor layouts of input tensors for the nn.Module, this is used to annotate the input tensors to
            become a DTensor. if some inputs are not tensor, `None` need to be specified as a placeholder.
        desired_input_layouts (Union[Placement, Tuple[Placement]]):
            The desired DTensor layout of input tensors for the nn.Module, this is used to ensure the inputs of the nn.Module
            have the desired DTensor layouts. This argument needs to have the same length with `input_layouts`.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of `DTensor` for the module inputs, default: False.
    Returns:
        A ParallelStyle object that prepares the sharding layouts of the nn.Module's inputs.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, PrepareModuleInput
        >>> ...
        >>> # According to the style specified below, the first input of attn will be annotated to Sharded DTensor
        >>> # and then redistributed to Replicated DTensor.
        >>> parallelize_module(
        >>>     module=block, # this can be a submodule or module
        >>>     ...,
        >>>     parallelize_plan={
        >>>         "attn": PrepareModuleInput(
        >>>             input_layouts=(Shard(0), None, None, ...),
        >>>             desired_input_layouts=(Replicate(), None, None, ...)
        >>>         ),
        >>>     }
        >>> )
    """

    def __init__(
        self,
        *,
        input_layouts: Union[Placement, Tuple[Placement]],
        desired_input_layouts: Union[Placement, Tuple[Placement]],
        use_local_output: bool = False
    ):
        self.input_layouts = (input_layouts,) if isinstance(input_layouts, Placement) else input_layouts
        self.desired_input_layouts = \
            (desired_input_layouts,) if isinstance(desired_input_layouts, Placement) else desired_input_layouts
        self.use_local_output = use_local_output
        assert len(self.input_layouts) == len(self.desired_input_layouts), \
            "input_layouts and desired_input_layouts should have same length!"

    def _prepare_input_fn(self, inputs, device_mesh):
        prepared_inputs = []
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        for inp, input_layout, desired_layout in zip(inputs, self.input_layouts, self.desired_input_layouts):
            if input_layout is not None:
                dt_inp = DTensor.from_local(inp, device_mesh, (input_layout,), run_check=False)
                if input_layout != desired_layout:
                    dt_inp = dt_inp.redistribute(placements=(desired_layout,))
                prepared_inputs.append(dt_inp.to_local() if self.use_local_output else dt_inp)
            else:
                prepared_inputs.append(inp)
        return tuple(prepared_inputs)

    def apply(self, module, device_mesh):
        module.register_forward_pre_hook(lambda _, inputs: self._prepare_input_fn(inputs, device_mesh))  # type: ignore[misc, call-arg]
        return module


class PrepareModuleOutput(ParallelStyle):
    """
    Annotate the output tensors of nn.Module with DTensors according to output_layouts, and perform layout redistribution
    according to the desired_output_layouts.

    Keyword Args:
        output_layouts (Union[Placement, Tuple[Placement]]):
            The DTensor layouts of output tensors for the nn.Module, this is used to annotate the output tensors to
            a DTensor if they are `torch.Tensor`. if some outputs are not tensor, `None` need to be specified as a placeholder.
        desired_output_layouts (Union[Placement, Tuple[Placement]]):
            The desired DTensor layouts of output tensors for the nn.Module, this is used to ensure the outputs of the nn.Module
            have the desired DTensor layouts.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of `DTensor` for the module outputs, default: False.
    Returns:
        A ParallelStyle object that prepares the sharding layouts of the nn.Module's outputs.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, PrepareModuleOutput
        >>> ...
        >>> # According to the style specified below, the first input of attn will be annotated to Sharded DTensor
        >>> # and then redistributed to Replicated DTensor.
        >>> parallelize_module(
        >>>     module=block, # this can be a submodule or module
        >>>     ...,
        >>>     parallelize_plan={
        >>>         "submodule": PrepareModuleOutput(
        >>>             output_layouts=Replicate(),
        >>>             desired_output_layouts=Shard(0)
        >>>         ),
        >>>     }
        >>> )
    """
    def __init__(
        self,
        *,
        output_layouts: Union[Placement, Tuple[Placement]],
        desired_output_layouts: Union[Placement, Tuple[Placement]],
        use_local_output: bool = True
    ):
        self.output_layouts = (output_layouts,) if isinstance(output_layouts, Placement) else output_layouts
        self.desired_output_layouts = \
            (desired_output_layouts,) if isinstance(desired_output_layouts, Placement) else desired_output_layouts
        self.use_local_output = use_local_output

    def _prepare_out_fn(self, outputs, device_mesh):
        prepared_outputs = []
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        for out, out_layout, desired_out_layout in zip(outputs, self.output_layouts, self.desired_output_layouts):
            if out_layout is not None:
                if isinstance(out, DTensor):
                    assert out.placements[0] == out_layout
                    dt_out = out
                else:
                    dt_out = DTensor.from_local(out, device_mesh, (out_layout,), run_check=False)

                if out_layout != desired_out_layout:
                    dt_out = dt_out.redistribute(placements=(desired_out_layout,))
                prepared_outputs.append(dt_out.to_local() if self.use_local_output else dt_out)
            else:
                prepared_outputs.append(out)
        if len(prepared_outputs) == 1:
            return prepared_outputs[0]
        else:
            return tuple(prepared_outputs)

    def apply(self, module, device_mesh):
        module.register_forward_hook(lambda _, inputs, outputs: self._prepare_out_fn(outputs, device_mesh))  # type: ignore[misc, call-arg]
        return module
