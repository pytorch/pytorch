# Copyright (c) Meta Platforms, Inc. and affiliates
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple
from functools import partial

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

    It only defines the ``apply`` method for ``parallelize_module`` to use, this allows maximum
    flexibility for different kind of style implementations.
    """

    @abstractmethod
    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        ...


class ColwiseParallel(ParallelStyle):
    """
    Partition a compatible nn.Module in a column-wise fashion. Currently supports nn.Linear and nn.Embedding.
    Users can compose it together with RowwiseParallel to achieve the sharding of more complicated modules.
    (i.e. MLP, Attention)

    Keyword Args:
        input_layouts (Placement, optional):
            The DTensor layout of input tensor for the nn.Module, this is used to annotate the input tensor to
            become a DTensor. If not specified, we assume the input tensor to be replicated.
        output_layouts (Placement, optional):
            The DTensor layout of the output for the nn.Module, this is used to ensure the output of the nn.Module
            with the user desired layout. If not specified, the output tensor is sharded on the last dimension.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module output, default: True.
    Returns:
        A :class:`ParallelStyle` object that represents Colwise sharding of the nn.Module.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> ...
        >>> m = Model(...)  # m is a nn.Module that contains a "w1" nn.Linear submodule
        >>> tp_mesh = init_device_mesh("cuda", (8,))
        >>>
        >>> # By default, the input of the "w1" Linear will be converted to Replicated DTensor
        >>> # and the output of "w1" will return :class:`torch.Tensor` that shards on the last dim.
        >>>
        >>> sharded_mod = parallelize_module(m, tp_mesh, {"w1": ColwiseParallel()})
        >>> ...

    .. note:: By default ``ColwiseParallel`` output is sharded on the last dimension if the ``output_layouts`` not
        specified, if there're operators that require specific tensor shape (i.e. before the paired ``RowwiseParallel``),
        keep in mind that if the output is sharded the operator might need to be adjusted to the sharded size.
    """

    def __init__(
        self,
        *,
        input_layouts: Optional[Placement] = None,
        output_layouts: Optional[Placement] = None,
        use_local_output: bool = True
    ):
        super().__init__()
        self.input_layouts = (input_layouts or Replicate(), )
        self.output_layouts = (output_layouts or Shard(-1), )
        # colwise linear runtime sharding (desired sharding):
        # 1. requires replicate input
        # 2. shard output on last dim
        self.desired_input_layouts = (Replicate(), )
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(input_layouts, desired_input_layouts, mod, inputs, device_mesh):
        # TODO: figure out dynamo support for instance method and switch this to instance method

        # annotate module input placements/sharding with input_layouts
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(input_tensor, device_mesh, input_layouts, run_check=False)

        # transform the input layouts to the desired layouts of ColwiseParallel
        if input_layouts != desired_input_layouts:
            input_tensor = input_tensor.redistribute(placements=desired_input_layouts)
        return input_tensor

    def _partition_linear_fn(self, name, module, device_mesh):
        # colwise shard weight/bias to Shard(0), weight be Shard(0)
        # means Colwise as Linear is input * weight^T + bias, where
        # weight would become Shard(1)
        for name, param in module.named_parameters():
            dist_param = nn.Parameter(
                distribute_tensor(param, device_mesh, [Shard(0)])
            )
            module.register_parameter(name, dist_param)

    def _partition_embedding_fn(self, name, module, device_mesh):
        # colwise shard embedding.weight is straight forward as Shard(1)
        for name, param in module.named_parameters():
            dist_param = nn.Parameter(
                distribute_tensor(param, device_mesh, [Shard(1)])
            )
            module.register_parameter(name, dist_param)

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        # outputs is a shard on last dimension DTensor, i.e. Shard(-1)
        outputs = outputs.redistribute(placements=output_layouts)
        # back to local tensor
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        if isinstance(module, nn.Linear):
            partition_fn = self._partition_linear_fn
        elif isinstance(module, nn.Embedding):
            partition_fn = self._partition_embedding_fn
        else:
            raise NotImplementedError("ColwiseParallel currently only support nn.Linear and nn.Embedding!")

        return distribute_module(
            module,
            device_mesh,
            partition_fn,
            partial(self._prepare_input_fn, self.input_layouts, self.desired_input_layouts),
            partial(self._prepare_output_fn, self.output_layouts, self.use_local_output),
        )


class RowwiseParallel(ParallelStyle):
    """
    Partition a compatible nn.Module in a row-wise fashion. Currently supports nn.Linear and nn.Embedding.
    Users can compose it with ColwiseParallel to achieve the sharding of more complicated modules.
    (i.e. MLP, Attention)

    Keyword Args:
        input_layouts (Placement, optional):
            The DTensor layout of input tensor for the nn.Module, this is used to annotate the input tensor to
            become a DTensor. If not specified, we assume the input tensor to be sharded on the last dimension.
        output_layouts (Placement, optional):
            The DTensor layout of the output for the nn.Module, this is used to ensure the output of the nn.Module
            with the user desired layout. If not specified, the output tensor is replicated.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module output, default: True.
    Returns:
        A :class:`ParallelStyle` object that represents Rowwise sharding of the nn.Module.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, RowwiseParallel
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> ...
        >>> m = Model(...)  # m is a nn.Module that contains a "w2" nn.Linear submodule
        >>> tp_mesh = init_device_mesh("cuda", (8,))
        >>>
        >>> # By default, the input of the "w2" Linear will be converted to DTensor that shards on the last dim
        >>> # and the output of "w2" will return a replicated :class:`torch.Tensor`.
        >>>
        >>> sharded_mod = parallelize_module(m, tp_mesh, {"w2": RowwiseParallel()}),
        >>> ...
    """

    def __init__(
        self,
        *,
        input_layouts: Optional[Placement] = None,
        output_layouts: Optional[Placement] = None,
        use_local_output: bool = True
    ):
        super().__init__()
        self.input_layouts = (input_layouts or Shard(-1), )
        self.output_layouts = (output_layouts or Replicate(), )
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(input_layouts, desired_input_layouts, mod, inputs, device_mesh):
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(input_tensor, device_mesh, input_layouts, run_check=False)

        if input_layouts != desired_input_layouts:
            input_tensor = input_tensor.redistribute(placements=desired_input_layouts)
        return input_tensor

    def _partition_linear_fn(self, name, module, device_mesh):
        # Rowwise shard weight to Shard(1), bias to Replicate(), weight be Shard(1)
        # means Rowwise as nn.Linear is input * weight^T + bias, where
        # weight would become Shard(0)
        module.register_parameter("weight", nn.Parameter(
            distribute_tensor(module.weight, device_mesh, [Shard(1)])
        ))
        if module.bias is not None:
            module.register_parameter("bias", nn.Parameter(
                distribute_tensor(module.bias, device_mesh, [Replicate()])
            ))

    def _partition_embedding_fn(self, name, module, device_mesh):
        # rowwise shard embedding.weight is Shard(0)
        for name, param in module.named_parameters():
            dist_param = nn.Parameter(
                distribute_tensor(param, device_mesh, [Shard(0)])
            )
            module.register_parameter(name, dist_param)

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        # Rowwise sharding produces partial output, depending on output layouts:
        # 1. to replicate -> allreduce
        # 2. to shard -> reduce_scatter
        outputs = outputs.redistribute(placements=output_layouts)
        # back to local tensor if use_local_output is True
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        if isinstance(module, nn.Linear):
            partition_fn = self._partition_linear_fn
            # rowwise linear runtime sharding requires input tensor shard on last dim
            self.desired_input_layouts: Tuple[Placement, ...] = (Shard(-1), )
        elif isinstance(module, nn.Embedding):
            partition_fn = self._partition_embedding_fn
            # rowwise embedding runtime sharding requires input tensor replicated
            self.desired_input_layouts = (Replicate(), )
        else:
            raise NotImplementedError("RowwiseParallel currently only support nn.Linear and nn.Embedding!")

        return distribute_module(
            module,
            device_mesh,
            partition_fn,
            partial(self._prepare_input_fn, self.input_layouts, self.desired_input_layouts),
            partial(self._prepare_output_fn, self.output_layouts, self.use_local_output),
        )


class PrepareModuleInput(ParallelStyle):
    """
    Configure the nn.Module's inputs to convert the input tensors of the nn.Module to DTensors at runtime according to
    ``input_layouts``, and perform layout redistribution according to the ``desired_input_layouts``.

    Keyword Args:
        input_layouts (Union[Placement, Tuple[Placement]]):
            The DTensor layouts of input tensors for the nn.Module, this is used to convert the input tensors to
            DTensors. If some inputs are not torch.Tensor or no need to convert to DTensors, ``None`` need to be specified
            as a placeholder.
        desired_input_layouts (Union[Placement, Tuple[Placement]]):
            The desired DTensor layout of input tensors for the nn.Module, this is used to ensure the inputs of the nn.Module
            have the desired DTensor layouts. This argument needs to have the same length with ``input_layouts``.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module inputs, default: False.
    Returns:
        A :class:`ParallelStyle` object that prepares the sharding layouts of the nn.Module's inputs.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, PrepareModuleInput
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> ...
        >>> block = TransformerBlock(...)  # block is a nn.Module that contains an "attn" Attention submodule
        >>> tp_mesh = init_device_mesh("cuda", (8,))
        >>>
        >>> # According to the style specified below, the first input of attn will be annotated to Sharded DTensor
        >>> # and then redistributed to Replicated DTensor.
        >>> parallelize_module(
        >>>     block, # this can be a submodule or module
        >>>     tp_mesh,
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
        if len(inputs) != len(self.input_layouts):
            raise ValueError("module inputs and input_layouts should have same length!")

        for inp, input_layout, desired_layout in zip(inputs, self.input_layouts, self.desired_input_layouts):
            if input_layout is not None:
                if isinstance(inp, DTensor):
                    # TODO: re-enable the check once we fix the compile path
                    # assert inp.placements[0] == input_layout
                    dt_inp = inp
                else:
                    dt_inp = DTensor.from_local(inp, device_mesh, (input_layout,), run_check=False)
                if input_layout != desired_layout:
                    dt_inp = dt_inp.redistribute(placements=(desired_layout,))
                prepared_inputs.append(dt_inp.to_local() if self.use_local_output else dt_inp)
            else:
                prepared_inputs.append(inp)
        return tuple(prepared_inputs)

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        module.register_forward_pre_hook(lambda _, inputs: self._prepare_input_fn(inputs, device_mesh))  # type: ignore[misc, call-arg]
        return module


class PrepareModuleOutput(ParallelStyle):
    """
    Configure the nn.Module's outputs to convert the output tensors of the nn.Module to DTensors at runtime according to
    ``output_layouts``, and perform layout redistribution according to the ``desired_output_layouts``.

    Keyword Args:
        output_layouts (Union[Placement, Tuple[Placement]]):
            The DTensor layouts of output tensors for the nn.Module, this is used to convert the output tensors to
            DTensors if they are :class:`torch.Tensor`. If some outputs are not torch.Tensor or no need to convert to DTensors,
            ``None`` need to be specified as a placeholder.
        desired_output_layouts (Union[Placement, Tuple[Placement]]):
            The desired DTensor layouts of output tensors for the nn.Module, this is used to ensure the outputs of the nn.Module
            have the desired DTensor layouts.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module outputs, default: False.
    Returns:
        A ParallelStyle object that prepares the sharding layouts of the nn.Module's outputs.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, PrepareModuleOutput
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> ...
        >>> block = TransformerBlock(...)  # block is a nn.Module that contains an "attn" Attention submodule
        >>> tp_mesh = init_device_mesh("cuda", (8,))
        >>>
        >>> # According to the style specified below, the output of the TransformerBlock will be converted to Replicated DTensor
        >>> # and then redistributed to Sharded DTensor.
        >>> parallelize_module(
        >>>     block, # this can be a submodule or module
        >>>     tp_mesh,
        >>>     parallelize_plan = PrepareModuleOutput(
        >>>         output_layouts=Replicate(),
        >>>         desired_output_layouts=Shard(0)
        >>>     )
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
        assert len(self.output_layouts) == len(self.desired_output_layouts), \
            "output_layouts and desired_output_layouts should have same length!"

    def _prepare_out_fn(self, outputs, device_mesh):
        prepared_outputs = []
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        if len(outputs) != len(self.output_layouts):
            raise ValueError("module outputs and output_layouts should have same length!")
        for out, out_layout, desired_out_layout in zip(outputs, self.output_layouts, self.desired_output_layouts):
            if out_layout is not None:
                if isinstance(out, DTensor):
                    # TODO: re-enable the check once we fix the compile path
                    # assert out.placements[0] == out_layout
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

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        module.register_forward_hook(lambda _, inputs, outputs: self._prepare_out_fn(outputs, device_mesh))  # type: ignore[misc, call-arg]
        return module
