from __future__ import annotations

import dataclasses
from typing import Any, Optional, TYPE_CHECKING, Union

import torch
from torch._inductor import config, ir
from torch._inductor.select_algorithm import (
    AlgorithmSelectorCache,
    AutotuneArgs,
    extern_kernel_choices,
    ExternKernelCaller,
    TritonBenchmarkRequest,
    TritonTemplateCaller,
)
from torch._inductor.virtualized import V


if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch._prims_common import ShapeType, StrideType


@dataclasses.dataclass
class TensorMeta:
    """
    Contains all the metadata needed to construct a random value for
    autotune benchmarking.

    TODO: This could replace TensorMeta in autotune_process.py
    """

    device: torch.device
    dtype: torch.dtype
    size: ShapeType
    stride: StrideType
    offset: int
    view_size: ShapeType
    view_stride: StrideType
    view_offset: int
    allocation_size: ShapeType
    name: str
    value: Optional[torch.Tensor] = None

    @classmethod
    def from_layout(cls, layout: ir.Layout) -> TensorMeta:
        return cls.from_irnode(ir.Buffer(name="fake", layout=layout))

    @classmethod
    def from_irnodes(cls, irnodes: Sequence[ir.IRNode]) -> list[TensorMeta]:
        return [cls.from_irnode(n) for n in irnodes]

    @classmethod
    def from_irnode(cls, node: ir.IRNode) -> TensorMeta:
        view = None
        if isinstance(node, ir.BaseView):
            # Triton templates want the base tensor. But in case the node is a view,
            # we save the metadata for the base tensor and the view so we can recreate
            # the same view for benchmarking extern kernels.
            view = node
            node = node.unwrap_view()

        size, stride, offset = cls._get_metadata(node)
        view_size, view_stride, view_offset = size, stride, offset
        if view is not None:
            view_size, view_stride, view_offset = cls._get_metadata(view)

        # Inplace padding may reinterpret a tensor to a larger tensor if the stride is
        # large enough. The V.graph.get_allocation_size takes this into account.  So we
        # need to call as_strided in the end to 'view' the tensor with the correct
        # sizes/strides.
        allocation_size = V.graph.sizevars.size_hints(
            V.graph.get_allocation_size(node),  # type: ignore[arg-type]
            fallback=config.unbacked_symint_fallback,
        )

        dtype = node.get_dtype()
        assert dtype is not None
        device = node.get_device()
        assert device is not None
        name = node.get_name()
        assert name is not None

        return cls(
            device,
            dtype,
            size,
            stride,
            offset,
            view_size,
            view_stride,
            view_offset,
            allocation_size,
            name,
        )

    @classmethod
    def _get_metadata(cls, node: ir.IRNode) -> tuple[ShapeType, StrideType, int]:
        size = V.graph.sizevars.size_hints(
            node.get_size(),
            fallback=config.unbacked_symint_fallback,
        )
        stride = V.graph.sizevars.size_hints(
            node.get_stride(),
            fallback=config.unbacked_symint_fallback,
        )
        offset = V.graph.sizevars.size_hint(
            node.get_layout().offset,
            fallback=config.unbacked_symint_fallback,
        )
        return size, stride, offset

    def to_tensor(self, for_extern: bool = False) -> torch.Tensor:
        """
        Create a random tensor from the metadata. 'for_extern' specifies whether
        the view should be set for benchmarking extern kernels, in which case
        we create the original view.
        """
        if self.value is None:
            self.value = AlgorithmSelectorCache.generate_example_value(
                self.size,
                self.stride,
                self.device,
                self.dtype,
                self.offset,
                self.allocation_size,
            )
        if for_extern:
            return torch.as_strided(
                self.value, self.view_size, self.view_stride, self.view_offset
            )
        else:
            return self.value


class ExternBenchmarkRequest:
    """
    Serialized object for benchmarking an extern kernel remotely. Contains the
    name of the kernel and kwargs.
    """

    def __init__(self, name: str, kwargs: Any):
        self.name = name
        self.kwargs = kwargs

    def to_caller(self) -> ExternKernelCaller:
        """
        Create an ExternKernelCaller that can be benchmarked.
        """
        choice = extern_kernel_choices[self.name]
        return choice.bind([], None, (), **self.kwargs)


RemoteBenchmarkChoice = Union[ExternBenchmarkRequest, TritonBenchmarkRequest]


@dataclasses.dataclass
class RemoteBenchmarkRequest:
    """
    Serialized object containing all the information to reconstruct a list of
    choices and their inputs remotely and benchmark them.
    """

    choices: list[RemoteBenchmarkChoice]
    input_meta: list[TensorMeta]
    output_meta: TensorMeta

    def benchmark(self) -> list[float]:
        """
        Materialize random inputs from the metadata and benchmark each choice.
        """
        # De-duplicate Triton kernel args
        unique = {m.name: m for m in self.input_meta}
        inputs = [m.to_tensor(for_extern=False) for m in unique.values()]

        # Extern args are not de-duplicated (and have the original view)
        inputs_extern = [
            unique[m.name].to_tensor(for_extern=True) for m in self.input_meta
        ]

        args = AutotuneArgs.from_choice_args(
            inputs,
            inputs_extern,
            self.output_meta.to_tensor(for_extern=False),
            self.output_meta.to_tensor(for_extern=True),
        )

        # Translate the extern kernel names into ExternKernelCaller objects expected by
        # benchmark_choices(). The TritonTemplateCaller choices can be passed directly.
        choices = [
            c.to_caller() if isinstance(c, ExternBenchmarkRequest) else c
            for c in self.choices
        ]
        return AlgorithmSelectorCache.benchmark_choices(choices, args)  # type: ignore[arg-type]

    @classmethod
    def from_choices(
        cls,
        choices: Sequence[ir.ChoiceCaller],
        input_nodes: Sequence[ir.IRNode],
        layout: ir.Layout,
    ) -> RemoteBenchmarkRequest:
        """
        Create a request from a list of choices and inputs.
        """
        remote_choices: list[RemoteBenchmarkChoice] = []
        for choice in choices:
            if isinstance(choice, ExternKernelCaller):
                req = ExternBenchmarkRequest(choice.name, choice.kwargs)
                remote_choices.append(req)
            else:
                assert isinstance(choice, TritonTemplateCaller)
                remote_choices.append(choice.bmreq)

        return cls(
            remote_choices,
            TensorMeta.from_irnodes(input_nodes),
            TensorMeta.from_layout(layout),
        )
