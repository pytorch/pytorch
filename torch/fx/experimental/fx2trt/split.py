import dataclasses as dc
import logging
import re
import typing as t

import torch.fx as fx
from .tools.trt_splitter import (
    create_trt_operator_support,
    TRTSplitter,
    TRTSplitterSetting,
)
from torch.fx.passes.operator_support import OperatorSupportBase


logger = logging.getLogger(__name__)

Input = t.Sequence[t.Any]


class SplitFunc:
    """Signature for fx2trt split functions"""

    def __call__(
        self,
        module: fx.GraphModule,
        input: Input,
    ) -> t.Tuple[fx.GraphModule, t.Sequence["SplitInfo"]]:
        """Splits a module into lowerable and non-lowerable subnets.

        This function splits a module into lowerable subnets that can be run
        via TensorRT, and non-lowerable subnets that will be run via CUDA.

        Args:
            module: the module to be split
            input: the sample input to the module

        Returns:
            a list of `SplitInfo` describing the result of the split
        """
        raise NotImplementedError()


@dc.dataclass(frozen=True)
class SplitInfo:
    """Describes the result of a `SplitFunc`

    Attributes:
        module: the module representing the split subnet
        input: the sample input to the subnet
        name: name of the subnet as it is contained by its parent module
        device: "acc" or "cpu" representing lowerable vs non-lowerable
        order: the order of the split
    """

    module: fx.GraphModule
    input: Input
    name: str
    device: str
    order: int


@dc.dataclass(frozen=True)
class Splitter(SplitFunc):
    """A composable fx2trtr splitter.

    See `SplitFunc`.

    Attributes:
        min_acc_module_size: minimum split module size
    """

    _INPUT_ATTR: t.ClassVar[str] = "_split_graph_recorded_input"
    min_acc_module_size: int
    operator_supported: OperatorSupportBase = dc.field(
        default_factory=lambda: create_trt_operator_support()
    )

    @classmethod
    def create(
        cls,
        use_implicit_batch_dim: bool,
        min_acc_module_size: int = 20,
    ):
        return Splitter(
            min_acc_module_size=min_acc_module_size,
            operator_supported=create_trt_operator_support(use_implicit_batch_dim),
        )

    def __call__(self, module, input) -> t.Tuple[fx.GraphModule, t.Sequence[SplitInfo]]:
        trt_split_result = self._trt_split(module, input)

        logger.debug(
            f"""TRT split result graph >>> {
                trt_split_result.graph
            }"""
        )

        Splitter._propagate_split_inputs(
            trt_split_result, input, dict(trt_split_result.named_children()).keys()
        )

        return (
            trt_split_result,
            [
                Splitter._create_split_info(name, subgraph, parent=trt_split_result)
                for name, subgraph in trt_split_result.named_children()
            ],
        )

    @classmethod
    def _propagate_split_inputs(
        cls,
        graph: fx.GraphModule,
        input: Input,
        target_modules: t.Collection[str],
    ) -> None:
        """
        Input propagation on subnets

        TODO: refactor so we don't set inputs onto the subgraphs
        """
        handles = []

        def pre_forward(mod, input):
            setattr(mod, cls._INPUT_ATTR, input)

        def _install_hook(g):
            nonlocal handles
            if not g:
                return
            for _n, _g in g.named_children():
                if _n in target_modules:
                    handles.append(_g.register_forward_pre_hook(pre_forward))
                    _install_hook(_g)

        try:
            _install_hook(graph)
            graph(*input)
        finally:
            for h in handles:
                h.remove()

    @classmethod
    def _create_split_info(cls, name, graph, parent) -> SplitInfo:
        device, order = cls._parse_splitter_subgraph_name(name)
        input = getattr(graph, cls._INPUT_ATTR)
        delattr(graph, cls._INPUT_ATTR)
        return SplitInfo(
            module=graph,
            input=input,
            name=name,
            device=device,
            order=order,
        )

    @classmethod
    def _parse_splitter_subgraph_name(cls, name: str) -> t.Tuple[str, int]:
        match = re.match("_run_on_([a-z]+)_([0-9]+)", name)
        assert match, f"{name} doesn't comform with splitter subgraph naming convention"
        return (match[1], int(match[2]))

    def _trt_split(self, graph: fx.GraphModule, input: Input) -> fx.GraphModule:
        splitter_settings = TRTSplitterSetting()
        splitter_settings.min_acc_module_size = self.min_acc_module_size

        splitter = TRTSplitter(
            graph,
            input,  # type: ignore[arg-type]
            self.operator_supported,
            settings=splitter_settings,
        )
        logger.info(
            f"""{splitter.node_support_preview.__name__}: {
            splitter.node_support_preview()
            }"""
        )
        return splitter()
