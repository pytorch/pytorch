import dataclasses as dc
import logging
import typing as t
from typing import List

import torch
import torch.fx as fx

import torch.fx.experimental.fx_acc.acc_tracer as acc_tracer
import torch.nn as nn
from torch.fx.experimental.fx2trt.tools.timing_cache_utils import (
    TimingCacheManager,
)
from torch.fx.experimental.fx2trt.split import (
    Splitter,
    SplitFunc,
)
from torch.fx.experimental.const_fold import split_const_subgraphs
from torch.fx.experimental.fx2trt import (
    TRTInterpreter,
    InputTensorSpec,
    TRTModule,
)
from torch.fx.experimental.fx2trt.passes.fuse_pass import (  # @manual=//caffe2:torch_fx2trt
    fuse_permute_linear,
    fuse_permute_matmul,
    fuse_unsqueeze_cat_sum,
)

# @manual=//caffe2:torch_fx2trt
from torch.fx.experimental.fx2trt.passes.remove_duplicate_output_args import (
    RemoveDuplicateOutputArgsFunc,
    remove_duplicate_output_args,
)
from dataclasses import field


logger = logging.getLogger(__name__)

Input = t.Sequence[t.Any]
TModule = t.TypeVar("TModule", bound=nn.Module)


@dc.dataclass
class LowerSetting:
    """
    Basic configuration for lowering stack.

    Args:
    max_batch_size: The maximum batch size which can be used at execution time,
    and also the batch size for which the ICudaEngine will be optimized.

    input_specs: Specs for inputs to engine, can either be a single size or a
    range defined by Min, Optimal, Max sizes.

    explicit_batch_dimension: Use explicit batch dimension during lowering.

    fp16_mode: Enable FP16 dtype during lowering.

    max_workspace_size: The maximum workspace size. The maximum GPU temporary
    memory which the TensorRT engine can use at execution time.

    strict_type_constraints: Require TensorRT engine to strictly follow data type
    setting at execution time.

    enable_fuse: Enable pass fuse duirng lowering, i.e. fuse multiple operations
    as (a->b->c->d)=>(e). Current available fuse source patterns are:
    sparse->matmul->add
    permute->linear
    permute->matmul
    unsqueeze->cat->sum

    enable_fuse_for_sparsity: Enable pass fuse for sparsity.

    verbose_log: Enable TensorRT engine verbose log mode.

    algo_selector: Enable TensorRT algorithm selector at execution time.

    timing_cache_prefix: TensorRT timing cache file path. TensorRT engine will use timing
    cache file at execution time if valid timing cache file is provided.

    save_timing_cache: Save updated timing cache data into timing cache file if the timing
    cache file is provided.

    """
    max_batch_size: int = 2048
    input_specs: List[InputTensorSpec] = field(default_factory=list)
    explicit_batch_dimension: bool = True
    fp16_mode: bool = False
    max_workspace_size: int = 1 << 30
    strict_type_constraints: bool = False
    enable_fuse: bool = True
    enable_fuse_for_sparsity = False
    verbose_log: bool = False
    algo_selector = None
    timing_cache_prefix: str = ""
    save_timing_cache: bool = False


@dc.dataclass
class LowerTrtInterpreter:
    lower_setting: LowerSetting
    timing_cache_manager: TimingCacheManager

    @classmethod
    def create(cls, lower_setting):
        timing_cache_manager = TimingCacheManager(
            lower_setting.timing_cache_prefix, lower_setting.save_timing_cache
        )
        return LowerTrtInterpreter(lower_setting, timing_cache_manager)

    def __call__(self, mod, input, split_name):
        input_specs_val = (
            self.lower_setting.input_specs
            if self.lower_setting.input_specs
            else InputTensorSpec.from_tensors(input)
        )
        if self.lower_setting.enable_fuse:
            mod = fuse_permute_matmul(mod)
            mod = fuse_permute_linear(mod)
            mod = fuse_unsqueeze_cat_sum(mod)

        # Prepare algorithm selector and timing_cache for TRTInterpreter
        algo_selector = None
        if self.lower_setting.algo_selector:
            algo_selector = self.lower_setting.algo_selector(f"{split_name}.json")
        cache_data = None
        if self.timing_cache_manager:
            try:
                cache_data = self.timing_cache_manager.get_timing_cache_trt(split_name)
            except Exception as e:
                logger.warning(f"Cannot load timing cache for {split_name}: {str(e)}")
                cache_data = None

        import tensorrt as trt  # @manual=//caffe2:torch_fx2trt # noqa: F401

        interpreter = TRTInterpreter(
            mod,
            input_specs=input_specs_val,
            explicit_batch_dimension=self.lower_setting.explicit_batch_dimension,
            logger_level=trt.Logger.VERBOSE
            if self.lower_setting.verbose_log
            else trt.Logger.WARNING,
        )

        interp_result = interpreter.run(
            max_batch_size=self.lower_setting.max_batch_size,
            max_workspace_size=self.lower_setting.max_workspace_size,
            fp16_mode=self.lower_setting.fp16_mode,
            strict_type_constraints=self.lower_setting.strict_type_constraints,
            algorithm_selector=algo_selector,
            timing_cache=cache_data,
        )

        # Update timing cache file if needed
        timing_cache = interp_result.serialized_cache
        if timing_cache and self.timing_cache_manager:
            self.timing_cache_manager.update_timing_cache(split_name, timing_cache)

        return interp_result


class LowerFunc:
    """Signature for fx2trt lower functions"""

    def __call__(
        self,
        module: fx.GraphModule,
        input: Input,
    ) -> nn.Module:
        """Lowers a module using fx2trt

        Args:
            module: module to be lowered
            input: sample input to the module

        Returns:
            the lowered module
        """
        raise NotImplementedError()


def fx2trt_lower(module: nn.Module, sample_input: t.Any) -> fx.GraphModule:
    """Lowers the module using fx2trt

    TODO: @kefeilu: this function's body should be moved into the actual calling
    site in the model publisher workflow, since now the lowering function
    signature (`LowerFunc`) is encapsulated in the `Lowerer` callable class.
    """

    assert isinstance(
        module, fx.GraphModule
    ), f"Expecting fx.GraphModule, got: {type(module)}"
    logger.info(f"Module FX Graph: {module.graph}")
    lower_setting = LowerSetting()
    lower = Lowerer.create(lower_setting=lower_setting)

    module_lowered = lower(module, sample_input)

    assert isinstance(module_lowered, fx.GraphModule)
    return module_lowered


@dc.dataclass(frozen=True)
class Lowerer(LowerFunc):
    """Lowers a module using fx2trt.

    This is a composable class to facilitate fx2trt. A normal fx2trt process
    composes of the following passes to transform an `fx.GraphModule`:

        1. split - the input graph module is split into several sub-nets,
            running either via TensorRT, or via regular CUDA.

    For each split that need to run via TRT, the following passes are
    invoked:

        2. acc_trace - trace the module for TRT conversion
        3. `remove_duplicate_output_args` - since fx2TRT doesn't support duplicate arguments in
            an `fx.GraphModule`'s `output` node, this pass is needed to remove the duplicated
            output args from the split and also update the parent module to fix their uses
            accordingly.
        4. `TRTInterpreter` - runs the acc traced module through `TRTInterpreter`
            to build the TRT engine
        5. Wraps the executable TRT engine into `TRTModule`, which is an `nn.Module`.
        6. The lowered subnet is then set back onto the top-level module

    # TODO: @kefeilu: also incorporates a validator to do inference (and optionally)
    # result comparison along the way.

    Attributes:
        split: the fx2trt split function.
        acc_trace: trace function for TRT conversion.
        remove_duplicate_output_args: moduel transformation pass to remove duplicate args in
            a subnet's `output` node.
        trt_interpret: function to create and run `TRTInterpreter` to convert `fx.GraphModule`
            into a TensorRT engine.
    """

    split: SplitFunc
    acc_trace: t.Callable[[fx.GraphModule, Input], fx.GraphModule]
    remove_duplicate_output_args: RemoveDuplicateOutputArgsFunc
    trt_interpreter: LowerTrtInterpreter
    fp16: bool


    @classmethod
    def create(
        cls,
        lower_setting: LowerSetting,
    ) -> "Lowerer":
        """Instantiate a `Lowerer` instance."""

        return Lowerer(
            split=Splitter.create(not lower_setting.explicit_batch_dimension),
            acc_trace=lambda mod, input: acc_tracer.trace(mod, input),  # type: ignore[arg-type]
            remove_duplicate_output_args=remove_duplicate_output_args,
            trt_interpreter=LowerTrtInterpreter.create(lower_setting),
            fp16=lower_setting.fp16_mode,
        )

    def __call__(
        self,
        module: nn.Module,
        input: Input,
        cuda_graph_batch_size: int = -1,
    ) -> nn.Module:
        """See `LowerFunc` protocol"""
        if self.fp16:
            module.eval().half()
            input = tuple(x.half() if x.dtype == torch.float32 else x for x in input)
        const_split_mod = split_const_subgraphs(module)
        const_split_mod.run_folding()
        module = self.acc_trace(const_split_mod, input)  # type: ignore[misc]
        split_module, splits = self.split(module, input)  # type: ignore[arg-type]
        split_module.eval()  # type: ignore[attr-defined]
        for _split in splits:  # type: ignore[attr-defined]
            if _split.device == "acc":
                # Ensure parent module is updated with the traced sub-net before running
                # remove_duplicate_output_args.
                self.remove_duplicate_output_args(_split.module, [_split.name])  # type: ignore[misc, operator]

                interp_res = self.trt_interpreter(
                    _split.module, _split.input, _split.name
                )

                trt_module = TRTModule(
                    engine=interp_res.engine,
                    input_names=interp_res.input_names,
                    output_names=interp_res.output_names,
                    cuda_graph_batch_size=cuda_graph_batch_size,
                )
                setattr(split_module, _split.name, trt_module)

        return split_module  # type: ignore[return-value]
