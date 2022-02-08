import dataclasses as dc
import logging
from typing import Callable, List, Any, Sequence, Type, Set, Optional

import tensorrt as trt
import torch
import torch.fx as fx
import torch.fx.experimental.fx_acc.acc_tracer as acc_tracer
import torch.nn as nn
from torch.fx.experimental.const_fold import split_const_subgraphs
from torch.fx.passes.splitter_base import SplitResult
import torch.fx.experimental.fx2trt.diagnostics as diagnostics

from .fx2trt import (
    TRTInterpreter,
)
from .input_tensor_spec import (
    InputTensorSpec,
)
from .passes.fuse_pass import (
    fuse_permute_linear,
    fuse_permute_matmul,
    fuse_unsqueeze_cat_sum,
)
from .passes.remove_duplicate_output_args import (
    remove_duplicate_output_args,
)
from .tools.timing_cache_utils import (
    TimingCacheManager,
)
from .tools.trt_splitter import TRTSplitter, TRTSplitterSetting
from .trt_module import (
    TRTModule,
)


logger = logging.getLogger(__name__)

Input = Sequence[Any]


def lower_to_trt(
    module: nn.Module,
    input,
    max_batch_size: int = 2048,
    max_workspace_size=1 << 25,
    explicit_batch_dimension=False,
    fp16_mode=True,
    enable_fuse=True,
    verbose_log=False,
    timing_cache_prefix="",
    save_timing_cache=False,
    cuda_graph_batch_size=-1,
) -> nn.Module:
    """
    Takes in original module, input and lowering setting, run lowering workflow to turn module
    into lowered module, or so called TRTModule.

    Args:
        module: Original module for lowering.
        input: Input for module.
        max_batch_size: Maximum batch size (must be >= 1 to be set, 0 means not set)
        max_workspace_size: Maximum size of workspace given to TensorRT.
        explicit_batch_dimension: Use explicit batch dimension in TensorRT if set True, otherwise use implicit batch dimension.
        fp16_mode: fp16 config given to TRTModule.
        enable_fuse: Enable pass fusion during lowering if set to true. l=Lowering will try to find pattern defined
        in torch.fx.experimental.fx2trt.passes from original module, and replace with optimized pass before apply lowering.
        verbose_log: Enable verbose log for TensorRT if set True.
        timing_cache_prefix: Timing cache file name for timing cache used by fx2trt.
        save_timing_cache: Update timing cache with current timing cache data if set to True.
        cuda_graph_batch_size: Cuda graph batch size, default to be -1.

    Returns:
        A torch.nn.Module lowered by TensorRT.
    """
    lower_setting = LowerSetting(
        max_batch_size=max_batch_size,
        max_workspace_size=max_workspace_size,
        explicit_batch_dimension=explicit_batch_dimension,
        fp16_mode=fp16_mode,
        enable_fuse=enable_fuse,
        verbose_log=verbose_log,
        timing_cache_prefix=timing_cache_prefix,
        save_timing_cache=save_timing_cache,
    )
    lowerer = Lowerer.create(lower_setting=lower_setting)
    return lowerer(module, input)


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

    explicit_precision: Use explicit precision during lowering.

    fp16_mode: Enable FP16 dtype during lowering.

    int8_mode: Enable Int8 dtype during lowering.

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

    ast_rewriter_allow_list (Optional[Set[nn.Module]]): Optional allow list of
    modules that need AST rewriting. This is aiming to eliminate input variable involve in
    exception checking control flow.

    leaf_module_list (Optional[Set[nn.Module]]): Optional leaf module list where
    modules will not be traced into.
    """
    max_batch_size: int = 2048
    input_specs: List[InputTensorSpec] = dc.field(default_factory=list)
    explicit_batch_dimension: bool = True
    explicit_precision: bool = False
    fp16_mode: bool = False
    int8_mode: bool = False
    max_workspace_size: int = 1 << 30
    strict_type_constraints: bool = False
    enable_fuse: bool = True
    enable_fuse_for_sparsity = False
    verbose_log: bool = False
    algo_selector = None
    timing_cache_prefix: str = ""
    save_timing_cache: bool = False
    ast_rewriter_allow_list: Optional[Set[Type[nn.Module]]] = None
    leaf_module_list: Optional[Set[Type[nn.Module]]] = None


def run_const_fold(traced_mod: torch.fx.GraphModule) -> torch.fx.GraphModule:
    # Now we do constant folding on traced module. We want to skip pattern like
    # weights -> quant -> dequant -> op during constant folding when the model is
    # a quantized int8 model.
    def skip_folding_quant_dequant(node: torch.fx.Node):
        if node.target != torch.quantize_per_tensor:
            return False
        # If quantize_per_node -> dequantize, then skip folding.
        for user in node.users:
            if user.target == "dequantize":
                return True
        return False

    const_split_mod = split_const_subgraphs(traced_mod, skip_folding_quant_dequant)
    const_split_mod.run_folding()
    return const_split_mod


def default_split_function(model: fx.GraphModule, inputs: Input, lower_setting: LowerSetting) -> SplitResult:
    splitter_setting = TRTSplitterSetting()
    splitter_setting.use_implicit_batch_dim = not lower_setting.explicit_batch_dimension
    # TODO: avoid hardcode here by introducing another flag in lowering setting.
    splitter_setting.min_acc_module_size = 10
    splitter = TRTSplitter(model, inputs, settings=splitter_setting)
    splitter.node_support_preview()
    return splitter.generate_split_results()


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

        interpreter = TRTInterpreter(
            mod,
            input_specs=input_specs_val,
            explicit_batch_dimension=self.lower_setting.explicit_batch_dimension,
            explicit_precision=self.lower_setting.explicit_precision,
            logger_level=trt.Logger.VERBOSE
            if self.lower_setting.verbose_log
            else trt.Logger.WARNING,
        )

        interp_result = interpreter.run(
            max_batch_size=self.lower_setting.max_batch_size,
            max_workspace_size=self.lower_setting.max_workspace_size,
            fp16_mode=self.lower_setting.fp16_mode,
            int8_mode=self.lower_setting.int8_mode,
            strict_type_constraints=self.lower_setting.strict_type_constraints,
            algorithm_selector=algo_selector,
            timing_cache=cache_data,
            profiling_verbosity=trt.ProfilingVerbosity.DETAILED,
        )

        # Update timing cache file if needed
        timing_cache = interp_result.serialized_cache
        if timing_cache and self.timing_cache_manager:
            self.timing_cache_manager.update_timing_cache(split_name, timing_cache)

        return interp_result


@dc.dataclass(frozen=True)
class Lowerer:
    """Lowers a module using fx2trt.

    This is a composable class to facilitate fx2trt. A normal fx2trt process
    composes of the following passes to transform an `fx.GraphModule`:

        1. trace - use torch.fx to trace the module so we can get the graph
            representation of the model.
        2. split - the graph module is split into several submodules,
            running either via TensorRT, or via regular CUDA.

    For each split that need to run via TRT, the following passes are
    invoked:

        3. `TRTInterpreter` - build the TRT engine for the submodule that
            can be supported through `TRTInterpreter`.
        4. Wraps the executable TRT engine into `TRTModule`, which is an `nn.Module`.
        5. The converted submodule is then set back onto the top-level module

    # TODO: @kefeilu: also incorporates a validator to do inference (and optionally)
    # result comparison along the way.

    Attributes:
        trace_func: fx trace function for TRT conversion.
        split_func: the fx2trt split function.
        trt_interpret: function to create and run `TRTInterpreter` to convert `fx.GraphModule`
            into a TensorRT engine.
        lower_setting: see above LowerSetting class for the details.
    """

    trace_func: Callable[[nn.Module, Input], fx.GraphModule]
    split_func: Callable[[fx.GraphModule, Input, LowerSetting], SplitResult]
    trt_interpreter: LowerTrtInterpreter
    lower_setting: LowerSetting
    trt_module_observer: Optional[Callable[[str, nn.Module, List[torch.Tensor]], None]]

    @classmethod
    def create(
        cls,
        lower_setting: LowerSetting,
        trt_module_observer: Optional[Callable[[str, nn.Module, List[torch.Tensor]], None]] = None,
    ) -> "Lowerer":
        """Instantiate a `Lowerer` instance."""

        return cls(
            trace_func=lambda module, inputs: acc_tracer.trace(
                module,
                inputs,  # type: ignore[arg-type]
                ast_rewriter_allow_list=lower_setting.ast_rewriter_allow_list,
                leaf_module_list=lower_setting.leaf_module_list),  # type: ignore[arg-type]
            split_func=default_split_function,
            trt_interpreter=LowerTrtInterpreter.create(lower_setting),
            lower_setting=lower_setting,
            trt_module_observer=trt_module_observer,
        )

    def __call__(
        self,
        module: nn.Module,
        inputs: Input,
        cuda_graph_batch_size: int = -1,
    ) -> nn.Module:
        module.eval()

        if self.lower_setting.fp16_mode:
            module.half()
            inputs = tuple(x.half() if x.dtype == torch.float32 else x for x in inputs)

        # Ensure ast_rewrite is done for input module before const_fold.
        traced_mod = self.trace_func(module, inputs)  # type: ignore[misc]

        # Run const folding.
        traced_mod = run_const_fold(traced_mod)

        # Retrace here to eliminate no-op introduced by const folding and map new introduced
        # nodes to acc op nodes.
        traced_mod = self.trace_func(traced_mod, inputs)  # type: ignore[misc]

        # Run split.
        split_result = self.split_func(traced_mod, inputs, self.lower_setting)  # type: ignore[misc,operator]

        # TesnorRT doesn't like duplicate outputs. Run this pass to eliminate such case.
        remove_duplicate_output_args(split_result.split_module, split_result.submodule_inputs.keys())

        for submod_name, submod_inputs in split_result.submodule_inputs.items():
            submod = getattr(split_result.split_module, submod_name)
            diagnostics.write(f"lower.split.{submod_name}.module.graph", lambda: str(submod.graph))
            if self.trt_module_observer:
                self.trt_module_observer(submod_name, submod, submod_inputs)  # type: ignore[arg-type]

            # We only lower acc submodules.
            if not submod_name.startswith(split_result.non_acc_submodule_prefix):
                interp_res = self.trt_interpreter(
                    submod, submod_inputs, submod_name
                )

                trt_module = TRTModule(
                    engine=interp_res.engine,
                    input_names=interp_res.input_names,
                    output_names=interp_res.output_names,
                    cuda_graph_batch_size=cuda_graph_batch_size,
                )
                setattr(split_result.split_module, submod_name, trt_module)

        return split_result.split_module
