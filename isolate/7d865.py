
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor import overrides

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
torch._dynamo.config.load_config(b'\x80\x02}q\x00(X\r\x00\x00\x00log_file_nameq\x01NX\x07\x00\x00\x00verboseq\x02\x89X\x12\x00\x00\x00verify_correctnessq\x03\x89X\x12\x00\x00\x00minimum_call_countq\x04K\x01X\x15\x00\x00\x00dead_code_eliminationq\x05\x88X\x10\x00\x00\x00cache_size_limitq\x06K@X\x0e\x00\x00\x00specialize_intq\x07\x89X\x0b\x00\x00\x00output_codeq\x08NX\x0e\x00\x00\x00dynamic_shapesq\t\x89X\x18\x00\x00\x00assume_static_by_defaultq\n\x89X\x10\x00\x00\x00guard_nn_modulesq\x0b\x89X\x1b\x00\x00\x00traceable_tensor_subclassesq\x0cc__builtin__\nset\nq\r]q\x0e\x85q\x0fRq\x10X\x0f\x00\x00\x00suppress_errorsq\x11\x89X\x15\x00\x00\x00replay_record_enabledq\x12\x89X \x00\x00\x00rewrite_assert_with_torch_assertq\x13\x88X\x12\x00\x00\x00print_graph_breaksq\x14\x89X\x0c\x00\x00\x00print_guardsq\x15\x89X\x15\x00\x00\x00print_specializationsq\x16\x89X\x07\x00\x00\x00disableq\x17\x89X*\x00\x00\x00allowed_functions_module_string_ignorelistq\x18h\r]q\x19(X\x0c\x00\x00\x00torch._primsq\x1aX\x13\x00\x00\x00torch.distributionsq\x1bX\r\x00\x00\x00torch._decompq\x1cX\r\x00\x00\x00torch.testingq\x1dX\x0b\x00\x00\x00torch._refsq\x1ee\x85q\x1fRq X\x12\x00\x00\x00repro_forward_onlyq!\x89X\x0f\x00\x00\x00repro_toleranceq"G?PbM\xd2\xf1\xa9\xfcX\x16\x00\x00\x00capture_scalar_outputsq#\x89X \x00\x00\x00capture_dynamic_output_shape_opsq$\x89X\x19\x00\x00\x00enforce_cond_guards_matchq%\x88X\x0c\x00\x00\x00optimize_ddpq&\x88X\x1a\x00\x00\x00raise_on_ctx_manager_usageq\'\x88X\x1c\x00\x00\x00raise_on_unsafe_aot_autogradq(\x89X\x17\x00\x00\x00raise_on_backend_changeq)\x89X\x18\x00\x00\x00error_on_nested_fx_traceq*\x88X\t\x00\x00\x00allow_rnnq+\x89X\x08\x00\x00\x00base_dirq,X\x19\x00\x00\x00/scratch/shunting/pytorchq-X\x12\x00\x00\x00DEBUG_DIR_VAR_NAMEq.X\x17\x00\x00\x00TORCH_COMPILE_DEBUG_DIRq/X\x0e\x00\x00\x00debug_dir_rootq0X-\x00\x00\x00/scratch/shunting/pytorch/torch_compile_debugq1X)\x00\x00\x00DO_NOT_USE_legacy_non_fake_example_inputsq2\x89X\x13\x00\x00\x00_save_config_ignoreq3h\r]q4(X\x0b\x00\x00\x00repro_afterq5X\x0b\x00\x00\x00repro_levelq6X\x12\x00\x00\x00constant_functionsq7X!\x00\x00\x00skipfiles_inline_module_allowlistq8e\x85q9Rq:u.')
torch._inductor.config.load_config(b'\x80\x02}q\x00(X\x05\x00\x00\x00debugq\x01\x89X\x10\x00\x00\x00disable_progressq\x02\x88X\x10\x00\x00\x00verbose_progressq\x03\x89X\x19\x00\x00\x00aot_codegen_output_prefixq\x04NX\x0b\x00\x00\x00cpp_wrapperq\x05\x89X\x03\x00\x00\x00dceq\x06\x89X\x14\x00\x00\x00static_weight_shapesq\x07\x88X\x0c\x00\x00\x00size_assertsq\x08\x88X\x10\x00\x00\x00pick_loop_ordersq\t\x88X\x0f\x00\x00\x00inplace_buffersq\n\x88X\x11\x00\x00\x00benchmark_harnessq\x0b\x88X\x0f\x00\x00\x00epilogue_fusionq\x0c\x88X\x15\x00\x00\x00epilogue_fusion_firstq\r\x89X\x0f\x00\x00\x00pattern_matcherq\x0e\x88X\n\x00\x00\x00reorderingq\x0f\x89X\x0c\x00\x00\x00max_autotuneq\x10\x88X\x16\x00\x00\x00max_autotune_pointwiseq\x11\x89X\x11\x00\x00\x00max_autotune_gemmq\x12\x89X\x15\x00\x00\x00search_autotune_cacheq\x13\x89X\x17\x00\x00\x00realize_reads_thresholdq\x14K\x04X\x17\x00\x00\x00realize_bytes_thresholdq\x15M\xd0\x07X\x1b\x00\x00\x00realize_acc_reads_thresholdq\x16K\x08X\x0f\x00\x00\x00fallback_randomq\x17\x89X\x12\x00\x00\x00implicit_fallbacksq\x18\x88X\x0b\x00\x00\x00tune_layoutq\x19\x89X\x11\x00\x00\x00aggressive_fusionq\x1a\x89X\x0f\x00\x00\x00max_fusion_sizeq\x1bK@X\x1b\x00\x00\x00unroll_reductions_thresholdq\x1cK\x08X\x0e\x00\x00\x00comment_originq\x1d\x89X\x10\x00\x00\x00benchmark_kernelq\x1e\x89X\x12\x00\x00\x00developer_warningsq\x1f\x88X\x0f\x00\x00\x00compile_threadsq K X\x11\x00\x00\x00global_cache_pathq!NX\x13\x00\x00\x00kernel_name_max_opsq"K\nX\r\x00\x00\x00shape_paddingq#\x89X\x0e\x00\x00\x00permute_fusionq$\x89X\x1a\x00\x00\x00profiler_mark_wrapper_callq%\x89X\x18\x00\x00\x00_raise_error_for_testingq&\x89X\x0c\x00\x00\x00_profile_varq\'X\x00\x00\x00\x00q(X\x11\x00\x00\x00profile_bandwidthq)\x89X\x17\x00\x00\x00profile_bandwidth_regexq*h(X\x0b\x00\x00\x00cpp.threadsq+J\xff\xff\xff\xffX\x13\x00\x00\x00cpp.dynamic_threadsq,\x89X\x0b\x00\x00\x00cpp.simdlenq-NX\x12\x00\x00\x00cpp.min_chunk_sizeq.M\x00\x10X\x07\x00\x00\x00cpp.cxxq/NX\x03\x00\x00\x00g++q0\x86q1X\x19\x00\x00\x00cpp.enable_kernel_profileq2\x89X\x12\x00\x00\x00cpp.weight_prepackq3\x88X\x11\x00\x00\x00triton.cudagraphsq4\x89X\x16\x00\x00\x00triton.cudagraph_treesq5\x89X\x1c\x00\x00\x00triton.debug_cudagraph_treesq6\x88X\x1c\x00\x00\x00triton.skip_cudagraph_warmupq7\x89X\x17\x00\x00\x00triton.debug_sync_graphq8\x89X\x18\x00\x00\x00triton.debug_sync_kernelq9\x89X\x15\x00\x00\x00triton.dense_indexingq:\x89X\x10\x00\x00\x00triton.max_tilesq;K\x02X\x19\x00\x00\x00triton.autotune_pointwiseq<\x88X\'\x00\x00\x00triton.tiling_prevents_pointwise_fusionq=\x88X\'\x00\x00\x00triton.tiling_prevents_reduction_fusionq>\x88X\x1a\x00\x00\x00triton.unique_kernel_namesq?\x89X\x18\x00\x00\x00triton.descriptive_namesq@X\r\x00\x00\x00original_atenqAX\x1c\x00\x00\x00triton.persistent_reductionsqB\x88X\x10\x00\x00\x00triton.max_blockqC}qD(X\x01\x00\x00\x00XqEM\x00\x08X\x01\x00\x00\x00YqFM\x00\x04X\x01\x00\x00\x00ZqGM\x00\x04uX\r\x00\x00\x00trace.enabledqH\x89X\x0f\x00\x00\x00trace.debug_logqI\x88X\x0e\x00\x00\x00trace.info_logqJ\x89X\x0e\x00\x00\x00trace.fx_graphqK\x88X\x1a\x00\x00\x00trace.fx_graph_transformedqL\x88X\x13\x00\x00\x00trace.ir_pre_fusionqM\x88X\x14\x00\x00\x00trace.ir_post_fusionqN\x88X\x11\x00\x00\x00trace.output_codeqO\x88X\x13\x00\x00\x00trace.graph_diagramqP\x89X\x15\x00\x00\x00trace.compile_profileqQ\x89X\x13\x00\x00\x00_save_config_ignoreqRc__builtin__\nset\nqS]qTX\x10\x00\x00\x00trace.upload_tarqUa\x85qVRqWX\x13\x00\x00\x00autotune_in_subprocqX\x89X\x0e\x00\x00\x00conv_1x1_as_mmqY\x89X\x14\x00\x00\x00is_nightly_or_sourceqZ\x88X\x1d\x00\x00\x00triton.fast_cudagraph_assertsq[\x88X\x1d\x00\x00\x00triton.slow_cudagraph_assertsq\\\x89u.')
torch._functorch.config.load_config(b'\x80\x02}q\x00(X\x11\x00\x00\x00use_functionalizeq\x01\x88X\x0f\x00\x00\x00use_fake_tensorq\x02\x88X\x16\x00\x00\x00fake_tensor_allow_metaq\x03\x88X\x0c\x00\x00\x00debug_assertq\x04\x89X\x14\x00\x00\x00debug_fake_cross_refq\x05\x89X\x11\x00\x00\x00debug_partitionerq\x06\x89X\x0c\x00\x00\x00debug_graphsq\x07\x89X\x0b\x00\x00\x00debug_jointq\x08\x89X\x14\x00\x00\x00static_weight_shapesq\t\x88X\x03\x00\x00\x00cseq\n\x88X\x10\x00\x00\x00max_dist_from_bwq\x0bK\x03X\t\x00\x00\x00log_levelq\x0cK\x14u.')


# REPLACEABLE COMMENT FOR TESTING PURPOSES


# torch version: 2.1.0a0+gitbe645eb
# torch cuda version: 11.6
# torch git version: be645eb93d597a34891418b9dc0e4656db970d4a


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2022 NVIDIA Corporation 
# Built on Thu_Feb_10_18:23:41_PST_2022 
# Cuda compilation tools, release 11.6, V11.6.112 
# Build cuda_11.6.r11.6/compiler.30978841_0 

# GPU Hardware Info: 
# NVIDIA A100-SXM4-40GB : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1):
        convert_element_type = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        mul = torch.ops.aten.mul.Tensor(convert_element_type, 1.0)
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        mul_1 = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.0)
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(arg5_1, torch.float32);  arg5_1 = None
        mul_2 = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.0)
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(arg8_1, torch.float32);  arg8_1 = None
        mul_3 = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.0)
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(arg10_1, torch.float32);  arg10_1 = None
        mul_4 = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.0)
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(arg12_1, torch.float32);  arg12_1 = None
        mul_5 = torch.ops.aten.mul.Tensor(convert_element_type_5, 1.0)
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(arg18_1, torch.float32);  arg18_1 = None
        le = torch.ops.aten.le.Scalar(convert_element_type_5, 0);  convert_element_type_5 = None
        mul_6 = torch.ops.aten.mul.Tensor(convert_element_type_6, 1)
        mul_7 = torch.ops.aten.mul.Tensor(mul_6, 1.7580993408473766);  mul_6 = None
        exp = torch.ops.aten.exp.default(mul_5);  mul_5 = None
        mul_8 = torch.ops.aten.mul.Tensor(mul_7, exp);  mul_7 = exp = None
        mul_9 = torch.ops.aten.mul.Tensor(convert_element_type_6, 1.0507009873554805);  convert_element_type_6 = None
        where = torch.ops.aten.where.self(le, mul_8, mul_9);  le = mul_8 = mul_9 = None
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(where, torch.float16);  where = None
        mm = torch.ops.aten.mm.default(convert_element_type_7, arg13_1);  arg13_1 = None
        permute = torch.ops.aten.permute.default(convert_element_type_7, [1, 0])
        mm_1 = torch.ops.aten.mm.default(permute, arg11_1);  permute = arg11_1 = None
        permute_1 = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(convert_element_type_7, [0], True);  convert_element_type_7 = None
        view = torch.ops.aten.view.default(sum_1, [197951]);  sum_1 = None
        permute_2 = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        convert_element_type_8 = torch.ops.prims.convert_element_type.default(permute_2, torch.float32);  permute_2 = None
        convert_element_type_9 = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        convert_element_type_10 = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        le_1 = torch.ops.aten.le.Scalar(convert_element_type_4, 0);  convert_element_type_4 = None
        mul_10 = torch.ops.aten.mul.Tensor(convert_element_type_10, 1)
        mul_11 = torch.ops.aten.mul.Tensor(mul_10, 1.7580993408473766);  mul_10 = None
        exp_1 = torch.ops.aten.exp.default(mul_4);  mul_4 = None
        mul_12 = torch.ops.aten.mul.Tensor(mul_11, exp_1);  mul_11 = exp_1 = None
        mul_13 = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.0507009873554805);  convert_element_type_10 = None
        where_1 = torch.ops.aten.where.self(le_1, mul_12, mul_13);  le_1 = mul_12 = mul_13 = None
        convert_element_type_11 = torch.ops.prims.convert_element_type.default(where_1, torch.float16);  where_1 = None
        mm_2 = torch.ops.aten.mm.default(convert_element_type_11, arg14_1);  arg14_1 = None
        permute_3 = torch.ops.aten.permute.default(convert_element_type_11, [1, 0])
        mm_3 = torch.ops.aten.mm.default(permute_3, arg9_1);  permute_3 = arg9_1 = None
        permute_4 = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(convert_element_type_11, [0], True);  convert_element_type_11 = None
        view_1 = torch.ops.aten.view.default(sum_2, [512]);  sum_2 = None
        permute_5 = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        convert_element_type_12 = torch.ops.prims.convert_element_type.default(permute_5, torch.float32);  permute_5 = None
        convert_element_type_13 = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        convert_element_type_14 = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        le_2 = torch.ops.aten.le.Scalar(convert_element_type_3, 0);  convert_element_type_3 = None
        mul_14 = torch.ops.aten.mul.Tensor(convert_element_type_14, 1)
        mul_15 = torch.ops.aten.mul.Tensor(mul_14, 1.7580993408473766);  mul_14 = None
        exp_2 = torch.ops.aten.exp.default(mul_3);  mul_3 = None
        mul_16 = torch.ops.aten.mul.Tensor(mul_15, exp_2);  mul_15 = exp_2 = None
        mul_17 = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.0507009873554805);  convert_element_type_14 = None
        where_2 = torch.ops.aten.where.self(le_2, mul_16, mul_17);  le_2 = mul_16 = mul_17 = None
        convert_element_type_15 = torch.ops.prims.convert_element_type.default(where_2, torch.float16);  where_2 = None
        mm_4 = torch.ops.aten.mm.default(convert_element_type_15, arg15_1);  arg15_1 = None
        permute_6 = torch.ops.aten.permute.default(convert_element_type_15, [1, 0])
        mm_5 = torch.ops.aten.mm.default(permute_6, arg7_1);  permute_6 = arg7_1 = None
        permute_7 = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(convert_element_type_15, [0], True);  convert_element_type_15 = None
        view_2 = torch.ops.aten.view.default(sum_3, [512]);  sum_3 = None
        permute_8 = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
        convert_element_type_16 = torch.ops.prims.convert_element_type.default(permute_8, torch.float32);  permute_8 = None
        convert_element_type_17 = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        philox_rand_like = torch.ops.prims.philox_rand_like.default(mm_4, arg6_1, 0);  arg6_1 = None
        gt = torch.ops.aten.gt.Scalar(philox_rand_like, 0.8);  philox_rand_like = None
        convert_element_type_18 = torch.ops.prims.convert_element_type.default(gt, torch.float16);  gt = None
        mul_18 = torch.ops.aten.mul.Tensor(convert_element_type_18, mm_4);  convert_element_type_18 = mm_4 = None
        mul_19 = torch.ops.aten.mul.Tensor(mul_18, 5.000000000000001);  mul_18 = None
        convert_element_type_19 = torch.ops.prims.convert_element_type.default(mul_19, torch.float32);  mul_19 = None
        le_3 = torch.ops.aten.le.Scalar(convert_element_type_2, 0);  convert_element_type_2 = None
        mul_20 = torch.ops.aten.mul.Tensor(convert_element_type_19, 1)
        mul_21 = torch.ops.aten.mul.Tensor(mul_20, 1.7580993408473766);  mul_20 = None
        exp_3 = torch.ops.aten.exp.default(mul_2);  mul_2 = None
        mul_22 = torch.ops.aten.mul.Tensor(mul_21, exp_3);  mul_21 = exp_3 = None
        mul_23 = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.0507009873554805);  convert_element_type_19 = None
        where_3 = torch.ops.aten.where.self(le_3, mul_22, mul_23);  le_3 = mul_22 = mul_23 = None
        convert_element_type_20 = torch.ops.prims.convert_element_type.default(where_3, torch.float16);  where_3 = None
        mm_6 = torch.ops.aten.mm.default(convert_element_type_20, arg16_1);  arg16_1 = None
        permute_9 = torch.ops.aten.permute.default(convert_element_type_20, [1, 0])
        mm_7 = torch.ops.aten.mm.default(permute_9, arg4_1);  permute_9 = arg4_1 = None
        permute_10 = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(convert_element_type_20, [0], True);  convert_element_type_20 = None
        view_3 = torch.ops.aten.view.default(sum_4, [1024]);  sum_4 = None
        permute_11 = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        convert_element_type_21 = torch.ops.prims.convert_element_type.default(permute_11, torch.float32);  permute_11 = None
        convert_element_type_22 = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        convert_element_type_23 = torch.ops.prims.convert_element_type.default(mm_6, torch.float32);  mm_6 = None
        le_4 = torch.ops.aten.le.Scalar(convert_element_type_1, 0);  convert_element_type_1 = None
        mul_24 = torch.ops.aten.mul.Tensor(convert_element_type_23, 1)
        mul_25 = torch.ops.aten.mul.Tensor(mul_24, 1.7580993408473766);  mul_24 = None
        exp_4 = torch.ops.aten.exp.default(mul_1);  mul_1 = None
        mul_26 = torch.ops.aten.mul.Tensor(mul_25, exp_4);  mul_25 = exp_4 = None
        mul_27 = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.0507009873554805);  convert_element_type_23 = None
        where_4 = torch.ops.aten.where.self(le_4, mul_26, mul_27);  le_4 = mul_26 = mul_27 = None
        convert_element_type_24 = torch.ops.prims.convert_element_type.default(where_4, torch.float16);  where_4 = None
        mm_8 = torch.ops.aten.mm.default(convert_element_type_24, arg17_1);  arg17_1 = None
        permute_12 = torch.ops.aten.permute.default(convert_element_type_24, [1, 0])
        mm_9 = torch.ops.aten.mm.default(permute_12, arg2_1);  permute_12 = arg2_1 = None
        permute_13 = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(convert_element_type_24, [0], True);  convert_element_type_24 = None
        view_4 = torch.ops.aten.view.default(sum_5, [512]);  sum_5 = None
        permute_14 = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        convert_element_type_25 = torch.ops.prims.convert_element_type.default(permute_14, torch.float32);  permute_14 = None
        convert_element_type_26 = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None
        convert_element_type_27 = torch.ops.prims.convert_element_type.default(mm_8, torch.float32);  mm_8 = None
        le_5 = torch.ops.aten.le.Scalar(convert_element_type, 0);  convert_element_type = None
        mul_28 = torch.ops.aten.mul.Tensor(convert_element_type_27, 1)
        mul_29 = torch.ops.aten.mul.Tensor(mul_28, 1.7580993408473766);  mul_28 = None
        exp_5 = torch.ops.aten.exp.default(mul);  mul = None
        mul_30 = torch.ops.aten.mul.Tensor(mul_29, exp_5);  mul_29 = exp_5 = None
        mul_31 = torch.ops.aten.mul.Tensor(convert_element_type_27, 1.0507009873554805);  convert_element_type_27 = None
        where_5 = torch.ops.aten.where.self(le_5, mul_30, mul_31);  le_5 = mul_30 = mul_31 = None
        convert_element_type_28 = torch.ops.prims.convert_element_type.default(where_5, torch.float16);  where_5 = None
        permute_15 = torch.ops.aten.permute.default(convert_element_type_28, [1, 0])
        mm_10 = torch.ops.aten.mm.default(permute_15, arg0_1);  permute_15 = arg0_1 = None
        permute_16 = torch.ops.aten.permute.default(mm_10, [1, 0]);  mm_10 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(convert_element_type_28, [0], True);  convert_element_type_28 = None
        view_5 = torch.ops.aten.view.default(sum_6, [512]);  sum_6 = None
        permute_17 = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
        convert_element_type_29 = torch.ops.prims.convert_element_type.default(permute_17, torch.float32);  permute_17 = None
        convert_element_type_30 = torch.ops.prims.convert_element_type.default(view_5, torch.float32);  view_5 = None
        return [convert_element_type_29, convert_element_type_25, convert_element_type_21, convert_element_type_30, convert_element_type_26, convert_element_type_22, convert_element_type_16, convert_element_type_12, convert_element_type_8, convert_element_type_17, convert_element_type_13, convert_element_type_9, None]
        
args = []
args.append(rand_strided((256, 197951), (197951, 1), torch.float16, 'cuda'))  # shape (256, 197951), stride (197951, 1)
args.append(rand_strided((256, 512), (512, 1), torch.float16, 'cuda'))  # shape (256, 512), stride (512, 1)
args.append(rand_strided((256, 512), (512, 1), torch.float16, 'cuda'))  # shape (256, 512), stride (512, 1)
args.append(rand_strided((256, 512), (512, 1), torch.float16, 'cuda'))  # shape (256, 512), stride (512, 1)
args.append(rand_strided((256, 512), (512, 1), torch.float16, 'cuda'))  # shape (256, 512), stride (512, 1)
args.append(rand_strided((256, 1024), (1024, 1), torch.float16, 'cuda'))  # shape (256, 1024), stride (1024, 1)
args.append(rand_strided((), (), torch.int64, 'cuda'))  # shape (), stride ()
args.append(rand_strided((256, 1024), (1024, 1), torch.float16, 'cuda'))  # shape (256, 1024), stride (1024, 1)
args.append(rand_strided((256, 512), (512, 1), torch.float16, 'cuda'))  # shape (256, 512), stride (512, 1)
args.append(rand_strided((256, 512), (512, 1), torch.float16, 'cuda'))  # shape (256, 512), stride (512, 1)
args.append(rand_strided((256, 512), (512, 1), torch.float16, 'cuda'))  # shape (256, 512), stride (512, 1)
args.append(rand_strided((256, 512), (512, 1), torch.float16, 'cuda'))  # shape (256, 512), stride (512, 1)
args.append(rand_strided((256, 197951), (197951, 1), torch.float16, 'cuda'))  # shape (256, 197951), stride (197951, 1)
args.append(rand_strided((197951, 512), (512, 1), torch.float16, 'cuda'))  # shape (197951, 512), stride (512, 1)
args.append(rand_strided((512, 512), (512, 1), torch.float16, 'cuda'))  # shape (512, 512), stride (512, 1)
args.append(rand_strided((512, 1024), (1024, 1), torch.float16, 'cuda'))  # shape (512, 1024), stride (1024, 1)
args.append(rand_strided((1024, 512), (512, 1), torch.float16, 'cuda'))  # shape (1024, 512), stride (512, 1)
args.append(rand_strided((512, 512), (512, 1), torch.float16, 'cuda'))  # shape (512, 512), stride (512, 1)
args.append(rand_strided((256, 197951), (197951, 1), torch.float16, 'cuda'))  # shape (256, 197951), stride (197951, 1)
mod = make_fx(Repro(), tracing_mode='real')(*args)

from torch._dynamo.debug_utils import inductor_fails

if inductor_fails(mod, args):
    exit(1)
else:
    exit(0)
