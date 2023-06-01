
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
from torch.fx.experimental.proxy_tensor import make_fx

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
torch._dynamo.config.load_config(b'\x80\x02}q\x00(X\r\x00\x00\x00log_file_nameq\x01NX\x07\x00\x00\x00verboseq\x02\x89X\x12\x00\x00\x00verify_correctnessq\x03\x89X\x12\x00\x00\x00minimum_call_countq\x04K\x01X\x15\x00\x00\x00dead_code_eliminationq\x05\x88X\x10\x00\x00\x00cache_size_limitq\x06K@X\x0e\x00\x00\x00specialize_intq\x07\x89X\x0b\x00\x00\x00output_codeq\x08NX\x0e\x00\x00\x00dynamic_shapesq\t\x89X\x18\x00\x00\x00assume_static_by_defaultq\n\x89X\x10\x00\x00\x00guard_nn_modulesq\x0b\x89X\x1b\x00\x00\x00traceable_tensor_subclassesq\x0cc__builtin__\nset\nq\r]q\x0e\x85q\x0fRq\x10X\x0f\x00\x00\x00suppress_errorsq\x11\x89X\x15\x00\x00\x00replay_record_enabledq\x12\x89X \x00\x00\x00rewrite_assert_with_torch_assertq\x13\x88X\x12\x00\x00\x00print_graph_breaksq\x14\x89X\x0c\x00\x00\x00print_guardsq\x15\x89X\x15\x00\x00\x00print_specializationsq\x16\x89X\x07\x00\x00\x00disableq\x17\x89X*\x00\x00\x00allowed_functions_module_string_ignorelistq\x18h\r]q\x19(X\r\x00\x00\x00torch._decompq\x1aX\x0c\x00\x00\x00torch._primsq\x1bX\x0b\x00\x00\x00torch._refsq\x1cX\r\x00\x00\x00torch.testingq\x1dX\x13\x00\x00\x00torch.distributionsq\x1ee\x85q\x1fRq X\x12\x00\x00\x00repro_forward_onlyq!\x89X\x0f\x00\x00\x00repro_toleranceq"G?PbM\xd2\xf1\xa9\xfcX\x16\x00\x00\x00capture_scalar_outputsq#\x89X \x00\x00\x00capture_dynamic_output_shape_opsq$\x89X\x19\x00\x00\x00enforce_cond_guards_matchq%\x88X\x0c\x00\x00\x00optimize_ddpq&\x88X\x1a\x00\x00\x00raise_on_ctx_manager_usageq\'\x88X\x1c\x00\x00\x00raise_on_unsafe_aot_autogradq(\x89X\x17\x00\x00\x00raise_on_backend_changeq)\x89X\x18\x00\x00\x00error_on_nested_fx_traceq*\x88X\t\x00\x00\x00allow_rnnq+\x89X\x08\x00\x00\x00base_dirq,X\x1d\x00\x00\x00/scratch/chilli/fresh/pytorchq-X\x12\x00\x00\x00DEBUG_DIR_VAR_NAMEq.X\x17\x00\x00\x00TORCH_COMPILE_DEBUG_DIRq/X\x0e\x00\x00\x00debug_dir_rootq0X1\x00\x00\x00/scratch/chilli/fresh/pytorch/torch_compile_debugq1X)\x00\x00\x00DO_NOT_USE_legacy_non_fake_example_inputsq2\x89X\x13\x00\x00\x00_save_config_ignoreq3h\r]q4(X\x0b\x00\x00\x00repro_afterq5X\x12\x00\x00\x00constant_functionsq6X!\x00\x00\x00skipfiles_inline_module_allowlistq7X\x0b\x00\x00\x00repro_levelq8e\x85q9Rq:u.')
torch._inductor.config.load_config(b'\x80\x02}q\x00(X\x05\x00\x00\x00debugq\x01\x89X\x10\x00\x00\x00disable_progressq\x02\x88X\x10\x00\x00\x00verbose_progressq\x03\x89X\x19\x00\x00\x00aot_codegen_output_prefixq\x04NX\x0b\x00\x00\x00cpp_wrapperq\x05\x89X\x03\x00\x00\x00dceq\x06\x89X\x14\x00\x00\x00static_weight_shapesq\x07\x88X\x0c\x00\x00\x00size_assertsq\x08\x88X\x10\x00\x00\x00pick_loop_ordersq\t\x88X\x0f\x00\x00\x00inplace_buffersq\n\x88X\x11\x00\x00\x00benchmark_harnessq\x0b\x88X\x0f\x00\x00\x00epilogue_fusionq\x0c\x89X\x15\x00\x00\x00epilogue_fusion_firstq\r\x89X\x0f\x00\x00\x00pattern_matcherq\x0e\x88X\n\x00\x00\x00reorderingq\x0f\x89X\x0c\x00\x00\x00max_autotuneq\x10\x89X\x16\x00\x00\x00max_autotune_pointwiseq\x11\x89X\x11\x00\x00\x00max_autotune_gemmq\x12\x89X\x15\x00\x00\x00search_autotune_cacheq\x13\x89X\x17\x00\x00\x00realize_reads_thresholdq\x14K\x04X\x17\x00\x00\x00realize_bytes_thresholdq\x15M\xd0\x07X\x1b\x00\x00\x00realize_acc_reads_thresholdq\x16K\x08X\x0f\x00\x00\x00fallback_randomq\x17\x89X\x12\x00\x00\x00implicit_fallbacksq\x18\x88X\x0b\x00\x00\x00tune_layoutq\x19\x89X\x11\x00\x00\x00aggressive_fusionq\x1a\x89X\x0f\x00\x00\x00max_fusion_sizeq\x1bK@X\x1b\x00\x00\x00unroll_reductions_thresholdq\x1cK\x08X\x0e\x00\x00\x00comment_originq\x1d\x89X\x10\x00\x00\x00benchmark_kernelq\x1e\x89X\x12\x00\x00\x00developer_warningsq\x1f\x88X\x0f\x00\x00\x00compile_threadsq K X\x11\x00\x00\x00global_cache_pathq!NX\x13\x00\x00\x00kernel_name_max_opsq"K\nX\r\x00\x00\x00shape_paddingq#\x89X\x0e\x00\x00\x00permute_fusionq$\x89X\x1a\x00\x00\x00profiler_mark_wrapper_callq%\x89X\x18\x00\x00\x00_raise_error_for_testingq&\x89X\x0c\x00\x00\x00_profile_varq\'X\x00\x00\x00\x00q(X\x11\x00\x00\x00profile_bandwidthq)\x89X\x17\x00\x00\x00profile_bandwidth_regexq*h(X\x0b\x00\x00\x00cpp.threadsq+J\xff\xff\xff\xffX\x13\x00\x00\x00cpp.dynamic_threadsq,\x89X\x0b\x00\x00\x00cpp.simdlenq-NX\x12\x00\x00\x00cpp.min_chunk_sizeq.M\x00\x10X\x07\x00\x00\x00cpp.cxxq/NX\x03\x00\x00\x00g++q0\x86q1X\x19\x00\x00\x00cpp.enable_kernel_profileq2\x89X\x12\x00\x00\x00cpp.weight_prepackq3\x88X\x11\x00\x00\x00triton.cudagraphsq4\x89X\x16\x00\x00\x00triton.cudagraph_treesq5\x89X\x1c\x00\x00\x00triton.debug_cudagraph_treesq6\x88X\x1c\x00\x00\x00triton.skip_cudagraph_warmupq7\x89X\x17\x00\x00\x00triton.debug_sync_graphq8\x89X\x18\x00\x00\x00triton.debug_sync_kernelq9\x89X\x15\x00\x00\x00triton.dense_indexingq:\x89X\x10\x00\x00\x00triton.max_tilesq;K\x02X\x19\x00\x00\x00triton.autotune_pointwiseq<\x88X\'\x00\x00\x00triton.tiling_prevents_pointwise_fusionq=\x88X\'\x00\x00\x00triton.tiling_prevents_reduction_fusionq>\x88X\x1a\x00\x00\x00triton.unique_kernel_namesq?\x89X\x18\x00\x00\x00triton.descriptive_namesq@X\r\x00\x00\x00original_atenqAX\x1c\x00\x00\x00triton.persistent_reductionsqB\x88X\x10\x00\x00\x00triton.max_blockqC}qD(X\x01\x00\x00\x00XqEM\x00\x08X\x01\x00\x00\x00YqFM\x00\x04X\x01\x00\x00\x00ZqGM\x00\x04uX\r\x00\x00\x00trace.enabledqH\x89X\x0f\x00\x00\x00trace.debug_logqI\x88X\x0e\x00\x00\x00trace.info_logqJ\x89X\x0e\x00\x00\x00trace.fx_graphqK\x88X\x1a\x00\x00\x00trace.fx_graph_transformedqL\x88X\x13\x00\x00\x00trace.ir_pre_fusionqM\x88X\x14\x00\x00\x00trace.ir_post_fusionqN\x88X\x11\x00\x00\x00trace.output_codeqO\x88X\x13\x00\x00\x00trace.graph_diagramqP\x89X\x15\x00\x00\x00trace.compile_profileqQ\x89X\x13\x00\x00\x00_save_config_ignoreqRc__builtin__\nset\nqS]qTX\x10\x00\x00\x00trace.upload_tarqUa\x85qVRqWu.')
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg122_1):
        convolution = torch.ops.aten.convolution.default(arg122_1, arg0_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1);  arg122_1 = arg0_1 = None
        add = torch.ops.aten.add.Tensor(arg64_1, 1);  arg64_1 = None
        var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_1 = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub = torch.ops.aten.sub.Tensor(convolution, getitem_1);  convolution = None
        mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        squeeze = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        squeeze_1 = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_1 = torch.ops.aten.mul.Tensor(squeeze, 0.1);  squeeze = None
        mul_2 = torch.ops.aten.mul.Tensor(arg62_1, 0.9);  arg62_1 = None
        add_2 = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        squeeze_2 = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_3 = torch.ops.aten.mul.Tensor(squeeze_2, 1.0000049824865598);  squeeze_2 = None
        mul_4 = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
        mul_5 = torch.ops.aten.mul.Tensor(arg63_1, 0.9);  arg63_1 = None
        add_3 = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(arg1_1, -1);  arg1_1 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        mul_6 = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        add_4 = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        relu = torch.ops.aten.relu.default(add_4);  add_4 = None
        max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(relu, [3, 3], [2, 2], [1, 1]);  relu = None
        getitem_2 = max_pool2d_with_indices[0]
        getitem_3 = max_pool2d_with_indices[1];  max_pool2d_with_indices = None
        convolution_1 = torch.ops.aten.convolution.default(getitem_2, arg3_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg3_1 = None
        add_5 = torch.ops.aten.add.Tensor(arg67_1, 1);  arg67_1 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
        getitem_4 = var_mean_1[0]
        getitem_5 = var_mean_1[1];  var_mean_1 = None
        add_6 = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_1 = torch.ops.aten.sub.Tensor(convolution_1, getitem_5);  convolution_1 = None
        mul_7 = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        squeeze_3 = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        squeeze_4 = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
        mul_8 = torch.ops.aten.mul.Tensor(squeeze_3, 0.1);  squeeze_3 = None
        mul_9 = torch.ops.aten.mul.Tensor(arg65_1, 0.9);  arg65_1 = None
        add_7 = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        squeeze_5 = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_10 = torch.ops.aten.mul.Tensor(squeeze_5, 1.0000199302441455);  squeeze_5 = None
        mul_11 = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
        mul_12 = torch.ops.aten.mul.Tensor(arg66_1, 0.9);  arg66_1 = None
        add_8 = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        mul_13 = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
        add_9 = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
        relu_1 = torch.ops.aten.relu.default(add_9);  add_9 = None
        convolution_2 = torch.ops.aten.convolution.default(relu_1, arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_1 = arg6_1 = None
        add_10 = torch.ops.aten.add.Tensor(arg70_1, 1);  arg70_1 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
        getitem_6 = var_mean_2[0]
        getitem_7 = var_mean_2[1];  var_mean_2 = None
        add_11 = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_2 = torch.ops.aten.sub.Tensor(convolution_2, getitem_7);  convolution_2 = None
        mul_14 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        squeeze_6 = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        squeeze_7 = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_15 = torch.ops.aten.mul.Tensor(squeeze_6, 0.1);  squeeze_6 = None
        mul_16 = torch.ops.aten.mul.Tensor(arg68_1, 0.9);  arg68_1 = None
        add_12 = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
        squeeze_8 = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
        mul_17 = torch.ops.aten.mul.Tensor(squeeze_8, 1.0000199302441455);  squeeze_8 = None
        mul_18 = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
        mul_19 = torch.ops.aten.mul.Tensor(arg69_1, 0.9);  arg69_1 = None
        add_13 = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(arg8_1, -1);  arg8_1 = None
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        mul_20 = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
        add_14 = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
        add_15 = torch.ops.aten.add.Tensor(add_14, getitem_2);  add_14 = getitem_2 = None
        relu_2 = torch.ops.aten.relu.default(add_15);  add_15 = None
        convolution_3 = torch.ops.aten.convolution.default(relu_2, arg9_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg9_1 = None
        add_16 = torch.ops.aten.add.Tensor(arg73_1, 1);  arg73_1 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
        getitem_8 = var_mean_3[0]
        getitem_9 = var_mean_3[1];  var_mean_3 = None
        add_17 = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        sub_3 = torch.ops.aten.sub.Tensor(convolution_3, getitem_9);  convolution_3 = None
        mul_21 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        squeeze_9 = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
        squeeze_10 = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
        mul_22 = torch.ops.aten.mul.Tensor(squeeze_9, 0.1);  squeeze_9 = None
        mul_23 = torch.ops.aten.mul.Tensor(arg71_1, 0.9);  arg71_1 = None
        add_18 = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
        squeeze_11 = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
        mul_24 = torch.ops.aten.mul.Tensor(squeeze_11, 1.0000199302441455);  squeeze_11 = None
        mul_25 = torch.ops.aten.mul.Tensor(mul_24, 0.1);  mul_24 = None
        mul_26 = torch.ops.aten.mul.Tensor(arg72_1, 0.9);  arg72_1 = None
        add_19 = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(arg11_1, -1);  arg11_1 = None
        unsqueeze_15 = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        mul_27 = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
        add_20 = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None
        relu_3 = torch.ops.aten.relu.default(add_20);  add_20 = None
        convolution_4 = torch.ops.aten.convolution.default(relu_3, arg12_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_3 = arg12_1 = None
        add_21 = torch.ops.aten.add.Tensor(arg76_1, 1);  arg76_1 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
        getitem_10 = var_mean_4[0]
        getitem_11 = var_mean_4[1];  var_mean_4 = None
        add_22 = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        sub_4 = torch.ops.aten.sub.Tensor(convolution_4, getitem_11);  convolution_4 = None
        mul_28 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        squeeze_12 = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
        squeeze_13 = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
        mul_29 = torch.ops.aten.mul.Tensor(squeeze_12, 0.1);  squeeze_12 = None
        mul_30 = torch.ops.aten.mul.Tensor(arg74_1, 0.9);  arg74_1 = None
        add_23 = torch.ops.aten.add.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None
        squeeze_14 = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
        mul_31 = torch.ops.aten.mul.Tensor(squeeze_14, 1.0000199302441455);  squeeze_14 = None
        mul_32 = torch.ops.aten.mul.Tensor(mul_31, 0.1);  mul_31 = None
        mul_33 = torch.ops.aten.mul.Tensor(arg75_1, 0.9);  arg75_1 = None
        add_24 = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
        unsqueeze_16 = torch.ops.aten.unsqueeze.default(arg13_1, -1);  arg13_1 = None
        unsqueeze_17 = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        unsqueeze_18 = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_19 = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        mul_34 = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
        add_25 = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None
        add_26 = torch.ops.aten.add.Tensor(add_25, relu_2);  add_25 = relu_2 = None
        relu_4 = torch.ops.aten.relu.default(add_26);  add_26 = None
        convolution_5 = torch.ops.aten.convolution.default(relu_4, arg15_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_4 = arg15_1 = None
        add_27 = torch.ops.aten.add.Tensor(arg79_1, 1);  arg79_1 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
        getitem_12 = var_mean_5[0]
        getitem_13 = var_mean_5[1];  var_mean_5 = None
        add_28 = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_5 = torch.ops.aten.sub.Tensor(convolution_5, getitem_13);  convolution_5 = None
        mul_35 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        squeeze_15 = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        squeeze_16 = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
        mul_36 = torch.ops.aten.mul.Tensor(squeeze_15, 0.1);  squeeze_15 = None
        mul_37 = torch.ops.aten.mul.Tensor(arg77_1, 0.9);  arg77_1 = None
        add_29 = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
        squeeze_17 = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
        mul_38 = torch.ops.aten.mul.Tensor(squeeze_17, 1.0000797257434426);  squeeze_17 = None
        mul_39 = torch.ops.aten.mul.Tensor(mul_38, 0.1);  mul_38 = None
        mul_40 = torch.ops.aten.mul.Tensor(arg78_1, 0.9);  arg78_1 = None
        add_30 = torch.ops.aten.add.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
        unsqueeze_20 = torch.ops.aten.unsqueeze.default(arg16_1, -1);  arg16_1 = None
        unsqueeze_21 = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        unsqueeze_22 = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
        unsqueeze_23 = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
        add_31 = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None
        relu_5 = torch.ops.aten.relu.default(add_31);  add_31 = None
        convolution_6 = torch.ops.aten.convolution.default(relu_5, arg18_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_5 = arg18_1 = None
        add_32 = torch.ops.aten.add.Tensor(arg82_1, 1);  arg82_1 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(convolution_6, [0, 2, 3], correction = 0, keepdim = True);  convolution_6 = None
        getitem_14 = var_mean_6[0]
        getitem_15 = var_mean_6[1];  var_mean_6 = None
        add_33 = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        return (add_33,)
        
args = []
args.append(rand_strided((64, 3, 7, 7), (147, 49, 7, 1), torch.float32, 'cuda'))  # shape (64, 3, 7, 7), stride (147, 49, 7, 1)
args.append(rand_strided((64,), (1,), torch.float32, 'cuda'))  # shape (64,), stride (1,)
args.append(rand_strided((64,), (1,), torch.float32, 'cuda'))  # shape (64,), stride (1,)
args.append(rand_strided((64, 64, 3, 3), (576, 9, 3, 1), torch.float32, 'cuda'))  # shape (64, 64, 3, 3), stride (576, 9, 3, 1)
args.append(rand_strided((64,), (1,), torch.float32, 'cuda'))  # shape (64,), stride (1,)
args.append(rand_strided((64,), (1,), torch.float32, 'cuda'))  # shape (64,), stride (1,)
args.append(rand_strided((64, 64, 3, 3), (576, 9, 3, 1), torch.float32, 'cuda'))  # shape (64, 64, 3, 3), stride (576, 9, 3, 1)
args.append(rand_strided((64,), (1,), torch.float32, 'cuda'))  # shape (64,), stride (1,)
args.append(rand_strided((64,), (1,), torch.float32, 'cuda'))  # shape (64,), stride (1,)
args.append(rand_strided((64, 64, 3, 3), (576, 9, 3, 1), torch.float32, 'cuda'))  # shape (64, 64, 3, 3), stride (576, 9, 3, 1)
args.append(rand_strided((64,), (1,), torch.float32, 'cuda'))  # shape (64,), stride (1,)
args.append(rand_strided((64,), (1,), torch.float32, 'cuda'))  # shape (64,), stride (1,)
args.append(rand_strided((64, 64, 3, 3), (576, 9, 3, 1), torch.float32, 'cuda'))  # shape (64, 64, 3, 3), stride (576, 9, 3, 1)
args.append(rand_strided((64,), (1,), torch.float32, 'cuda'))  # shape (64,), stride (1,)
args.append(rand_strided((64,), (1,), torch.float32, 'cuda'))  # shape (64,), stride (1,)
args.append(rand_strided((128, 64, 3, 3), (576, 9, 3, 1), torch.float32, 'cuda'))  # shape (128, 64, 3, 3), stride (576, 9, 3, 1)
args.append(rand_strided((128,), (1,), torch.float32, 'cuda'))  # shape (128,), stride (1,)
args.append(rand_strided((128,), (1,), torch.float32, 'cuda'))  # shape (128,), stride (1,)
args.append(rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), torch.float32, 'cuda'))  # shape (128, 128, 3, 3), stride (1152, 9, 3, 1)
args.append(rand_strided((128,), (1,), torch.float32, 'cuda'))  # shape (128,), stride (1,)
args.append(rand_strided((128,), (1,), torch.float32, 'cuda'))  # shape (128,), stride (1,)
args.append(rand_strided((128, 64, 1, 1), (64, 1, 1, 1), torch.float32, 'cuda'))  # shape (128, 64, 1, 1), stride (64, 1, 1, 1)
args.append(rand_strided((128,), (1,), torch.float32, 'cuda'))  # shape (128,), stride (1,)
args.append(rand_strided((128,), (1,), torch.float32, 'cuda'))  # shape (128,), stride (1,)
args.append(rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), torch.float32, 'cuda'))  # shape (128, 128, 3, 3), stride (1152, 9, 3, 1)
args.append(rand_strided((128,), (1,), torch.float32, 'cuda'))  # shape (128,), stride (1,)
args.append(rand_strided((128,), (1,), torch.float32, 'cuda'))  # shape (128,), stride (1,)
args.append(rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), torch.float32, 'cuda'))  # shape (128, 128, 3, 3), stride (1152, 9, 3, 1)
args.append(rand_strided((128,), (1,), torch.float32, 'cuda'))  # shape (128,), stride (1,)
args.append(rand_strided((128,), (1,), torch.float32, 'cuda'))  # shape (128,), stride (1,)
args.append(rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), torch.float32, 'cuda'))  # shape (256, 128, 3, 3), stride (1152, 9, 3, 1)
args.append(rand_strided((256,), (1,), torch.float32, 'cuda'))  # shape (256,), stride (1,)
args.append(rand_strided((256,), (1,), torch.float32, 'cuda'))  # shape (256,), stride (1,)
args.append(rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), torch.float32, 'cuda'))  # shape (256, 256, 3, 3), stride (2304, 9, 3, 1)
args.append(rand_strided((256,), (1,), torch.float32, 'cuda'))  # shape (256,), stride (1,)
args.append(rand_strided((256,), (1,), torch.float32, 'cuda'))  # shape (256,), stride (1,)
args.append(rand_strided((256, 128, 1, 1), (128, 1, 1, 1), torch.float32, 'cuda'))  # shape (256, 128, 1, 1), stride (128, 1, 1, 1)
args.append(rand_strided((256,), (1,), torch.float32, 'cuda'))  # shape (256,), stride (1,)
args.append(rand_strided((256,), (1,), torch.float32, 'cuda'))  # shape (256,), stride (1,)
args.append(rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), torch.float32, 'cuda'))  # shape (256, 256, 3, 3), stride (2304, 9, 3, 1)
args.append(rand_strided((256,), (1,), torch.float32, 'cuda'))  # shape (256,), stride (1,)
args.append(rand_strided((256,), (1,), torch.float32, 'cuda'))  # shape (256,), stride (1,)
args.append(rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), torch.float32, 'cuda'))  # shape (256, 256, 3, 3), stride (2304, 9, 3, 1)
args.append(rand_strided((256,), (1,), torch.float32, 'cuda'))  # shape (256,), stride (1,)
args.append(rand_strided((64,), (1,), torch.float32, 'cuda'))  # shape (64,), stride (1,)
args.append(rand_strided((64,), (1,), torch.float32, 'cuda'))  # shape (64,), stride (1,)
args.append(rand_strided((), (), torch.int64, 'cuda'))  # shape (), stride ()
args.append(rand_strided((64,), (1,), torch.float32, 'cuda'))  # shape (64,), stride (1,)
args.append(rand_strided((64,), (1,), torch.float32, 'cuda'))  # shape (64,), stride (1,)
args.append(rand_strided((), (), torch.int64, 'cuda'))  # shape (), stride ()
args.append(rand_strided((64,), (1,), torch.float32, 'cuda'))  # shape (64,), stride (1,)
args.append(rand_strided((64,), (1,), torch.float32, 'cuda'))  # shape (64,), stride (1,)
args.append(rand_strided((), (), torch.int64, 'cuda'))  # shape (), stride ()
args.append(rand_strided((64,), (1,), torch.float32, 'cuda'))  # shape (64,), stride (1,)
args.append(rand_strided((64,), (1,), torch.float32, 'cuda'))  # shape (64,), stride (1,)
args.append(rand_strided((), (), torch.int64, 'cuda'))  # shape (), stride ()
args.append(rand_strided((64,), (1,), torch.float32, 'cuda'))  # shape (64,), stride (1,)
args.append(rand_strided((64,), (1,), torch.float32, 'cuda'))  # shape (64,), stride (1,)
args.append(rand_strided((), (), torch.int64, 'cuda'))  # shape (), stride ()
args.append(rand_strided((128,), (1,), torch.float32, 'cuda'))  # shape (128,), stride (1,)
args.append(rand_strided((128,), (1,), torch.float32, 'cuda'))  # shape (128,), stride (1,)
args.append(rand_strided((), (), torch.int64, 'cuda'))  # shape (), stride ()
args.append(rand_strided((128,), (1,), torch.float32, 'cuda'))  # shape (128,), stride (1,)
args.append(rand_strided((128,), (1,), torch.float32, 'cuda'))  # shape (128,), stride (1,)
args.append(rand_strided((), (), torch.int64, 'cuda'))  # shape (), stride ()
args.append(rand_strided((128,), (1,), torch.float32, 'cuda'))  # shape (128,), stride (1,)
args.append(rand_strided((128,), (1,), torch.float32, 'cuda'))  # shape (128,), stride (1,)
args.append(rand_strided((), (), torch.int64, 'cuda'))  # shape (), stride ()
args.append(rand_strided((128,), (1,), torch.float32, 'cuda'))  # shape (128,), stride (1,)
args.append(rand_strided((128,), (1,), torch.float32, 'cuda'))  # shape (128,), stride (1,)
args.append(rand_strided((), (), torch.int64, 'cuda'))  # shape (), stride ()
args.append(rand_strided((128,), (1,), torch.float32, 'cuda'))  # shape (128,), stride (1,)
args.append(rand_strided((128,), (1,), torch.float32, 'cuda'))  # shape (128,), stride (1,)
args.append(rand_strided((), (), torch.int64, 'cuda'))  # shape (), stride ()
args.append(rand_strided((256,), (1,), torch.float32, 'cuda'))  # shape (256,), stride (1,)
args.append(rand_strided((256,), (1,), torch.float32, 'cuda'))  # shape (256,), stride (1,)
args.append(rand_strided((), (), torch.int64, 'cuda'))  # shape (), stride ()
args.append(rand_strided((256,), (1,), torch.float32, 'cuda'))  # shape (256,), stride (1,)
args.append(rand_strided((256,), (1,), torch.float32, 'cuda'))  # shape (256,), stride (1,)
args.append(rand_strided((), (), torch.int64, 'cuda'))  # shape (), stride ()
args.append(rand_strided((256,), (1,), torch.float32, 'cuda'))  # shape (256,), stride (1,)
args.append(rand_strided((256,), (1,), torch.float32, 'cuda'))  # shape (256,), stride (1,)
args.append(rand_strided((), (), torch.int64, 'cuda'))  # shape (), stride ()
args.append(rand_strided((256,), (1,), torch.float32, 'cuda'))  # shape (256,), stride (1,)
args.append(rand_strided((256,), (1,), torch.float32, 'cuda'))  # shape (256,), stride (1,)
args.append(rand_strided((), (), torch.int64, 'cuda'))  # shape (), stride ()
args.append(rand_strided((256,), (1,), torch.float32, 'cuda'))  # shape (256,), stride (1,)
args.append(rand_strided((256,), (1,), torch.float32, 'cuda'))  # shape (256,), stride (1,)
args.append(rand_strided((), (), torch.int64, 'cuda'))  # shape (), stride ()
args.append(rand_strided((16, 3, 224, 224), (150528, 50176, 224, 1), torch.float32, 'cuda'))  # shape (16, 3, 224, 224), stride (150528, 50176, 224, 1)
mod = make_fx(Repro(), tracing_mode='real')(*args)

from torch._dynamo.debug_utils import inductor_fails

if inductor_fails(mod, args):
    exit(1)
else:
    exit(0)
