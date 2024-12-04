TODO:
1. Talked to James about logging. Need to think about how we'll handle subproc
   logging. Maybe just log it and refer to the subproc log in the main process.

2. Do we need to pass back the kernel files?

3. PyREClientV2?

TO TEST:

TORCHINDUCTOR_FX_GRAPH_ASYNC_COMPILE=1 python test/inductor/test_codecache.py TestFxGraphCache.test_remote_cache_load_function_device_cpu_bfloat16_dynamic_False

CUDA problems:
  python test/inductor/test_triton_kernels.py -k test_triton_kernel_native_grad_True_dynamic_True_backend_inductor
(py39) $ /data/users/aorenste/miniconda3/envs/py39/lib/python3.9/multiprocessing/resource_tracker.py:216: UserWarning: resource_tracker: There appear to be 5 leaked semaphore objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '

TORCHINDUCTOR_FX_GRAPH_ASYNC_COMPILE=1 pytest -n 10 test/inductor/test_torchinductor.py

Serialized:
Individual test sequence:
TORCHINDUCTOR_FX_GRAPH_ASYNC_COMPILE=1 python test/inductor/test_torchinductor.py CpuTests.test_bool_cpu
TORCHINDUCTOR_FX_GRAPH_ASYNC_COMPILE=1 python test/inductor/test_torchinductor.py GPUTests.test_config_option_dont_assume_alignment_cuda
TORCHINDUCTOR_FX_GRAPH_ASYNC_COMPILE=1 python test/inductor/test_torchinductor.py CpuTests.test_dtype_sympy_expr_cpu

----------------------------------------------------------------------------------------------------------------------------------------------------------------

CI FAILURES:
test/distributed/test_c10d_functional_native.py::CompileTest::test_inductor_all_gather_into_tensor_coalesced
test/distributed/test_c10d_functional_native.py::CompileTest::test_inductor_all_gather_into_tensor_coalesced
test/distributed/test_dynamo_distributed.py::TestMultiProc::test_asymmetric_compilation_with_fx_cache
test/dynamo/test_dynamic_shapes.py::DynamicShapesFunctionTests::test_filter_fallback_dynamic_shapes
test/dynamo/test_dynamic_shapes.py::DynamicShapesFunctionTests::test_filter_fallback_dynamic_shapes
test/functorch/test_eager_transforms.py::TestCompileTransformsCPU::test_compile_vmap_hessian_cpu
test/inductor/test_benchmark_fusion.py::BenchmarkMultiTemplateFusionCudaTest::test_equivalent_template_code
test/inductor/test_cpu_repro.py::CPUReproTests::test_channel_shuffle_cl_output
test/inductor/test_cpu_repro.py::CPUReproTests::test_channel_shuffle_cl_output
test/inductor/test_cpu_repro.py::CPUReproTests::test_channel_shuffle_cl_output
test/inductor/test_cpu_repro.py::CPUReproTests::test_channel_shuffle_cl_output
test/inductor/test_cpu_repro.py::CPUReproTests::test_embedding_vec_bf16
test/inductor/test_cpu_repro.py::CPUReproTests::test_expr_vec_non_contiguous
test/inductor/test_cpu_repro.py::CPUReproTests::test_group_norm_vec
test/inductor/test_cpu_repro.py::CPUReproTests::test_ir_node_str
test/inductor/test_cpu_repro.py::CPUReproTests::test_local_buffer_in_outer_loop_fusion
test/inductor/test_cpu_repro.py::CPUReproTests::test_local_buffer_in_outer_loop_fusion
test/inductor/test_cpu_repro.py::CPUReproTests::test_local_buffer_in_outer_loop_fusion
test/inductor/test_cpu_repro.py::CPUReproTests::test_lstm_packed_unbatched_False_input_size_1_hidden_size_2_num_layers_1_bidirectional_False_bias_False_empty_state_False_batch_first_False_batch_size_2_seq_len_1
test/inductor/test_cpu_repro.py::CPUReproTests::test_lstm_packed_unbatched_False_input_size_1_hidden_size_2_num_layers_1_bidirectional_False_bias_False_empty_state_False_batch_first_True_batch_size_2_seq_len_1
test/inductor/test_max_autotune.py::TestMaxAutotune::test_cat_max_autotune_triton
test/inductor/test_torchinductor_codegen_dynamic_shapes.py::DynamicShapesCodegenCpuTests::test__unsafe_masked_index_put_accumulate_dynamic_shapes_cpu
test/inductor/test_torchinductor_codegen_dynamic_shapes.py::DynamicShapesCodegenCpuTests::test__unsafe_masked_index_put_accumulate_dynamic_shapes_cpu
test/inductor/test_torchinductor_codegen_dynamic_shapes.py::DynamicShapesCodegenCpuTests::test__unsafe_masked_index_put_accumulate_dynamic_shapes_cpu
test/inductor/test_torchinductor_codegen_dynamic_shapes.py::DynamicShapesCodegenCpuTests::test__unsafe_masked_index_put_accumulate_dynamic_shapes_cpu
test/inductor/test_torchinductor_codegen_dynamic_shapes.py::DynamicShapesCodegenCpuTests::test_adaptive_avg_pool1d_argmax_dynamic_shapes_cpu
test/inductor/test_torchinductor_codegen_dynamic_shapes.py::DynamicShapesCodegenCpuTests::test_add_const_int_dynamic_shapes_cpu
test/inductor/test_torchinductor_codegen_dynamic_shapes.py::DynamicShapesCodegenCpuTests::test_add_const_int_dynamic_shapes_cpu
test/inductor/test_torchinductor_codegen_dynamic_shapes.py::DynamicShapesCodegenCpuTests::test_add_const_int_dynamic_shapes_cpu
test/inductor/test_torchinductor_codegen_dynamic_shapes.py::DynamicShapesCodegenCpuTests::test_add_const_int_dynamic_shapes_cpu
test/inductor/test_torchinductor_codegen_dynamic_shapes.py::DynamicShapesCodegenCpuTests::test_add_const_int_dynamic_shapes_cpu
test/inductor/test_torchinductor_codegen_dynamic_shapes.py::DynamicShapesCodegenCpuTests::test_angle_dynamic_shapes_cpu

lennard_jones
mixer_b16_224
speech_transformer
vision_maskrcnn
vision_maskrcnn
vit_base_patch16_224
##[error]Process completed with exit code 1.
Process completed with exit code 255.
detectron2_fcos_r_50_fpn
detectron2_maskrcnn_r_50_fpn

REVIEWS:
split
top-level interface on inductor - elias or jansel

dig into torch.compile code: multiple graphs - hang off the code object
registered w/ optimized things need to run dynamo - might graph break. I need to
actually run backend eager. AOT autograd. Need to set up a new code object htat
has the bytecode and fx graph. Need to have set up the full aot autograd fw and
bkwrd. Call into inductor async object. Sometimes FxGraphModule - need to flip
to compiled version. Calling into the dynamo bytecode & aot autograd
function. At the point in ttime when I can switch into compiled need to
test. This is the runtime object - two impls - eager + pending future. Test it
and eventually future is filled in.

NOT fx_codegen_and_compile() - but internal to that. THe code object it returns
has to do the check.

START GETTING REVIEWS

ISSUES:

1. What about not running autotuning or triton as part of it? Should I be
working toward parallel compile + add it in or reverse?

2. Running eager in BG - does AOT eager need inductor or just run off the fx
graph? If fx graph should we be running that so we can run ahead of the compile
and kick off more compiles?

compiled FX graph output product - complicated triton bundling, etc.
AOTI is a single file
pt2 zip file
load into AOTI runner

1. AOTI vs direct path - big diff = cpp wrapper
   a. less good backtraces
   b. cpp codegen is it's own thing + less features
   c. more risk
   d. dealing with the bundling + triton cache hit would be better

Prob. move ahead w/ AOTI
Use the AOTI compiled products
Useful to have end-to-end.
= async + parallel

12/4/24 Meeting w/ Ed

Use AOTI - if loading the triton cache is a blocker
