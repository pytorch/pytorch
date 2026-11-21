#!/usr/bin/env python3
"""
Auto-classify skip reasons for ROCm parity CSV tests.

Takes a parity CSV (output of summarize_xml_testreports.py) and automatically
assigns skip_reason categories to tests where ROCm=SKIPPED/MISSED and CUDA=PASSED
based on patterns in:
  - The skip message (message_rocm column)
  - The test file name
  - The test class name
  - The test name

Rules are ordered by specificity: combined match rules first, then message-based,
then file+class combos, then file-only fallbacks. First matching rule wins.

Usage:
  python auto_classify_skip_reasons.py -i input.csv -o output.csv [--report]
  python auto_classify_skip_reasons.py -i input.csv -o output.csv --tsv-out updated_skip_reasons.tsv
  python auto_classify_skip_reasons.py -i input.csv --dry-run --report
"""

import argparse
import ast
import csv
import re
import sys
from collections import Counter, defaultdict


# ---------------------------------------------------------------------------
# Rules are evaluated top-to-bottom; first match wins.
# Each rule is a dict with:
#   reason:   the skip_reason category string
#   msg:      (optional) regex to match against the skip message
#   file:     (optional) regex to match against test_file
#   cls:      (optional) regex to match against test_class
#   name:     (optional) regex to match against test_name
#   workflow: (optional) one of "default", "distributed", "inductor", "slow"
#
# All provided fields must match (AND logic). Omitted fields match anything.
# msg="" matches empty messages; omitting msg matches anything.
# ---------------------------------------------------------------------------

RULES = [
    # ==================================================================
    # TIER 1: High-specificity combined rules (message + file/class)
    # ==================================================================

    # --- PT2.0 - Convolution: conv2d backward parametrized skipped on ROCm ---
    # skipIfRocmArch(...) skips test_conv2d_backward_parametrized (see upstream
    # #188671, CI timeout). Categorize it as a convolution issue rather than the
    # generic Misc "test skipped on ('gfx...')" bucket below. Must precede that
    # Misc arch rule.
    {"reason": "PT2.0 - Convolution",
     "msg": r"test skipped on \('gfx",
     "name": r"(?i)conv2d_backward"},

    # --- bfloat16_SDPA_ME: dropout mask in test_transformers with bfloat16 in TEST NAME ---
    # Must be before generic SDPA_ME rule
    {"reason": "bfloat16_SDPA_ME",
     "msg": r"_fill_mem_eff_dropout_mask",
     "file": r"^test_transformers$",
     "name": r"(?i)bfloat16|bf16"},

    # --- GEMMS: test_mm_bmm in test_matmul_cuda with accuracy regression ---
    # Must be before generic hipblas rule
    {"reason": "GEMMS",
     "msg": r"accuracy regression in hipblas",
     "file": r"^test_matmul_cuda$",
     "name": r"test_mm_bmm"},

    # --- hipblas hipblaslt: test_addmm/test_cublas/other in test_matmul_cuda ---
    {"reason": "hipblas hipblaslt",
     "msg": r"accuracy regression in hipblas",
     "file": r"^test_matmul_cuda$"},
    {"reason": "hipblas hipblaslt",
     "msg": r"skipIfRocm.*doesn't currently work",
     "file": r"^test_matmul_cuda$"},
    {"reason": "hipblas hipblaslt",
     "file": r"^test_matmul_cuda$",
     "msg": r"Green contexts are not supported"},

    # --- Expected to work: skipCUDAIfRocm in test_meta for ldl_solve ops ---
    {"reason": "Expected to work",
     "msg": r"skipCUDAIfRocm.*doesn't currently work",
     "file": r"^test_meta$",
     "name": r"(?i)ldl_solve"},

    # --- Linalg: skipCUDAIfRocm in test_meta for other linalg ops ---
    {"reason": "Linalg",
     "msg": r"skipCUDAIfRocm.*doesn't currently work",
     "file": r"^test_meta$"},

    # --- Linalg: skipCUDAIfRocm in test_ops/test_linalg/test_meta/test_ops_fwd_gradients/test_ops_gradients ---
    # These are ops like linalg.svd, linalg.eigh, etc.
    {"reason": "Linalg",
     "msg": r"skipCUDAIfRocm.*doesn't currently work",
     "file": r"^test_linalg$"},
    {"reason": "Linalg",
     "msg": r"_convert_weight_to_int4pack_cuda.*(supported only for|is supported only for) CDNA"},
    {"reason": "Linalg",
     "msg": r"bfloat16 NCHW train failed"},
    {"reason": "Linalg",
     "msg": r"skipCUDAIfRocm.*doesn't currently work",
     "file": r"^test_ops$",
     "name": r"(?i)linalg|svd|eig[hs]?|cholesky|lstsq|solve|inv|det|qr|lu|pinv|matrix_rank|cross|norm|cond|householder|ormqr|geqrf|triangular|vecdot|multi_dot"},
    {"reason": "Linalg",
     "msg": r"skipCUDAIfRocm.*doesn't currently work",
     "file": r"^test_ops_fwd_gradients$"},
    {"reason": "Linalg",
     "msg": r"skipCUDAIfRocm.*doesn't currently work",
     "file": r"^test_ops_gradients$",
     "name": r"(?i)linalg|svd|eig[hs]?|cholesky|lstsq|solve|inv|det|qr|lu|pinv|householder|ormqr|geqrf|triangular"},
    {"reason": "Linalg",
     "msg": r"skipCUDAIfRocm.*doesn't currently work",
     "file": r"^test_meta$",
     "name": r"(?i)linalg|svd|eig[hs]?|cholesky|lstsq|solve|inv|det|qr|lu|pinv|householder|ormqr|geqrf|triangular"},
    {"reason": "Linalg",
     "file": r"^test_nn$",
     "msg": r"skipIfRocm.*doesn't currently work"},

    # --- Linalg: hipSOLVER xgeev requires ROCm >= 7.14 (linalg.eig via xgeev, pytorch#188720) ---
    {"reason": "Linalg",
     "msg": r"hipSOLVER xgeev"},
    # --- Linalg: ROCm 6.4 regression on linalg ops (householder/decomp) ---
    {"reason": "Linalg",
     "msg": r"regression in ROCm 6\.4"},

    # --- Profiler: CUPTI-dependent profiler tests (libcupti / cupti-python) ---
    {"reason": "Profiler",
     "msg": r"(?i)cupti"},

    # --- block_table: ROCm does not support paged-KV block_table (varlen attention) ---
    {"reason": "block_table",
     "msg": r"ROCm does not support block_table"},

    # --- explicit NVIDIA test: CUDA / SM-gated tests skipped on ROCm ---
    {"reason": "explicit NVIDIA test",
     "msg": r"only supported on NVIDIA CUDA"},
    {"reason": "explicit NVIDIA test",
     "msg": r"^CUDA-only$"},
    {"reason": "explicit NVIDIA test",
     "msg": r"requires CUDA SM80"},
    {"reason": "explicit NVIDIA test",
     "msg": r"CUTLASS.*CUDA-only"},

    # --- hipSolver/Magma: skipCUDAIfRocm in test_ops for ldl_solve, scaled_dot_product, conv_transpose3d ---
    {"reason": "hipSolver/Magma",
     "msg": r"skipCUDAIfRocm.*doesn't currently work",
     "file": r"^test_ops$",
     "name": r"(?i)ldl_solve|scaled_dot_product|conv_transpose3d"},
    {"reason": "hipSolver/Magma",
     "msg": r"skipCUDAIfRocm.*doesn't currently work",
     "file": r"^test_ops_jit$"},
    {"reason": "hipSolver/Magma",
     "msg": r"skipCUDAIfRocm.*doesn't currently work",
     "file": r"^test_decomp$"},
    {"reason": "hipSolver/Magma",
     "msg": r"skipCUDAIfRocm.*doesn't currently work",
     "file": r"^test_schema_check$"},
    {"reason": "hipSolver/Magma",
     "msg": r"skipCUDAIfRocm.*doesn't currently work",
     "file": r"^test_testing$"},
    {"reason": "hipSolver/Magma",
     "msg": r"Skipped for ROCm!"},
    {"reason": "hipSolver/Magma",
     "msg": r"test_cow_input does not work with efficient attention on ROCM"},
    {"reason": "hipSolver/Magma",
     "msg": r"hipSOLVER DnXsytrs requires ROCm >= 7\.14"},

    # --- Compiler issue: "Skipped!" in test_ops for specific compiler-related tests ---
    {"reason": "Compiler issue",
     "msg": r"^Skipped!$",
     "file": r"^test_ops$",
     "name": r"(?i)special_hermite_polynomial_h|special_laguerre"},

    # --- non-standard bool: "Skipped!" in test_ops for bool-related tests ---
    {"reason": "non-standard bool",
     "msg": r"^Skipped!$",
     "file": r"^test_ops$",
     "name": r"(?i)bool"},

    # --- pow: "Skipped!" in test_ops/test_decomp for pow tests ---
    {"reason": "pow",
     "msg": r"^Skipped!$",
     "file": r"^test_ops$|^test_decomp$",
     "name": r"(?i)^pow$|_pow_|float_power"},

    # --- fft: "Skipped!" or "Skipped on ROCm" in test_ops for fft tests ---
    {"reason": "fft",
     "msg": r"^Skipped(!| on ROCm)$",
     "file": r"^test_ops$",
     "name": r"(?i)fft"},

    # --- NHWC: "Skipped!" in test_modules for NHWC tests ---
    {"reason": "NHWC",
     "msg": r"^Skipped!$",
     "file": r"^test_modules$"},

    # (FakeTensor removed — "Requires CUDA" messages are explicit NVIDIA test per policy)

    # --- hermite_polynomial_h: custom_mask_type in test_ops for hermite ---
    {"reason": "hermite_polynomial_h",
     "msg": r"Efficient attention on ROCM doesn't support custom_mask_type",
     "file": r"^test_ops$",
     "name": r"(?i)hermite"},

    # --- fake_crossref: skipCUDAIfRocm in test_ops for crossref tests ---
    {"reason": "fake_crossref",
     "msg": r"skipCUDAIfRocm.*doesn't currently work",
     "file": r"^test_ops$",
     "name": r"(?i)crossref|fake_crossref"},

    # --- Jit: Tensor-likes not close in test_jit_fuser ---
    {"reason": "Jit",
     "msg": r"Tensor-likes are not close",
     "file": r"test_jit_fuser"},

    # --- Memory allocation: TestBlockStateAbsorption in test_cuda ---
    {"reason": "Memory allocation",
     "file": r"^test_cuda$",
     "cls": r"^TestBlockStateAbsorption$"},

    # --- cuda allocator: TestCudaAllocator in test_cuda ---
    {"reason": "cuda allocator",
     "file": r"^test_cuda$",
     "cls": r"^TestCudaAllocator$"},

    # --- hipGraph/cudaGraph: CudaGraph-related classes in test_cuda ---
    {"reason": "hipGraph/cudaGraph",
     "file": r"^test_cuda$",
     "cls": r"CachingHostAllocatorCudaGraph|GreenContext"},

    # --- Memory allocation: TestMemPool in test_cuda ---
    {"reason": "Memory allocation",
     "file": r"^test_cuda$",
     "cls": r"^TestMemPool$"},

    # --- Profiler: TestFXMemoryProfiler in test_cuda ---
    {"reason": "Profiler",
     "file": r"^test_cuda$",
     "cls": r"FXMemoryProfiler"},

    # --- compiled optimizer: ROCm numerical behavior in inductor.test_compiled_optimizers ---
    {"reason": "compiled optimizer",
     "msg": r"ROCm may have different numerical behavior",
     "file": r"inductor\.test_compiled_optimizers"},

    # --- functorch: FuncTorch classes in inductor.test_compiled_autograd ---
    {"reason": "functorch",
     "file": r"^inductor\.test_compiled_autograd$",
     "cls": r"FuncTorch"},

    # --- PT2.0 - Distributed: DTensor classes in inductor.test_compiled_autograd ---
    {"reason": "PT2.0 - Distributed",
     "file": r"^inductor\.test_compiled_autograd$",
     "cls": r"DTensor"},

    # --- hipdnn: cudnn Attention messages ---
    {"reason": "hipdnn",
     "msg": r"[Cc]u[Dd][Nn][Nn] Attention is not supported"},
    {"reason": "hipdnn",
     "msg": r"Efficient or cuDNN Attention was not built"},

    # --- Will not be supported on ROCm: test_transformers with (no message) ---
    {"reason": "Will not be supported on ROCm",
     "file": r"^test_transformers$",
     "cls": r"SDPA.*CUDA",
     "msg": r"^$"},

    # --- transformers: test_transformers / test_flop_counter with misc messages ---
    {"reason": "transformers",
     "file": r"^test_transformers$",
     "msg": r"Does not support all SDPA backends"},
    {"reason": "transformers",
     "file": r"^test_flop_counter$"},

    # --- bfloat16: test_sparse_csr with (no message) ---
    {"reason": "bfloat16",
     "file": r"^test_sparse_csr$",
     "cls": r"[Bb]float16|bf16"},
    {"reason": "bfloat16",
     "file": r"^test_sparse$",
     "cls": r"[Bb]float16|bf16"},
    {"reason": "bfloat16",
     "file": r"^test_matmul_cuda$",
     "msg": r"ROCm doesn't support CUTLASS"},

    # --- explicit NVIDIA test: test_sparse_semi_structured with cutlass in NAME ---
    {"reason": "explicit NVIDIA test",
     "file": r"^test_sparse_semi_structured$",
     "name": r"(?i)cutlass"},

    # --- cusparselt: everything else in test_sparse_semi_structured ---
    {"reason": "cusparselt",
     "file": r"^test_sparse_semi_structured$"},

    # --- Quantization: distributed quantization tests ---
    {"reason": "Quantization",
     "msg": r"Test skipped for ROCm",
     "file": r"distributed\.algorithms\.quantization"},

    # --- Process Group: distributed spawn/c10d with "Test skipped for ROCm" ---
    {"reason": "Process Group",
     "msg": r"Test skipped for ROCm",
     "file": r"distributed\.test_distributed_spawn.*nccl"},

    # ==================================================================
    # TIER 2: Message-based rules (strong signal from skip message)
    # ==================================================================

    # SDPA_ME
    {"reason": "SDPA_ME",
     "msg": r"_fill_mem_eff_dropout_mask"},
    {"reason": "SDPA_ME",
     "msg": r"Efficient attention on ROCM doesn't support custom_mask_type"},
    {"reason": "SDPA_ME",
     "msg": r"Efficient Attention on ROCM does not support head_dim"},

    # SDPA_FA
    {"reason": "SDPA_FA",
     "msg": r"Large numerical errors on ROCM"},
    {"reason": "SDPA_FA",
     "msg": r"flash attention not supported"},

    # Will not be supported on ROCm
    {"reason": "Will not be supported on ROCm",
     "msg": r"head_dim != head_dim_v unsupported on ROCm"},

    # Triton 3.7 bump
    {"reason": "triton 3.7 bump",
     "msg": r"skipIfRocm.*Fails with Triton 3\.7"},

    # MIOpen
    {"reason": "MIOpen Convolutions",
     "msg": r"Marked as skipped for MIOpen"},

    # Static CUDA launcher
    {"reason": "static cuda launcher",
     "msg": r"Static cuda launcher doesn't work with ROCM"},

    # NUMBA
    {"reason": "NUMBA",
     "msg": r"No numba\.cuda"},

    # int4
    {"reason": "int4",
     "msg": r"_int4_mm is supported only for CDNA"},

    # FP8
    {"reason": "FP8",
     "msg": r"cuBLAS blockwise scaling"},

    # variable length attention
    {"reason": "variable length attention",
     "msg": r"ROCm does not support seqused_k"},

    # CUDA IPC
    {"reason": "Pass with unskip or minor mod",
     "msg": r"CUDA IPC not available"},

    # Python version
    {"reason": "Python version",
     "msg": r"Not supported in Python 3\.1[0-9]+"},

    # cpp_test / CUDA not found
    {"reason": "cpp_test",
     "msg": r"CUDA not found"},
    {"reason": "cpp_test",
     "msg": r"CUDA_HOME not set"},

    # Foreach
    {"reason": "Foreach",
     "msg": r"failed starting on ROCm"},

    # CUTLASS
    {"reason": "cutlass",
     "msg": r"ROCm doesn't support CUTLASS|CUTLASS backend is not supported on HIP|ROCm and Windows doesn't support CUTLASS"},

    # Transformers dependency
    {"reason": "transformers",
     "msg": r"No transformers"},

    # hipGraph / cudaGraph (but NOT in functorch files -- those stay functorch)
    {"reason": "hipGraph/cudaGraph",
     "msg": r"Green contexts are not supported"},
    {"reason": "functorch",
     "msg": r"CUDA 12\.4 or greater is required for CUDA Graphs",
     "file": r"^functorch\."},
    {"reason": "hipGraph/cudaGraph",
     "msg": r"CUDA 12\.4 or greater is required for CUDA Graphs"},
    {"reason": "hipGraph/cudaGraph",
     "msg": r"ROCM >= 5\.3 required for graphs.*cuda-bindings"},

    # TMA / Blackwell
    {"reason": "Will not be supported on ROCm",
     "msg": r"Need.*TMA support"},
    {"reason": "Will not be supported on ROCm",
     "msg": r"Need Blackwell"},

    # CUDA SM requirements
    {"reason": "explicit NVIDIA test",
     "msg": r"Requires CUDA SM >= [0-9]"},
    {"reason": "explicit NVIDIA test",
     "msg": r"Requires CUDA with SM >= [0-9]"},
    {"reason": "explicit NVIDIA test",
     "msg": r"Test is only supported on CUDA 1[0-9]"},
    {"reason": "explicit NVIDIA test",
     "msg": r"Requires NCCL version greater than"},
    {"reason": "explicit NVIDIA test",
     "msg": r"Excluded from CUDA tests"},

    # FP8 — MI300+ / H100+ only
    {"reason": "FP8",
     "msg": r"FP8 is only supported on H100\+|FP8 is not supported on this platform|FP8 requires H100\+"},
    {"reason": "FP8",
     "msg": r"requires gpu with fp8 support"},

    # Symmetric memory
    {"reason": "Symmetric memory",
     "msg": r"SymmMem is not supported on this ROCm arch"},

    # Python version / 3.12+
    {"reason": "Python version",
     "msg": r"Failing on python 3\.12\+|torch\.compile is not supported on python 3\.12\+|complex flaky in 3\.12"},

    # Greater than 4 GPU (distributed)
    {"reason": "Greater than 4 GPU",
     "msg": r"Need at least 4 (?:CUDA|accelerator) devices"},
    {"reason": "Greater than 4 GPU",
     "msg": r"Test requires.*world size of 4"},
    {"reason": "Greater than 4 GPU",
     "msg": r"requires [34] GPUs, found [12]"},

    # tensor_parallel — architecture-specific skip
    {"reason": "tensor_parallel",
     "msg": r"test only runs on \('gfx942'"},

    # Process Group: subprocess level skip
    {"reason": "Process Group",
     "msg": r"Test skipped at subprocess level"},

    # Sharded Tensor: subprocess level skip in _shard
    {"reason": "Sharded Tensor",
     "msg": r"Test skipped at subprocess level",
     "file": r"distributed\._shard"},

    # Process Group: NCCL version / device assert
    {"reason": "Process Group",
     "msg": r"NCCL test requires 2\+ GPUs"},

    # Misc: ROCm preserves subnormals
    {"reason": "Misc",
     "msg": r"ROCm preserves subnormals"},

    # Misc: GCC codegen
    {"reason": "Misc",
     "msg": r"Fails under GCC 1[0-9] due to vector codegen"},

    # Misc: Skipped on ROCm due to hang
    {"reason": "Misc",
     "msg": r"Skipped on ROCm due to hang"},

    # Misc: Test skipped for ROCm (generic distributed)
    {"reason": "Misc",
     "msg": r"Test skipped for ROCm"},

    # Misc: architecture-specific skips
    {"reason": "Misc",
     "msg": r"test skipped on \('gfx"},

    # Misc: generic skipIfRocm pointing at a tracked GitHub issue (no specific
    # category; heterogeneous inductor/varlen tests)
    {"reason": "Misc",
     "msg": r"skipIfRocm: https?://"},

    # cuFFT-specific
    {"reason": "Misc",
     "msg": r"cuFFT-specific"},

    # ROCTracer profiler
    {"reason": "Memory allocation",
     "msg": r"ROCTracer does not capture"},

    # expandable_segments-related messages
    {"reason": "expandable_segments",
     "msg": r"expandable_segments mode is not supported on ROCm"},
    {"reason": "expandable_segments",
     "msg": r"CUDA >= 11\.0 required for external events in cuda graphs.*rocm"},

    # not enabled by default on rocm
    {"reason": "expandable_segments",
     "msg": r"not enabled by default on rocm"},

    # HIP runtime context
    {"reason": "Misc",
     "msg": r"HIP runtime doesn't create context"},

    # ==================================================================
    # TIER 3: File + class based rules (for empty/generic messages)
    # ==================================================================

    # --- test_cuda class-based disambiguation ---
    {"reason": "Misc",
     "file": r"^test_cuda$",
     "cls": r"^TestCuda$"},
    {"reason": "compiled optimizer",
     "file": r"^test_cuda$",
     "cls": r"TestCudaOptims"},
    {"reason": "Misc",
     "file": r"^test_cuda$",
     "cls": r"TestCudaAutocast"},
    {"reason": "cpp_test",
     "file": r"^test_cuda$",
     "cls": r"TestCompileKernel"},

    # --- test_nn (MI200-specific skips, no message) ---
    {"reason": "Misc",
     "file": r"^test_nn$"},

    # --- inductor.test_fp8 ---
    {"reason": "FP8",
     "file": r"^inductor\.test_fp8$"},

    # --- test_scaled_matmul_cuda ---
    {"reason": "FP8",
     "file": r"^test_scaled_matmul_cuda$"},

    # --- inductor.test_torchinductor_strided_blocks ---
    {"reason": "PT2.0 - Inductor",
     "file": r"^inductor\.test_torchinductor_strided_blocks$"},

    # --- inductor.test_strict_numerics ---
    {"reason": "PT2.0 - Inductor",
     "file": r"^inductor[./]test_strict_numerics$"},

    # --- inductor.test_flex_decoding ---
    {"reason": "flex_decoding",
     "file": r"^inductor\.test_flex_decoding$"},

    # --- inductor.test_loop_ordering ---
    {"reason": "PT2.0 - Inductor",
     "file": r"^inductor\.test_loop_ordering$"},

    # --- inductor.test_triton_kernels (Triton HOP / custom-dialect mutation
    # analysis tests). AI-analysis flagged these as churning an unrelated
    # skip-category when new upstream tests land; they belong to Inductor. ---
    {"reason": "PT2.0 - Inductor",
     "file": r"^inductor[./]test_triton_kernels$"},

    # --- inductor.test_triton_heuristics (skipUnless(HAS_GPU_AND_TRITON)
    # guards; GPU/Triton-required inductor tests) ---
    {"reason": "PT2.0 - Inductor",
     "file": r"^inductor[./]test_triton_heuristics$"},
    {"reason": "PT2.0 - Inductor",
     "file": r"^inductor[./]test_fused_attention$"},

    # --- dynamo.test_backends (TVM and other dynamo backend tests; e.g. TVM
    # backend tests enabled/skipped by apache-tvm availability) ---
    {"reason": "PT2.0 - Dynamo",
     "file": r"^dynamo[./]test_backends$"},

    # --- torch_np / numpy tests ---
    {"reason": "NumPy",
     "file": r"^torch_np\."},

    # --- test_binary_ufuncs ---
    {"reason": "Misc",
     "file": r"^test_binary_ufuncs$"},

    # --- test_fx ---
    {"reason": "FX",
     "file": r"^test_fx$"},

    # --- profiler.test_execution_trace ---
    {"reason": "Profiler",
     "file": r"^profiler\.test_execution_trace$"},

    # --- test_cpp_api_parity ---
    {"reason": "cpp_test",
     "file": r"^test_cpp_api_parity$"},

    # --- test_expanded_weights ---
    {"reason": "Misc",
     "file": r"^test_expanded_weights$"},

    # --- test_linalg (arch-specific skips) ---
    {"reason": "Linalg",
     "file": r"^test_linalg$"},

    # --- test_torch (arch-specific skips) ---
    {"reason": "Misc",
     "file": r"^test_torch$"},

    # --- nn.test_convolution (arch-specific) ---
    {"reason": "MIOpen Convolutions",
     "file": r"^nn\.test_convolution$"},

    # --- inductor.test_aot_inductor_arrayref ---
    {"reason": "PT2.0 - AOTInductor",
     "file": r"^inductor\.test_aot_inductor_arrayref$"},

    # --- distributed.test_symmetric_memory ---
    {"reason": "Symmetric memory",
     "file": r"^distributed\.test_symmetric_memory$"},

    # --- inductor.test_compiled_autograd HigherOrderOp (MI300 has more classes) ---
    {"reason": "functorch",
     "file": r"^inductor\.test_compiled_autograd$",
     "cls": r"HigherOrderOp"},

    # --- explicit NVIDIA test in various files ---
    {"reason": "explicit NVIDIA test",
     "file": r"^test_cuda_nvml_based_avail$"},
    {"reason": "explicit NVIDIA test",
     "file": r"^test_cpp_extensions_aot"},

    # --- hipGraph/cudaGraph: only test_graph_* (NOT test_cuda_graph_*) in test_cuda_expandable_segments ---
    {"reason": "hipGraph/cudaGraph",
     "file": r"^test_cuda_expandable_segments$",
     "name": r"^test_graph_"},

    # --- expandable_segments (everything else in test_cuda_expandable_segments) ---
    {"reason": "expandable_segments",
     "file": r"^test_cuda_expandable_segments$"},

    # --- Profiler ---
    {"reason": "Profiler",
     "file": r"^profiler\.test_profiler$"},

    # --- serialization ---
    {"reason": "serialization",
     "file": r"^test_serialization$"},

    # --- dataloader ---
    {"reason": "dataloader",
     "file": r"^test_dataloader$"},

    # --- Multi-Processing ---
    {"reason": "Multi-Processing",
     "file": r"^test_multiprocessing_spawn$"},
    {"reason": "Multi-Processing",
     "file": r"^test_multiprocessing$"},

    # --- hipSparse ---
    {"reason": "hipSparse",
     "file": r"^test_sparse_csr$"},
    {"reason": "hipSparse",
     "file": r"^test_sparse$",
     "msg": r"^$"},

    # --- nested tensor ---
    {"reason": "nested tensor",
     "file": r"^test_nestedtensor$"},

    # --- asm_elementwise ---
    {"reason": "asm_elementwise",
     "file": r"higher_order_ops\.test_inline_asm_elementwise"},

    # --- torchinductor_opinfo_properties ---
    {"reason": "torchinductor_opinfo_properties",
     "file": r"^inductor\.test_torchinductor_opinfo_properties$"},

    # --- flex_attention ---
    {"reason": "flex_attention",
     "file": r"^inductor\.test_flex_attention$"},

    # --- compiled optimizer ---
    {"reason": "compiled optimizer",
     "file": r"^inductor\.test_compiled_optimizers$"},

    # --- inductor combo_kernels ---
    {"reason": "PT2.0 - Inductor",
     "file": r"^inductor\.test_combo_kernels$"},

    # --- inductor compiled_autograd (remaining after FuncTorch/DTensor class rules) ---
    {"reason": "PT2.0 - Inductor",
     "file": r"^inductor\.test_compiled_autograd$"},

    # --- Foreach (inductor) ---
    {"reason": "Foreach",
     "file": r"^inductor\.test_foreach$"},

    # --- inductor codecache / cudacodecache ---
    {"reason": "PT2.0 - Inductor",
     "file": r"^inductor\.test_codecache$"},
    {"reason": "PT2.0 - Inductor",
     "file": r"^inductor\.test_cudacodecache$"},

    # --- inductor GPU cpp wrapper ---
    {"reason": "PT2.0 - Inductor",
     "file": r"^inductor\.test_gpu_cpp_wrapper$"},

    # --- inductor torchinductor variants ---
    {"reason": "PT2.0 - Inductor",
     "file": r"^inductor\.test_torchinductor$"},
    {"reason": "PT2.0 - Inductor",
     "file": r"^inductor\.test_torchinductor_dynamic_shapes$"},
    {"reason": "PT2.0 - Inductor",
     "file": r"^inductor\.test_torchinductor_codegen_dynamic_shapes$"},
    {"reason": "PT2.0 - Inductor",
     "file": r"^inductor\.test_torchinductor_opinfo$"},

    # --- inductor compile subprocess ---
    {"reason": "PT2.0 - Inductor",
     "file": r"^inductor\.test_compile_subprocess$"},
    {"reason": "PT2.0 - Inductor",
     "file": r"^inductor\.test_compile_worker$"},

    # --- inductor cpu/cuda repro ---
    {"reason": "PT2.0 - Inductor",
     "file": r"^inductor\.test_cpu_repro$"},
    {"reason": "PT2.0 - Inductor",
     "file": r"^inductor\.test_cuda_repro$"},

    # --- inductor custom lowering / minifier ---
    {"reason": "PT2.0 - Inductor",
     "file": r"^inductor\.test_custom_lowering$"},
    {"reason": "PT2.0 - Inductor",
     "file": r"^inductor\.test_minifier"},
    {"reason": "PT2.0 - Inductor",
     "file": r"^inductor\.test_mix_order"},

    # --- inductor aot_inductor ---
    {"reason": "PT2.0 - AOTInductor",
     "file": r"^inductor\.test_aot_inductor"},

    # --- functorch ---
    {"reason": "functorch",
     "file": r"^functorch\."},

    # --- dynamo ---
    {"reason": "PT2.0 - Dynamo",
     "file": r"^dynamo\."},

    # --- export ---
    {"reason": "PT2.0 - Inductor",
     "file": r"^export\."},

    # --- tf32: test_nn with "Test is disabled" ---
    {"reason": "tf32",
     "file": r"^test_nn$",
     "msg": r"Test is disabled"},

    # --- MIOpen Convolutions ---
    {"reason": "MIOpen Convolutions",
     "file": r"^nn\.test_convolution$"},

    # --- test_stateless ---
    {"reason": "Misc",
     "file": r"^test_stateless$"},

    # --- test_cuda_primary_ctx ---
    {"reason": "Misc",
     "file": r"^test_cuda_primary_ctx$"},

    # --- test_torchfuzz ---
    {"reason": "Misc",
     "file": r"^test_torchfuzz"},

    # ==================================================================
    # TIER 4: Distributed file-based rules
    # ==================================================================

    # Sharded Tensor
    {"reason": "Sharded Tensor",
     "file": r"^distributed\._shard\."},
    {"reason": "Sharded Tensor",
     "file": r"^distributed\._composable\.fsdp\.test_fully_shard_training$"},
    {"reason": "Sharded Tensor",
     "file": r"^distributed\._composable\.fsdp\.test_fully_shard_clip_grad"},

    # tensor_parallel
    {"reason": "tensor_parallel",
     "file": r"^distributed\.tensor\.parallel\."},

    # pipeline_parallel
    {"reason": "pipeline_parallel",
     "file": r"^distributed\.pipelining\."},

    # FSDP
    {"reason": "FSDP",
     "file": r"^distributed\.fsdp\."},
    {"reason": "FSDP",
     "file": r"^distributed\._composable\.fsdp\."},

    # 2D FSDP / composability
    {"reason": "2D FSDP",
     "file": r"^distributed\._composable\.test_composability"},

    # DDP / replicate
    {"reason": "DDP",
     "file": r"^distributed\._composable\.test_replicate"},

    # Process Group / c10d
    {"reason": "Process Group",
     "file": r"^distributed\.test_c10d_"},

    # PT2.0 - Distributed (dynamo_distributed)
    {"reason": "PT2.0 - Distributed",
     "file": r"^distributed\.test_dynamo_distributed$"},

    # Collectives (tensor ops, composability, nccl)
    {"reason": "Collectives",
     "file": r"^distributed\.tensor\.test_"},
    {"reason": "Collectives",
     "file": r"^distributed\.test_composability$"},
    {"reason": "Collectives",
     "file": r"^distributed\.test_nccl$"},

    # Distributed tools
    {"reason": "Misc",
     "file": r"^distributed\._tools\."},

    # Distributed elastic
    {"reason": "elastic",
     "file": r"^distributed\.elastic\."},

    # Distributed quantization
    {"reason": "Quantization",
     "file": r"^distributed\.algorithms\.quantization"},

    # Distributed rpc
    {"reason": "Misc",
     "file": r"^distributed\.rpc\."},

    # Distributed spawn
    {"reason": "Misc",
     "file": r"^distributed\.test_distributed_spawn"},

    # Distributed (generic catch-all)
    {"reason": "Misc",
     "file": r"^distributed\."},

    # ==================================================================
    # TIER 5: Generic message fallbacks
    # ==================================================================

    # "Test is disabled" messages
    {"reason": "Misc",
     "msg": r"Test is disabled because an issue exists disabling it"},

    # Generic skipIfRocm / skipCUDAIfRocm
    {"reason": "Misc",
     "msg": r"skipIfRocm.*doesn't currently work on the ROCm stack"},
    {"reason": "Misc",
     "msg": r"skipCUDAIfRocm.*doesn't currently work on the ROCm stack"},

    # "Skipped!" / "Skipped"
    {"reason": "Misc",
     "msg": r"^Skipped!?$"},

    # "Skipped on ROCm"
    {"reason": "Misc",
     "msg": r"^Skipped on ROCm$"},

    # Not supported on ROCm (generic)
    {"reason": "Will not be supported on ROCm",
     "msg": r"Not supported on ROCm"},

    # ==================================================================
    # TIER 6: Catch-all for remaining test_cuda (no message, generic class)
    # ==================================================================
    {"reason": "Misc",
     "file": r"^test_cuda$"},
]


def extract_message(raw_msg: str) -> str:
    """Extract a clean message string from the raw CSV message_rocm value."""
    if not raw_msg or raw_msg.strip() == '':
        return ''
    try:
        d = ast.literal_eval(raw_msg)
        if isinstance(d, dict):
            return d.get('message', str(d))
    except (ValueError, SyntaxError):
        pass
    return raw_msg.strip()


def classify_test(msg: str, test_file: str, test_class: str, test_name: str,
                  workflow: str = '') -> str | None:
    """Return the skip_reason for a test, or None if no rule matches."""
    for rule in RULES:
        match = True
        if 'msg' in rule:
            if not re.search(rule['msg'], msg, re.IGNORECASE):
                match = False
        if 'file' in rule and match:
            if not re.search(rule['file'], test_file):
                match = False
        if 'cls' in rule and match:
            if not re.search(rule['cls'], test_class):
                match = False
        if 'name' in rule and match:
            if not re.search(rule['name'], test_name):
                match = False
        if 'workflow' in rule and match:
            if workflow and workflow != rule['workflow']:
                match = False
        if match:
            return rule['reason']
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Auto-classify skip reasons for ROCm parity CSVs')
    parser.add_argument('-i', '--input', required=True,
                        help='Input parity CSV file')
    parser.add_argument('-o', '--output',
                        help='Output CSV with auto-classified skip_reason column')
    parser.add_argument('--tsv-out',
                        help='Also write a TSV file in skip_reasons format '
                             '(compatible with --skip_reasons in summarize_xml_testreports.py)')
    parser.add_argument('--only-unclassified', action='store_true',
                        help='Only classify tests that have no skip_reason (default)')
    parser.add_argument('--reclassify-all', action='store_true',
                        help='Re-classify all tests, overwriting existing skip_reason')
    parser.add_argument('--report', action='store_true',
                        help='Print classification report to stderr')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print report but do not write output files')
    return parser.parse_args()


def detect_columns(fieldnames):
    """Detect primary/secondary status and primary message columns.

    The parity workflow lets callers label either side (for example
    ``preview`` instead of ``rocm``), so generated columns are not limited to
    the historical status_rocm/status_cuda or status_set1/status_set2 pairs.
    Preserve those well-known pairs, then fall back to the status columns in
    CSV order, preferring status_cuda as the secondary side when present.
    """
    if 'status_rocm' in fieldnames:
        return 'status_rocm', 'status_cuda', 'message_rocm'
    if 'status_set1' in fieldnames:
        return 'status_set1', 'status_set2', 'message_set1'

    status_cols = [name for name in fieldnames if name.startswith('status_')]
    if len(status_cols) < 2:
        raise ValueError(f"Cannot detect status columns. Available: {fieldnames}")

    if 'status_cuda' in status_cols:
        secondary = 'status_cuda'
        primary = next((name for name in status_cols if name != secondary), None)
    else:
        primary, secondary = status_cols[:2]

    message = primary.replace('status_', 'message_', 1) if primary else ''
    if not primary or message not in fieldnames:
        raise ValueError(f"Cannot detect status/message columns. Available: {fieldnames}")
    return primary, secondary, message


def main():
    args = parse_args()

    rows = []
    with open(args.input, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            rows.append(row)

    # A partial download can legitimately produce an empty parity CSV. There
    # is nothing to enrich, and failing here would suppress later diagnostic
    # steps in callers that do not use an always() condition.
    if not fieldnames:
        print(
            f"WARNING: Empty parity CSV {args.input}; skipping classification",
            file=sys.stderr,
        )
        return

    col_rocm, col_cuda, col_msg = detect_columns(fieldnames)

    for col in ('skip_reason', 'assignee', 'comments'):
        if col not in fieldnames:
            fieldnames.append(col)

    classified_count = 0
    already_had_count = 0
    unclassified_count = 0
    overwritten_count = 0
    auto_reasons = Counter()
    unclassified_msgs = Counter()
    unclassified_files = Counter()
    unclassified_details = []

    tsv_entries = []

    for row in rows:
        status_rocm = row.get(col_rocm, '')
        status_cuda = row.get(col_cuda, '')
        existing_reason = row.get('skip_reason', '').strip()

        needs_reason = (
            status_rocm in ('SKIPPED', 'MISSED')
            and status_cuda == 'PASSED'
        )

        if not needs_reason:
            continue

        raw_msg = row.get(col_msg, '')
        msg = extract_message(raw_msg)
        test_file = row.get('test_file', '')
        test_class = row.get('test_class', '')
        test_name = row.get('test_name', '')
        workflow = row.get('test_config', '')

        if existing_reason and not args.reclassify_all:
            already_had_count += 1
            tsv_entries.append({
                'test_file': test_file,
                'test_name': test_name,
                'test_class': test_class,
                'skip_reason': existing_reason,
                'assignee': row.get('assignee', ' '),
                'comments': row.get('comments', ' '),
            })
            continue

        reason = classify_test(msg, test_file, test_class, test_name, workflow)

        if reason:
            if existing_reason and existing_reason != reason:
                overwritten_count += 1
            row['skip_reason'] = reason
            row.setdefault('assignee', '')
            row.setdefault('comments', 'auto-classified')
            classified_count += 1
            auto_reasons[reason] += 1
            tsv_entries.append({
                'test_file': test_file,
                'test_name': test_name,
                'test_class': test_class,
                'skip_reason': reason,
                'assignee': row.get('assignee', ' ') if not args.reclassify_all else ' ',
                'comments': 'auto-classified',
            })
        else:
            unclassified_count += 1
            display_msg = msg[:100] if msg else '(no message)'
            unclassified_msgs[display_msg] += 1
            unclassified_files[test_file] += 1
            unclassified_details.append(
                f"  {test_file:55s} {test_class:45s} {test_name[:40]:42s} {display_msg[:50]}")

    if args.report or args.dry_run:
        total = already_had_count + classified_count + unclassified_count
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"AUTO-CLASSIFICATION REPORT", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        print(f"Already had skip_reason:  {already_had_count}", file=sys.stderr)
        print(f"Auto-classified:          {classified_count}", file=sys.stderr)
        if overwritten_count:
            print(f"  (overwritten existing:  {overwritten_count})", file=sys.stderr)
        print(f"Still unclassified:       {unclassified_count}", file=sys.stderr)
        if total:
            pct = (already_had_count + classified_count) / total * 100
            print(f"Coverage:                 {pct:.1f}%", file=sys.stderr)
        print(f"Total target tests:       {total}", file=sys.stderr)

        if auto_reasons:
            print(f"\nAuto-classified by category:", file=sys.stderr)
            for reason, cnt in auto_reasons.most_common():
                print(f"  {cnt:5d}  {reason}", file=sys.stderr)

        if unclassified_msgs:
            print(f"\nUnclassified — top messages:", file=sys.stderr)
            for msg_key, cnt in unclassified_msgs.most_common(15):
                print(f"  {cnt:5d}  {msg_key}", file=sys.stderr)

        if unclassified_files:
            print(f"\nUnclassified — top files:", file=sys.stderr)
            for f, cnt in unclassified_files.most_common(15):
                print(f"  {cnt:5d}  {f}", file=sys.stderr)

        if unclassified_details and len(unclassified_details) <= 50:
            print(f"\nUnclassified tests:", file=sys.stderr)
            for d in unclassified_details:
                print(d, file=sys.stderr)

    if args.dry_run:
        return

    if not args.output:
        print("No --output specified; use --dry-run for report-only mode.",
              file=sys.stderr)
        sys.exit(1)

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    if args.tsv_out and tsv_entries:
        with open(args.tsv_out, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['test_file', 'test_name', 'test_class',
                            'skip_reason', 'assignee', 'comments'],
                delimiter='\t',
            )
            writer.writeheader()
            for entry in tsv_entries:
                writer.writerow(entry)
        print(f"\nWrote TSV with {len(tsv_entries)} entries to {args.tsv_out}",
              file=sys.stderr)

    print(f"Wrote {len(rows)} rows to {args.output}", file=sys.stderr)


if __name__ == '__main__':
    main()
