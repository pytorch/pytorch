#include <ATen/native/sparse/cuda/cuSPARSELtOps.h>
#include <c10/util/hash.h>
#include <atomic>
#include <unordered_map>
#include <mutex>
#include <string_view>
#include <compare>
#if AT_CUSPARSELT_ENABLED()

namespace at::native {

// Ideally we would use the same DeviceThreadHandlePool mechanism as used in
// aten/src/ATen/cuda/CuSparseHandlePool.cpp which would handle this for us.
// However, the cuSPARSELt handle signature is different from that of
// cuSPARSE/cuBLAS, so it's not possible to reuse the existing pooling
// mechanism. Instead we have to handle our handles ourselves, which is why
// these variables are thread local. Once cuSPARSELt updates their handle
// signature to be consistent with the rest of CUDA, we can switch to using
// DeviceThreadHandlePool.
thread_local cusparseLtHandle_t handle;
thread_local bool handle_initialized = false;

static std::atomic<int64_t> g_cslt_plan_init_count{0};
static std::atomic<int64_t> g_cslt_cache_hit_count{0};
static constexpr int64_t kLogInterval = 100;
static constexpr size_t kMaxPlanCacheSize = 1024;

// ---------------------------------------------------------------------------
// Plan Cache: avoids recreating cuSPARSELt descriptors / plan / workspace on
// every _cslt_sparse_mm call. In inference the set of (m,k,n,dtype,...) tuples
// is small and fixed, so caching gives a large CPU-side speedup.
// ---------------------------------------------------------------------------
struct CuSparseLtPlanCacheKey {
  int64_t m, k, n;
  cudaDataType input_type;
  cudaDataType output_type;
  cudaDataType C_type;
  cusparseComputeType compute_type;
  bool transpose_result;
  bool is_B_contiguous;
  int alg_id;
  int split_k;
  int split_k_mode;
  bool has_bias;
  int tensor_alpha_mode;

  bool operator==(const CuSparseLtPlanCacheKey&) const = default;
};

struct CuSparseLtPlanCacheKeyHash {
  size_t operator()(const CuSparseLtPlanCacheKey& k) const {
    return c10::get_hash(
        k.m, k.k, k.n,
        k.input_type, k.output_type, k.C_type,
        k.compute_type,
        k.transpose_result, k.is_B_contiguous,
        k.alg_id, k.split_k, k.split_k_mode,
        k.has_bias, k.tensor_alpha_mode);
  }
};

struct CachedPlanEntry {
  cusparseLtMatDescriptor_t sparse_input_descriptor;
  cusparseLtMatDescriptor_t dense_input_descriptor;
  cusparseLtMatDescriptor_t res_descriptor;
  cusparseLtMatDescriptor_t C_descriptor;
  cusparseLtMatmulDescriptor_t matmul;
  cusparseLtMatmulAlgSelection_t alg_sel;
  cusparseLtMatmulPlan_t plan;
  size_t workspace_size;
  bool initialized = false;

  CachedPlanEntry() = default;
  CachedPlanEntry(const CachedPlanEntry&) = delete;
  CachedPlanEntry& operator=(const CachedPlanEntry&) = delete;
  CachedPlanEntry(CachedPlanEntry&& other) noexcept
      : sparse_input_descriptor(other.sparse_input_descriptor),
        dense_input_descriptor(other.dense_input_descriptor),
        res_descriptor(other.res_descriptor),
        C_descriptor(other.C_descriptor),
        matmul(other.matmul),
        alg_sel(other.alg_sel),
        plan(other.plan),
        workspace_size(other.workspace_size),
        initialized(other.initialized) {
    other.initialized = false;
  }
  CachedPlanEntry& operator=(CachedPlanEntry&& other) noexcept {
    if (this != &other) {
      destroy();
      sparse_input_descriptor = other.sparse_input_descriptor;
      dense_input_descriptor = other.dense_input_descriptor;
      res_descriptor = other.res_descriptor;
      C_descriptor = other.C_descriptor;
      matmul = other.matmul;
      alg_sel = other.alg_sel;
      plan = other.plan;
      workspace_size = other.workspace_size;
      initialized = other.initialized;
      other.initialized = false;
    }
    return *this;
  }

  ~CachedPlanEntry() {
    destroy();
  }

 private:
  void destroy() {
    if (!initialized) {
      return;
    }
    cusparseLtMatmulPlanDestroy(&plan);
    cusparseLtMatDescriptorDestroy(&sparse_input_descriptor);
    cusparseLtMatDescriptorDestroy(&dense_input_descriptor);
    cusparseLtMatDescriptorDestroy(&res_descriptor);
    cusparseLtMatDescriptorDestroy(&C_descriptor);
    initialized = false;
  }
};

using PlanCache = std::unordered_map<
    CuSparseLtPlanCacheKey,
    CachedPlanEntry,
    CuSparseLtPlanCacheKeyHash>;

// Thread-local so no locking is needed; matches the thread-local handle.
thread_local PlanCache plan_cache;

#ifdef USE_ROCM
// Single global flag for platform-wide hipSparseLt support
c10::once_flag g_hipSparseLtSupportInitFlag;
static bool g_hipSparseLtSupported = false;

// Initialize the hipSparseLt support status once for the platform
static void initHipSparseLtSupport() {
    // Default to not supported
    g_hipSparseLtSupported = false;

    // Check only the first available device
    try {
        if (at::cuda::device_count() > 0) {
            g_hipSparseLtSupported = at::detail::getCUDAHooks().isGPUArch({"gfx950", "gfx942"}, 0);
        }
    } catch (const std::exception&) {
        // If an exception occurs during device property check, we assume hipSparseLt is not supported
        // This could happen due to driver issues, device access problems, or other runtime errors
        g_hipSparseLtSupported = false;
        TORCH_WARN("Exception occurred while checking hipSparseLt support. Assuming not supported.");
    }
}

static bool isHipSparseLtSupported() {
    // Initialize support check only once
    c10::call_once(g_hipSparseLtSupportInitFlag, initHipSparseLtSupport);

    // Return cached result (platform-wide)
    if (!g_hipSparseLtSupported) {
        TORCH_CHECK(
            false,
            "hipSparseLt not supported on this device, supported architectures: "
            "gfx950, gfx942. "
            "required ROCM version: 6.4.0 or later.");
    }
    return g_hipSparseLtSupported;
}
#endif

at::Tensor _cslt_compress(const Tensor& sparse_input) {
  if (!handle_initialized) {
    TORCH_CUDASPARSE_CHECK(cusparseLtInit(&handle));
    handle_initialized = true;
  }
  // create sparse descriptor, dtype
  cusparseLtMatDescriptor_t sparse_input_descriptor;
  cudaDataType type;

  #ifdef USE_ROCM
  TORCH_CHECK(isHipSparseLtSupported());
  #endif

  switch (sparse_input.scalar_type()) {
    case at::ScalarType::Char:
      type = CUDA_R_8I;
      break;
    case at::ScalarType::Half:
      type = CUDA_R_16F;
      break;
    case at::ScalarType::BFloat16:
      type = CUDA_R_16BF;
      break;
#ifndef USE_ROCM
    case at::ScalarType::Float:
      type = CUDA_R_32F;
      break;
#endif
#if defined(CUSPARSELT_VERSION) && CUSPARSELT_VERSION >= 602 || defined(USE_ROCM)
    case at::ScalarType::Float8_e4m3fn:
      type = CUDA_R_8F_E4M3;
      break;
#endif
    default:
      TORCH_CHECK(false, "Unsupported dtype for cuSPARSELt/hipSparseLt compressed matrix");
      break;
  }

  TORCH_CUDASPARSE_CHECK(cusparseLtStructuredDescriptorInit(
      &handle,
      &sparse_input_descriptor,
      sparse_input.size(0),
      sparse_input.size(1),
      sparse_input.size(1),
      16,
      type,
      CUSPARSE_ORDER_ROW,
      CUSPARSELT_SPARSITY_50_PERCENT));

  // compress input
  //--------------------------------------------------------------------------
  size_t compressed_size, compressed_buffer_size;
  TORCH_CUDASPARSE_CHECK(cusparseLtSpMMACompressedSize2(
      &handle,
      &sparse_input_descriptor,
      &compressed_size,
      &compressed_buffer_size));

  // create a new compressed tensor with the same dtype as the input,
  // and with packed data/metadata stored in an array with original
  // number of rows, and sufficient columns to provide compressed_size
  // buffer (in bytes)
  size_t orig_m = sparse_input.size(0);
  size_t div = orig_m * sparse_input.itemsize();
  size_t new_n = (compressed_size + div - 1) / div; // ceil(s,d) = (s+d-1)/d
  auto compressed_tensor = sparse_input.new_empty({(int64_t)orig_m, (int64_t)new_n});

  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto compressedBufferPtr = allocator.allocate(compressed_buffer_size);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  TORCH_CUDASPARSE_CHECK(cusparseLtSpMMACompress2(
      &handle,
      &sparse_input_descriptor,
      true,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      sparse_input.data_ptr(),
      compressed_tensor.data_ptr(),
      compressedBufferPtr.get(),
      stream));

  return compressed_tensor;
}

std::tuple<at::Tensor, int64_t, int64_t, int64_t, int64_t> _cslt_sparse_mm_impl(
    const Tensor& compressed_A,
    const Tensor& dense_B,
    const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& alpha_opt,
    const std::optional<c10::ScalarType> out_dtype_opt,
    bool transpose_result,
    int alg_id,
    int split_k,
    int split_k_mode,
    bool search_alg_id) {
  if (!handle_initialized) {
    TORCH_CUDASPARSE_CHECK(cusparseLtInit(&handle));
    handle_initialized = true;
  }

  cudaDataType input_type;
  cudaDataType output_type;
  cudaDataType C_type;
  cusparseComputeType compute_type;

  #ifdef USE_ROCM
  TORCH_CHECK(isHipSparseLtSupported());
  #endif

  switch (compressed_A.scalar_type()) {
    case at::ScalarType::Char:
      input_type = CUDA_R_8I;
      output_type = CUDA_R_8I;
      C_type = CUDA_R_8I;
      compute_type = CUSPARSE_COMPUTE_32I;
      break;

// cuSPARSELt v0.5.2 onwards changes CUSPARSE_COMPUTE_TF32, CUSPARSE_COMPUT_16F
// to CUSPARSE_COMPUTE_32F
#if defined(CUSPARSELT_VERSION) && CUSPARSELT_VERSION >= 502 || defined(USE_ROCM)
    case at::ScalarType::Half:
      input_type = CUDA_R_16F;
      output_type = CUDA_R_16F;
      C_type = CUDA_R_16F;
      compute_type = CUSPARSE_COMPUTE_32F;
      break;
    case at::ScalarType::BFloat16:
      input_type = CUDA_R_16BF;
      output_type = CUDA_R_16BF;
      C_type = CUDA_R_16BF;
      compute_type = CUSPARSE_COMPUTE_32F;
      break;
#ifndef USE_ROCM
    case at::ScalarType::Float:
      input_type = CUDA_R_32F;
      output_type = CUDA_R_32F;
      C_type = CUDA_R_32F;
      compute_type = CUSPARSE_COMPUTE_32F;
      break;
#endif
// cuSPARSELt >= 0.6.2 or hipSparseLt: add Float8 support
#if defined(CUSPARSELT_VERSION) && CUSPARSELT_VERSION >= 602 || defined(USE_ROCM)
    case at::ScalarType::Float8_e4m3fn:
      input_type = CUDA_R_8F_E4M3;
#ifdef USE_ROCM
      // hipSparseLt 0.2.7: FP8 input only supports FP32 output
      output_type = CUDA_R_32F;
      C_type = CUDA_R_32F;
#else
      output_type = CUDA_R_8F_E4M3;
      C_type = CUDA_R_16F;
#endif
      compute_type = CUSPARSE_COMPUTE_32F;
      break;
#endif
// cuSPARSELt <= v0.5.2 uses CUSPARSE_COMPUTE_TF32, CUSPARSE_COMPUTE_16F
#else
    case at::ScalarType::Half:
      input_type = CUDA_R_16F;
      output_type = CUDA_R_16F;
      C_type = CUDA_R_16F;
      compute_type = CUSPARSE_COMPUTE_16F;
      break;
    case at::ScalarType::BFloat16:
      input_type = CUDA_R_16BF;
      output_type = CUDA_R_16BF;
      C_type = CUDA_R_16BF;
      compute_type = CUSPARSE_COMPUTE_16F;
      break;
    case at::ScalarType::Float:
      input_type = CUDA_R_32F;
      output_type = CUDA_R_32F;
      C_type = CUDA_R_32F;
      compute_type = CUSPARSE_COMPUTE_TF32;
      break;
#endif
    default:
      TORCH_CHECK(
          false,
          "Unsupported dtype for cuSPARSELt compressed matrix multiplication.");
      break;
  }
  ScalarType out_dtype = dense_B.scalar_type();
  // special check for mixed dtype support for 8 bit dtypes
  // cslt 0.5.2+: int8 int8 -> {fp16, bf16, int32} support
  if (out_dtype_opt.has_value()) {
    out_dtype = out_dtype_opt.value();
    if (input_type == CUDA_R_8I) {
      switch (out_dtype) {
        case at::ScalarType::Half:
          C_type = CUDA_R_16F;
          output_type = CUDA_R_16F;
          break;
        case at::ScalarType::BFloat16:
          C_type = CUDA_R_16BF;
          output_type = CUDA_R_16BF;
          break;
        case at::ScalarType::Int:
          C_type = CUDA_R_32I;
          output_type = CUDA_R_32I;
          break;
        default:
          TORCH_CHECK(
              false,
              "Unsupported out_dtype passed, must be one of {fp16, bf16, int32} for int8 inputs");
          break;
      }
    }
// cslt 0.6.2+ or hipSparseLt: fp8 output dtype support
#if defined(CUSPARSELT_VERSION) && CUSPARSELT_VERSION >= 602 || defined(USE_ROCM)
    else if (input_type == CUDA_R_8F_E4M3) {
      switch (out_dtype) {
#ifndef USE_ROCM
        case at::ScalarType::Float8_e4m3fn:
          output_type = CUDA_R_8F_E4M3;
          C_type = CUDA_R_16F;
          break;
        case at::ScalarType::Half:
          output_type = CUDA_R_16F;
          C_type = CUDA_R_16F;
          break;
        case at::ScalarType::BFloat16:
          output_type = CUDA_R_16BF;
          C_type = CUDA_R_16BF;
          break;
#endif
        case at::ScalarType::Float:
          output_type = CUDA_R_32F;
          C_type = CUDA_R_32F;
          break;
        default:
          TORCH_CHECK(
              false,
#ifdef USE_ROCM
              "Unsupported out_dtype passed, must be float32 for fp8 inputs on ROCm");
#else
              "Unsupported out_dtype passed, must be one of {fp8, fp16, bf16, float32} for fp8 inputs");
#endif
          break;
      }
    }
#endif
    else {
      TORCH_CHECK(
          false, "out_dtype support only available for int8/fp8 inputs");
    }
  }

  TORCH_INTERNAL_ASSERT(compressed_A.dim() == 2); // encoded M x S
  int64_t k = dense_B.size(0);
  int64_t n = dense_B.size(1);
  int64_t m = compressed_A.size(0);

  // Resolve alpha early so we can include tensor_alpha_mode in cache key.
  int tensor_alpha_mode = 0;
  float alpha = 1.0;
  float beta = 0.0;
  const auto alpha_tensor = alpha_opt.has_value() ? *alpha_opt : Tensor{};
  auto alpha_ptr = &alpha;
  if (alpha_opt.has_value()) {
    if (alpha_tensor.numel() == 1) {
      alpha = alpha_tensor.item<float>();
    } else {
      tensor_alpha_mode = 1;
      alpha_ptr = static_cast<float*>(alpha_tensor.data_ptr());
    }
  }

  bool has_bias = bias_opt.has_value();
  const void* bias_data_ptr =
      has_bias ? bias_opt.value().data_ptr() : nullptr;
  bool b_is_contiguous = dense_B.is_contiguous();

  // ------------------------------------------------------------------
  // Helper lambda: initialise all cuSPARSELt objects inside *entry*.
  // The objects MUST be initialised at their final address because the
  // library may store internal cross-pointers (e.g. the plan references
  // the matmul descriptor, which references the matrix descriptors).
  // ------------------------------------------------------------------
  auto init_plan_entry = [&](CachedPlanEntry& entry) {
    TORCH_CUDASPARSE_CHECK(cusparseLtStructuredDescriptorInit(
        &handle,
        &entry.sparse_input_descriptor,
        m,
        k,
        k,
        16,
        input_type,
        CUSPARSE_ORDER_ROW,
        CUSPARSELT_SPARSITY_50_PERCENT));

    TORCH_CUDASPARSE_CHECK(cusparseLtDenseDescriptorInit(
        &handle,
        &entry.dense_input_descriptor,
        b_is_contiguous ? k : n,
        b_is_contiguous ? n : k,
        b_is_contiguous ? n : k,
        16,
        input_type,
        CUSPARSE_ORDER_ROW));

    TORCH_CUDASPARSE_CHECK(cusparseLtDenseDescriptorInit(
        &handle,
        &entry.res_descriptor,
        m,
        n,
        transpose_result ? m : n,
        16,
        output_type,
        transpose_result ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW));

    TORCH_CUDASPARSE_CHECK(cusparseLtDenseDescriptorInit(
        &handle,
        &entry.C_descriptor,
        m,
        n,
        transpose_result ? m : n,
        16,
        C_type,
        transpose_result ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW));

    TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescriptorInit(
        &handle,
        &entry.matmul,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        b_is_contiguous ? CUSPARSE_OPERATION_NON_TRANSPOSE
                        : CUSPARSE_OPERATION_TRANSPOSE,
        &entry.sparse_input_descriptor,
        &entry.dense_input_descriptor,
        &entry.C_descriptor,
        &entry.res_descriptor,
        compute_type));

    if (has_bias) {
      void* dBias = const_cast<void*>(bias_data_ptr);
      TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescSetAttribute(
          &handle,
          &entry.matmul,
          CUSPARSELT_MATMUL_BIAS_POINTER,
          &dBias,
          sizeof(dBias)));
    }

    TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgSelectionInit(
        &handle,
        &entry.alg_sel,
        &entry.matmul,
        CUSPARSELT_MATMUL_ALG_DEFAULT));

    TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgSetAttribute(
        &handle,
        &entry.alg_sel,
        CUSPARSELT_MATMUL_ALG_CONFIG_ID,
        &alg_id,
        sizeof(alg_id)));

    if (split_k != 1) {
      TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgSetAttribute(
          &handle, &entry.alg_sel,
          CUSPARSELT_MATMUL_SPLIT_K,
          &split_k,
          sizeof(split_k)));
      if (split_k_mode > 0) {
        auto skMode = static_cast<cusparseLtSplitKMode_t>(split_k_mode);
        TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgSetAttribute(
            &handle,
            &entry.alg_sel,
            CUSPARSELT_MATMUL_SPLIT_K_MODE,
            &skMode,
            sizeof(skMode)));
      }
    }

    if (tensor_alpha_mode == 1) {
      TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescSetAttribute(
          &handle,
          &entry.matmul,
          CUSPARSELT_MATMUL_ALPHA_VECTOR_SCALING,
          &tensor_alpha_mode,
          sizeof(tensor_alpha_mode)));
    }

    TORCH_CUDASPARSE_CHECK(
        cusparseLtMatmulPlanInit(&handle, &entry.plan, &entry.matmul, &entry.alg_sel));

    int64_t cnt = g_cslt_plan_init_count.fetch_add(1, std::memory_order_relaxed) + 1;
    if (cnt % kLogInterval == 0 || cnt <= 3) {
      TORCH_WARN("[cuSPARSELt CACHE MISS] cusparseLtMatmulPlanInit called ", cnt,
                 " times, m=", m, " k=", k, " n=", n, " alg_id=", alg_id);
    }

    TORCH_CUDASPARSE_CHECK(
        cusparseLtMatmulGetWorkspace(&handle, &entry.plan, &entry.workspace_size));

    entry.initialized = true;
  };

  // ------------------------------------------------------------------
  // Fast path: reuse a cached plan when not doing algorithm search.
  // ------------------------------------------------------------------
  if (!search_alg_id) {
    CuSparseLtPlanCacheKey cache_key{
        m, k, n,
        input_type, output_type, C_type, compute_type,
        transpose_result, b_is_contiguous,
        alg_id, split_k, split_k_mode,
        has_bias, tensor_alpha_mode};

    auto cache_it = plan_cache.find(cache_key);

    if (cache_it != plan_cache.end()) {
      int64_t hit_cnt = g_cslt_cache_hit_count.fetch_add(1, std::memory_order_relaxed) + 1;
      if (hit_cnt % kLogInterval == 0 || hit_cnt <= 3) {
        TORCH_WARN("[cuSPARSELt CACHE HIT] count=", hit_cnt,
                   ", m=", m, " k=", k, " n=", n);
      }
    }

    if (cache_it == plan_cache.end()) {
      // ---- cache miss: evict all if cache is full -------------------
      if (plan_cache.size() >= kMaxPlanCacheSize) {
        plan_cache.clear();
      }
      // ---- init directly at stable map address ----------------------
      auto [it, inserted] = plan_cache.try_emplace(cache_key);
      struct CacheEraseGuard {
        PlanCache& cache;
        PlanCache::iterator iter;
        bool committed = false;
        ~CacheEraseGuard() {
          if (!committed) {
            cache.erase(iter);
          }
        }
      } guard{plan_cache, it};
      init_plan_entry(it->second);
      guard.committed = true;
      cache_it = it;
    }

    // ---- execute matmul from cached plan ----------------------------
    auto& cached = cache_it->second;

    // Re-set bias pointer before every matmul because the caller may
    // pass a different tensor each time (same shape but different
    // data_ptr due to PyTorch's caching allocator).
    if (has_bias) {
      void* dBias = const_cast<void*>(bias_data_ptr);
      TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescSetAttribute(
          &handle,
          &cached.matmul,
          CUSPARSELT_MATMUL_BIAS_POINTER,
          &dBias,
          sizeof(dBias)));
    }

    auto res_tensor_options =
        c10::TensorOptions().dtype(out_dtype).device(dense_B.device());
    at::Tensor res = transpose_result
        ? at::empty({n, m}, res_tensor_options)
        : at::empty({m, n}, res_tensor_options);

    auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
    auto workspacePtr = allocator.allocate(cached.workspace_size);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    TORCH_CUDASPARSE_CHECK(cusparseLtMatmul(
        &handle,
        &cached.plan,
        alpha_ptr,
        compressed_A.data_ptr(),
        dense_B.data_ptr(),
        &beta,
        res.data_ptr(),
        res.data_ptr(),
        workspacePtr.get(),
        &stream,
        1));

    return {
        res,
        alg_id,
        split_k,
        static_cast<int64_t>(split_k_mode),
        0};
  }

  // ------------------------------------------------------------------
  // Search path: build everything on the stack, destroy after search.
  // ------------------------------------------------------------------
  cusparseLtMatDescriptor_t sparse_input_descriptor;
  TORCH_CUDASPARSE_CHECK(cusparseLtStructuredDescriptorInit(
      &handle,
      &sparse_input_descriptor,
      m,
      k,
      k,
      16,
      input_type,
      CUSPARSE_ORDER_ROW,
      CUSPARSELT_SPARSITY_50_PERCENT));

  cusparseLtMatDescriptor_t dense_input_descriptor;
  TORCH_CUDASPARSE_CHECK(cusparseLtDenseDescriptorInit(
      &handle, &dense_input_descriptor,
      b_is_contiguous ? k : n,
      b_is_contiguous ? n : k,
      b_is_contiguous ? n : k,
      16,
      input_type,
      CUSPARSE_ORDER_ROW));

  auto res_tensor_options =
      c10::TensorOptions().dtype(out_dtype).device(dense_B.device());
  at::Tensor res = transpose_result
      ? at::empty({n, m}, res_tensor_options)
      : at::empty({m, n}, res_tensor_options);

  cusparseLtMatDescriptor_t res_descriptor;
  TORCH_CUDASPARSE_CHECK(cusparseLtDenseDescriptorInit(
      &handle,
      &res_descriptor,
      m,
      n,
      transpose_result ? m : n,
      16,
      output_type,
      transpose_result ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW));

  cusparseLtMatDescriptor_t C_descriptor;
  TORCH_CUDASPARSE_CHECK(cusparseLtDenseDescriptorInit(
      &handle,
      &C_descriptor,
      m,
      n,
      transpose_result ? m : n,
      16,
      C_type,
      transpose_result ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW));

  cusparseLtMatmulDescriptor_t matmul;
  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescriptorInit(
      &handle,
      &matmul,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      b_is_contiguous ? CUSPARSE_OPERATION_NON_TRANSPOSE
                      : CUSPARSE_OPERATION_TRANSPOSE,
      &sparse_input_descriptor,
      &dense_input_descriptor,
      &C_descriptor,
      &res_descriptor,
      compute_type));

  if (has_bias) {
    void* dBias = const_cast<void*>(bias_data_ptr);
    TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescSetAttribute(
        &handle,
        &matmul,
        CUSPARSELT_MATMUL_BIAS_POINTER,
        &dBias,
        sizeof(dBias)));
  }

  cusparseLtMatmulAlgSelection_t alg_sel;
  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgSelectionInit(
      &handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT));

  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgSetAttribute(
      &handle,
      &alg_sel,
      CUSPARSELT_MATMUL_ALG_CONFIG_ID,
      &alg_id,
      sizeof(alg_id)));

  cusparseLtSplitKMode_t splitKMode;
  int max_alg_id = 0;
  if (split_k != 1) {
    TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgSetAttribute(
        &handle,
        &alg_sel,
        CUSPARSELT_MATMUL_SPLIT_K,
        &split_k,
        sizeof(split_k)));
    if (split_k_mode > 0) {
      splitKMode = static_cast<cusparseLtSplitKMode_t>(split_k_mode);
      TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgSetAttribute(
          &handle,
          &alg_sel,
          CUSPARSELT_MATMUL_SPLIT_K_MODE,
          &splitKMode,
          sizeof(splitKMode)));
    }
  }

  if (tensor_alpha_mode == 1) {
    TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescSetAttribute(
        &handle,
        &matmul,
        CUSPARSELT_MATMUL_ALPHA_VECTOR_SCALING,
        &tensor_alpha_mode,
        sizeof(tensor_alpha_mode)));
  }

  cusparseLtMatmulPlan_t plan;
  TORCH_CUDASPARSE_CHECK(
      cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel));

  size_t workspace_size;
  TORCH_CUDASPARSE_CHECK(
      cusparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size));

  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto workspacePtr = allocator.allocate(workspace_size);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulSearch(
      &handle,
      &plan,
      alpha_ptr,
      compressed_A.data_ptr(),
      dense_B.data_ptr(),
      &beta,
      res.data_ptr(),
      res.data_ptr(),
      workspacePtr.get(),
      &stream,
      1));

  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgGetAttribute(
      &handle,
      &alg_sel,
      CUSPARSELT_MATMUL_ALG_CONFIG_ID,
      &alg_id,
      sizeof(alg_id)));
  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgGetAttribute(
      &handle,
      &alg_sel,
      CUSPARSELT_MATMUL_SPLIT_K,
      &split_k,
      sizeof(split_k)));
  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgGetAttribute(
      &handle,
      &alg_sel,
      CUSPARSELT_MATMUL_SPLIT_K_MODE,
      &splitKMode,
      sizeof(splitKMode)));
  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgGetAttribute(
      &handle,
      &alg_sel,
      CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID,
      &max_alg_id,
      sizeof(max_alg_id)));

  TORCH_CUDASPARSE_CHECK(
      cusparseLtMatDescriptorDestroy(&sparse_input_descriptor));
  TORCH_CUDASPARSE_CHECK(
      cusparseLtMatDescriptorDestroy(&dense_input_descriptor));
  TORCH_CUDASPARSE_CHECK(cusparseLtMatDescriptorDestroy(&res_descriptor));
  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulPlanDestroy(&plan));

  return {
      res,
      alg_id,
      split_k,
      static_cast<int64_t>(splitKMode),
      max_alg_id};
}

at::Tensor _cslt_sparse_mm(
    const Tensor& compressed_A,
    const Tensor& dense_B,
    const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& alpha_opt,
    const std::optional<c10::ScalarType> out_dtype_opt,
    bool transpose_result,
    int64_t alg_id,
    int64_t split_k,
    int64_t split_k_mode) {
  auto result = _cslt_sparse_mm_impl(
      compressed_A,
      dense_B,
      bias_opt,
      alpha_opt,
      out_dtype_opt,
      transpose_result,
      (int)alg_id,
      (int)split_k,
      (int)split_k_mode,
      false);
  return std::get<0>(result);
}

int64_t _cslt_sparse_mm_search(
    const Tensor& compressed_A,
    const Tensor& dense_B,
    const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& alpha_opt,
    const std::optional<c10::ScalarType> out_dtype_opt,
    bool transpose_result) {
  TORCH_WARN_ONCE(
      "torch._cslt_sparse_mm_search is deprecated and will be removed in a future PyTorch release. Please use torch._C._cusparselt.mm_search instead.");
  int alg_id_int = 0;
  int split_k = 1;
  int split_k_mode = -1;
  auto result = _cslt_sparse_mm_impl(
      compressed_A,
      dense_B,
      bias_opt,
      alpha_opt,
      out_dtype_opt,
      transpose_result,
      alg_id_int,
      split_k,
      split_k_mode,
      true);
  return (int64_t)std::get<1>(result);
}

} // namespace at::native

#else // No cuSPARSELt support, report error if these functions are called.

namespace at::native {

at::Tensor _cslt_compress(const Tensor& sparse_input) {
  TORCH_CHECK(false, "cuSPARSELt not supported on your machine.");
}

at::Tensor _cslt_sparse_mm(
    const Tensor& compressed_A,
    const Tensor& dense_B,
    const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& alpha_opt,
    const std::optional<c10::ScalarType> out_dtype,
    bool transpose_result,
    int64_t alg_id,
    int64_t split_k,
    int64_t split_k_mode) {
  TORCH_CHECK(false, "cuSPARSELt not supported on your machine.");
}

int64_t _cslt_sparse_mm_search(
    const Tensor& compressed_A,
    const Tensor& dense_B,
    const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& alpha_opt,
    const std::optional<c10::ScalarType> out_dtype,
    bool transpose_result) {
  TORCH_CHECK(false, "cuSPARSELt not supported on your machine.");
}

} // namespace at::native

#endif
