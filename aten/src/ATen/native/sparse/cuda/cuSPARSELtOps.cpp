#include <ATen/native/sparse/cuda/cuSPARSELtOps.h>
#include <ATen/native/utils/ParamsHash.h>
#include <list>
#include <optional>
#include <unordered_map>
#include <mutex>
#include <string_view>
#include <utility>
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

/////////////////////////////////////////////////////////////////////

struct SparseInputDescriptorCacheKey {
  int m = 0;
  int k = 0;
  bool is_contiguous = true;
  bool is_dense = false;
  int32_t input_type = 0;
};

using SparseInputDescriptorCacheKeyHash = ParamsHash<SparseInputDescriptorCacheKey>;
using SparseInputDescriptorCacheKeyEqual = ParamsEqual<SparseInputDescriptorCacheKey>;

struct CachedSparseInputDescriptor {
  cusparseLtMatDescriptor_t descriptor;
  bool has_descriptor = false;

  CachedSparseInputDescriptor() = default;
  ~CachedSparseInputDescriptor() {
    if (has_descriptor) {
      cusparseLtMatDescriptorDestroy(&descriptor);
      has_descriptor = false;
    }
  }
  CachedSparseInputDescriptor(CachedSparseInputDescriptor&& other) noexcept
      : descriptor(other.descriptor),
        has_descriptor(other.has_descriptor) {
    other.has_descriptor = false;
  }
  CachedSparseInputDescriptor& operator=(CachedSparseInputDescriptor&& other) noexcept {
    if (this != &other) {
      if (has_descriptor) {
        cusparseLtMatDescriptorDestroy(&descriptor);
      }
      descriptor = other.descriptor;
      has_descriptor = other.has_descriptor;
      other.has_descriptor = false;
    }
    return *this;
  }
  CachedSparseInputDescriptor(const CachedSparseInputDescriptor&) = delete;
  CachedSparseInputDescriptor& operator=(const CachedSparseInputDescriptor&) = delete;
};

constexpr size_t SparseInputDescriptorCacheMaxSize = 128;

using SparseInputDescriptorLruList =
    std::list<std::pair<SparseInputDescriptorCacheKey, CachedSparseInputDescriptor>>;
using SparseInputDescriptorCacheMap = std::unordered_map<
    SparseInputDescriptorCacheKey,
    SparseInputDescriptorLruList::iterator,
    SparseInputDescriptorCacheKeyHash,
    SparseInputDescriptorCacheKeyEqual>;


struct SparseInputDescriptorCache {
  SparseInputDescriptorLruList lru_list;
  SparseInputDescriptorCacheMap cache_map;

  // Returns plan pointer and workspace_size if found; moves entry to front for LRU.
  std::optional<cusparseLtMatDescriptor_t*> lookup(
      const SparseInputDescriptorCacheKey& key) {
    auto it = cache_map.find(key);
    if (it == cache_map.end()) {
      return std::nullopt;
    }
    lru_list.splice(lru_list.begin(), lru_list, it->second);
    CachedSparseInputDescriptor& entry = it->second->second;
    return &entry.descriptor;
  }

  void insert(SparseInputDescriptorCacheKey key, CachedSparseInputDescriptor descriptor) {
    if (lru_list.size() >= SparseInputDescriptorCacheMaxSize) {
      auto& back = lru_list.back();
      cache_map.erase(back.first);
      lru_list.pop_back();
    }
    lru_list.push_front({std::move(key), std::move(descriptor)});
    cache_map[lru_list.front().first] = lru_list.begin();
  }
};

thread_local std::unique_ptr<SparseInputDescriptorCache> sparse_input_descriptor_cache;

SparseInputDescriptorCache& getSparseInputDescriptorCache() {
  if (!sparse_input_descriptor_cache) {
    sparse_input_descriptor_cache = std::make_unique<SparseInputDescriptorCache>();
  }
  return *sparse_input_descriptor_cache;
}

static cusparseLtMatDescriptor_t* _get_sparse_input_descriptor_ptr(
    cusparseLtHandle_t& handle,
    int64_t m,
    int64_t k,
    cudaDataType input_type) {
  SparseInputDescriptorCacheKey key{};
  key.m = m;
  key.k = k;
  key.is_contiguous = true;
  key.is_dense = false;
  key.input_type = static_cast<int32_t>(input_type);
  auto cached = getSparseInputDescriptorCache().lookup(key);
  if (cached.has_value()) {
    return *cached;
  } 
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
      
  CachedSparseInputDescriptor cached_descriptor;
  cached_descriptor.descriptor = sparse_input_descriptor;
  cached_descriptor.has_descriptor = true;
  getSparseInputDescriptorCache().insert(std::move(key), std::move(cached_descriptor));
  cached = getSparseInputDescriptorCache().lookup(key);
  return *cached;
}


static cusparseLtMatDescriptor_t* _get_dense_input_descriptor_ptr(
    cusparseLtHandle_t& handle,
    int64_t k,
    int64_t n,
    bool is_contiguous,
    cudaDataType input_type) {
  SparseInputDescriptorCacheKey key{};
  key.m = n;
  key.k = k;
  key.is_contiguous = is_contiguous;
  key.is_dense = true;
  key.input_type = static_cast<int32_t>(input_type);
  auto cached = getSparseInputDescriptorCache().lookup(key);
  if (cached.has_value()) {
    return *cached;
  } 
  cusparseLtMatDescriptor_t dense_input_descriptor;
  TORCH_CUDASPARSE_CHECK(cusparseLtDenseDescriptorInit(
      &handle,
      &dense_input_descriptor,
      is_contiguous ? k : n,
      is_contiguous ? n : k,
      is_contiguous ? n : k,
      16,
      input_type,
      CUSPARSE_ORDER_ROW));
      
  CachedSparseInputDescriptor cached_descriptor;
  cached_descriptor.descriptor = dense_input_descriptor;
  cached_descriptor.has_descriptor = true;
  getSparseInputDescriptorCache().insert(std::move(key), std::move(cached_descriptor));
  cached = getSparseInputDescriptorCache().lookup(key);
  return *cached;
}



//////////////////////////////////////////////////////////////////////


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
#if defined(CUSPARSELT_VERSION) && CUSPARSELT_VERSION >= 602 && !defined(USE_ROCM)
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
  // cuSPARSELt constructs
  cusparseLtMatmulDescriptor_t matmul;
  cusparseLtMatmulPlan_t plan;
  cusparseLtMatmulAlgSelection_t alg_sel;

  float alpha = 1.0;
  float beta = 0.0;
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
// if cuSPARSELt >= 6.2.3, we can add Float8 support
#if defined(CUSPARSELT_VERSION) && CUSPARSELT_VERSION >= 602 && !defined(USE_ROCM)
    case at::ScalarType::Float8_e4m3fn:
      input_type = CUDA_R_8F_E4M3;
      output_type = CUDA_R_8F_E4M3;
      C_type = CUDA_R_16F;
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
// cslt 0.6.2+: fp8 fp8 -> {fp8, fp16, bf16, fp32} support
#if defined(CUSPARSELT_VERSION) && CUSPARSELT_VERSION >= 602 && !defined(USE_ROCM)
    else if (input_type == CUDA_R_8F_E4M3) {
      switch (out_dtype) {
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
        case at::ScalarType::Float:
          output_type = CUDA_R_32F;
          C_type = CUDA_R_32F;
          break;
        default:
          TORCH_CHECK(
              false,
              "Unsupported out_dtype passed, must be one of {fp16, bf16, float32} for fp8 inputs");
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

  //////////////////////////////////////////////////////////////
  // initialize sparse descriptor
  // auto t3 = std::chrono::steady_clock::now();
  cusparseLtMatDescriptor_t* sparse_input_descriptor_ptr =
    _get_sparse_input_descriptor_ptr(handle, m, k, input_type);
  // auto t4 = std::chrono::steady_clock::now();
  // double duration_sparse_input_descriptor = std::chrono::duration<double, std::milli>(t4 - t3).count();
  // std::cout << "duration_sparse_input_descriptor = " << duration_sparse_input_descriptor << std::endl;
  /////////////////////////////////////////////////////////////

  // initialize dense input descriptor
  cusparseLtMatDescriptor_t* dense_input_descriptor_ptr = 
    _get_dense_input_descriptor_ptr(handle, k, n, dense_B.is_contiguous(), input_type);
  
  // create result tensor
  auto res_tensor_options =
      c10::TensorOptions().dtype(out_dtype).device(dense_B.device());
  at::Tensor res = (transpose_result) ? at::empty({n, m}, res_tensor_options)
                                      : at::empty({m, n}, res_tensor_options);

  cusparseLtMatDescriptor_t res_descriptor;
  TORCH_CUDASPARSE_CHECK(cusparseLtDenseDescriptorInit(
      &handle,
      &res_descriptor,
      m,
      n,
      (transpose_result) ? m : n,
      16,
      output_type,
      (transpose_result) ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW));

  // For float8, need fp16 C_descriptor, can't use FP8 for this matrix
  cusparseLtMatDescriptor_t C_descriptor;
  TORCH_CUDASPARSE_CHECK(cusparseLtDenseDescriptorInit(
      &handle,
      &C_descriptor,
      m,
      n,
      (transpose_result) ? m : n,
      16,
      C_type,
      (transpose_result) ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW));

  // initialize matmul
  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescriptorInit(
      &handle,
      &matmul,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      (dense_B.is_contiguous()) ? CUSPARSE_OPERATION_NON_TRANSPOSE
                                : CUSPARSE_OPERATION_TRANSPOSE,
      sparse_input_descriptor_ptr,
      dense_input_descriptor_ptr,
      &C_descriptor,
      &res_descriptor,
      compute_type));

  // set bias pointer for matmul, need to assign to get location
  if (bias_opt.has_value()) {
    auto& bias = bias_opt.value();
    void* dBias = bias.data_ptr();
    TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescSetAttribute(
        &handle,
        &matmul,
        CUSPARSELT_MATMUL_BIAS_POINTER,
        &dBias,
        sizeof(dBias)));
  }

  // auto t1 = std::chrono::steady_clock::now();
  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgSelectionInit(
      &handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT));
  // auto t2 = std::chrono::steady_clock::now();
  // double duration_plan_init = std::chrono::duration<double, std::milli>(t2 - t1).count();
  // std::cout << "duration_plan_init = " << duration_plan_init << std::endl;

  // set matmul search params
  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgSetAttribute(
      &handle,
      &alg_sel,
      CUSPARSELT_MATMUL_ALG_CONFIG_ID,
      &alg_id,
      sizeof(alg_id)));

  cusparseLtSplitKMode_t splitKMode;
  int max_alg_id;
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

  // set tensor_alpha_mode and alpha pointer for matmul
  const auto alpha_tensor = alpha_opt.has_value() ? *alpha_opt : Tensor{};
  auto alpha_ptr = &alpha;
  if (alpha_opt.has_value()) {
    if (alpha_tensor.numel() == 1) {
      alpha = alpha_tensor.item<float>();
    } else {
      int tensor_alpha_mode = 1;
      TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescSetAttribute(
          &handle,
          &matmul,
          CUSPARSELT_MATMUL_ALPHA_VECTOR_SCALING,
          &tensor_alpha_mode,
          sizeof(tensor_alpha_mode)));
      alpha_ptr = static_cast<float*>(alpha_tensor.data_ptr());
    }
  }

  TORCH_CUDASPARSE_CHECK(
      cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel));

  size_t workspace_size;
  TORCH_CUDASPARSE_CHECK(
      cusparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size));


  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto workspacePtr = allocator.allocate(workspace_size);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (search_alg_id) {
    // run matmul search
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
        // jank because of the way we want this to be an array of streams
        &stream,
        1));

    // get matmul params used
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

  } else {
    // do normal matmul
    TORCH_CUDASPARSE_CHECK(cusparseLtMatmul(
        &handle,
        &plan,
        alpha_ptr,
        compressed_A.data_ptr(),
        dense_B.data_ptr(),
        &beta,
        res.data_ptr(),
        res.data_ptr(),
        workspacePtr.get(),
        // jank because of the way we want this to be an array of streams
        &stream,
        1));
  }

  // destroy descriptors
  // TORCH_CUDASPARSE_CHECK(
  //     cusparseLtMatDescriptorDestroy(&sparse_input_descriptor));
  // TORCH_CUDASPARSE_CHECK(
  //     cusparseLtMatDescriptorDestroy(&dense_input_descriptor));
  TORCH_CUDASPARSE_CHECK(cusparseLtMatDescriptorDestroy(&res_descriptor));
  // destroy plan
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

#else // No cuSPARSELt support, throw error if these functions are called.

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
