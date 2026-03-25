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

// One cache key for the full matmul descriptor bundle and plan.
struct MatDescriptorCacheKey {
  int m = 0;
  int k = 0;
  int n = 0;
  bool dense_is_contiguous = true;
  bool transpose_result = false;
  int32_t input_type = 0;
  int32_t output_type = 0;
  int32_t C_type = 0;
  int32_t compute_type = 0;
  bool has_bias = false;
  bool has_alpha = false;
  int32_t alg_id = 0;
  int32_t split_k = 1;
  int32_t split_k_mode = -1;
};

using MatDescriptorCacheKeyHash = ParamsHash<MatDescriptorCacheKey>;
using MatDescriptorCacheKeyEqual = ParamsEqual<MatDescriptorCacheKey>;

struct MatDescriptorPtrs {
  cusparseLtMatmulDescriptor_t* matmul = nullptr;
  cusparseLtMatmulAlgSelection_t* alg_sel = nullptr;
  cusparseLtMatmulPlan_t* plan = nullptr;
};

// Cached entry holds matmul, alg_sel, plan (exposed via ptrs) plus the operand
// descriptors (sparse_input, dense_input, res, C) which must stay alive because
// the library retains pointers to them; they are not exposed in MatDescriptorPtrs.
struct CachedMatDescriptor {
  cusparseLtMatDescriptor_t sparse_input;
  cusparseLtMatDescriptor_t dense_input;
  cusparseLtMatDescriptor_t res;
  cusparseLtMatDescriptor_t C;
  cusparseLtMatmulDescriptor_t matmul;
  cusparseLtMatmulAlgSelection_t alg_sel;
  cusparseLtMatmulPlan_t plan;
  MatDescriptorPtrs ptrs;
  bool has_descriptors = false;

  CachedMatDescriptor() = default;
  ~CachedMatDescriptor() {
    if (has_descriptors) {
      cusparseLtMatDescriptorDestroy(&sparse_input);
      cusparseLtMatDescriptorDestroy(&dense_input);
      cusparseLtMatDescriptorDestroy(&res);
      cusparseLtMatDescriptorDestroy(&C);
      #if defined(CUSPARSELT_VERSION) && CUSPARSELT_VERSION >= 800
      cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
      #endif
      cusparseLtMatmulPlanDestroy(&plan);
      has_descriptors = false;
    }
  }
  CachedMatDescriptor(CachedMatDescriptor&& other) noexcept
      : sparse_input(other.sparse_input),
        dense_input(other.dense_input),
        res(other.res),
        C(other.C),
        matmul(other.matmul),
        alg_sel(other.alg_sel),
        plan(other.plan),
        ptrs(other.ptrs),
        has_descriptors(other.has_descriptors) {
    other.has_descriptors = false;
  }
  CachedMatDescriptor& operator=(CachedMatDescriptor&& other) noexcept {
    if (this != &other) {
      if (has_descriptors) {
        cusparseLtMatDescriptorDestroy(&sparse_input);
        cusparseLtMatDescriptorDestroy(&dense_input);
        cusparseLtMatDescriptorDestroy(&res);
        cusparseLtMatDescriptorDestroy(&C);
        #if defined(CUSPARSELT_VERSION) && CUSPARSELT_VERSION >= 800
        cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
        #endif
        cusparseLtMatmulPlanDestroy(&plan);
      }
      sparse_input = other.sparse_input;
      dense_input = other.dense_input;
      res = other.res;
      C = other.C;
      matmul = other.matmul;
      alg_sel = other.alg_sel;
      plan = other.plan;
      ptrs = other.ptrs;
      has_descriptors = other.has_descriptors;
      other.has_descriptors = false;
    }
    return *this;
  }
  CachedMatDescriptor(const CachedMatDescriptor&) = delete;
  CachedMatDescriptor& operator=(const CachedMatDescriptor&) = delete;
};

constexpr size_t MatDescriptorCacheMaxSize = 128;

using MatDescriptorLruList =
    std::list<std::pair<MatDescriptorCacheKey, CachedMatDescriptor>>;
using MatDescriptorCacheMap = std::unordered_map<
    MatDescriptorCacheKey,
    MatDescriptorLruList::iterator,
    MatDescriptorCacheKeyHash,
    MatDescriptorCacheKeyEqual>;

// Cache for matmul descriptor, alg selection, and plan (keyed by shape/dtype/alg).
struct MatDescriptorCache {
  MatDescriptorLruList lru_list;
  MatDescriptorCacheMap cache_map;

  MatDescriptorPtrs* lookup(const MatDescriptorCacheKey& key) {
    auto it = cache_map.find(key);
    if (it == cache_map.end()) {
      return nullptr;
    }
    lru_list.splice(lru_list.begin(), lru_list, it->second);
    CachedMatDescriptor& entry = it->second->second;
    entry.ptrs.matmul = &entry.matmul;
    entry.ptrs.alg_sel = &entry.alg_sel;
    entry.ptrs.plan = &entry.plan;
    return &entry.ptrs;
  }

  MatDescriptorPtrs* getOrCreate(
      cusparseLtHandle_t& handle,
      int64_t m,
      int64_t k,
      int64_t n,
      bool dense_is_contiguous,
      bool transpose_result,
      cudaDataType input_type,
      cudaDataType output_type,
      cudaDataType C_type,
      cusparseComputeType compute_type,
      int alg_id,
      int split_k,
      int split_k_mode,
      const std::optional<Tensor>& bias_opt,
      const std::optional<Tensor>& alpha_opt) {
    MatDescriptorCacheKey key{};
    key.m = static_cast<int>(m);
    key.k = static_cast<int>(k);
    key.n = static_cast<int>(n);
    key.dense_is_contiguous = dense_is_contiguous;
    key.transpose_result = transpose_result;
    key.input_type = static_cast<int32_t>(input_type);
    key.output_type = static_cast<int32_t>(output_type);
    key.C_type = static_cast<int32_t>(C_type);
    key.compute_type = static_cast<int32_t>(compute_type);
    key.has_bias = bias_opt.has_value();
    key.has_alpha = alpha_opt.has_value() && alpha_opt.value().numel() != 1;
    key.alg_id = static_cast<int32_t>(alg_id);
    key.split_k = static_cast<int32_t>(split_k);
    key.split_k_mode = static_cast<int32_t>(split_k_mode);
    MatDescriptorPtrs* result = lookup(key);

    if (result == nullptr) {
      if (lru_list.size() >= MatDescriptorCacheMaxSize) {
        auto& back = lru_list.back();
        cache_map.erase(back.first);
        lru_list.pop_back();
      }
      lru_list.push_front({key, CachedMatDescriptor{}});
      CachedMatDescriptor& entry = lru_list.front().second;
      int64_t dense_rows = dense_is_contiguous ? k : n;
      int64_t dense_cols = dense_is_contiguous ? n : k;
      int64_t res_ld = transpose_result ? m : n;
      cusparseOrder_t res_order =
          transpose_result ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW;

      TORCH_CUDASPARSE_CHECK(cusparseLtStructuredDescriptorInit(
          &handle,
          &entry.sparse_input,
          m,
          k,
          k,
          16,
          input_type,
          CUSPARSE_ORDER_ROW,
          CUSPARSELT_SPARSITY_50_PERCENT));
      TORCH_CUDASPARSE_CHECK(cusparseLtDenseDescriptorInit(
          &handle,
          &entry.dense_input,
          dense_rows,
          dense_cols,
          dense_cols,
          16,
          input_type,
          CUSPARSE_ORDER_ROW));
      TORCH_CUDASPARSE_CHECK(cusparseLtDenseDescriptorInit(
          &handle,
          &entry.res,
          m,
          n,
          res_ld,
          16,
          output_type,
          res_order));
      TORCH_CUDASPARSE_CHECK(cusparseLtDenseDescriptorInit(
          &handle,
          &entry.C,
          m,
          n,
          res_ld,
          16,
          C_type,
          res_order));
      TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescriptorInit(
          &handle,
          &entry.matmul,
          CUSPARSE_OPERATION_NON_TRANSPOSE,
          dense_is_contiguous ? CUSPARSE_OPERATION_NON_TRANSPOSE
                              : CUSPARSE_OPERATION_TRANSPOSE,
          &entry.sparse_input,
          &entry.dense_input,
          &entry.C,
          &entry.res,
          compute_type));
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
            &handle,
            &entry.alg_sel,
            CUSPARSELT_MATMUL_SPLIT_K,
            &split_k,
            sizeof(split_k)));
        if (split_k_mode > 0) {
          cusparseLtSplitKMode_t splitKMode =
              static_cast<cusparseLtSplitKMode_t>(split_k_mode);
          TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgSetAttribute(
              &handle,
              &entry.alg_sel,
              CUSPARSELT_MATMUL_SPLIT_K_MODE,
              &splitKMode,
              sizeof(splitKMode)));
        }
      }

      entry.ptrs.matmul = &entry.matmul;
      entry.ptrs.alg_sel = &entry.alg_sel;
      entry.ptrs.plan = &entry.plan;
      entry.has_descriptors = true;
      cache_map[lru_list.front().first] = lru_list.begin();
      result = &entry.ptrs;
    }

    if (bias_opt.has_value()) {
      auto& bias = bias_opt.value();
      void* dBias = bias.data_ptr();
      TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescSetAttribute(
          &handle,
          result->matmul,
          CUSPARSELT_MATMUL_BIAS_POINTER,
          &dBias,
          sizeof(dBias)));
    }
    if (alpha_opt.has_value() && alpha_opt.value().numel() != 1) {
      int tensor_alpha_mode = 1;
      TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescSetAttribute(
          &handle,
          result->matmul,
          CUSPARSELT_MATMUL_ALPHA_VECTOR_SCALING,
          &tensor_alpha_mode,
          sizeof(tensor_alpha_mode)));
    }

    TORCH_CUDASPARSE_CHECK(cusparseLtMatmulPlanInit(
        &handle, result->plan, result->matmul, result->alg_sel));
    return result;
  }

  void insert(MatDescriptorCacheKey key, CachedMatDescriptor descriptor) {
    if (lru_list.size() >= MatDescriptorCacheMaxSize) {
      auto& back = lru_list.back();
      cache_map.erase(back.first);
      lru_list.pop_back();
    }
    lru_list.push_front({key, std::move(descriptor)});
    cache_map[lru_list.front().first] = lru_list.begin();
  }
};

thread_local std::unique_ptr<MatDescriptorCache> mat_descriptor_cache;

MatDescriptorCache& getMatDescriptorCache() {
  if (!mat_descriptor_cache) {
    mat_descriptor_cache = std::make_unique<MatDescriptorCache>();
  }
  return *mat_descriptor_cache;
}

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

  // set up matmul descriptors and cache them or retrieve from cache.
  MatDescriptorPtrs* desc_ptrs = getMatDescriptorCache().getOrCreate(
      handle,
      m,
      k,
      n,
      dense_B.is_contiguous(),
      transpose_result,
      input_type,
      output_type,
      C_type,
      compute_type,
      alg_id,
      split_k,
      split_k_mode,
      bias_opt,
      alpha_opt);
  TORCH_CHECK(
      desc_ptrs != nullptr,
      "cuSPARSELt: mat descriptor cache getOrCreate failed");

  // set alpha pointer for matmul
  const auto alpha_tensor = alpha_opt.has_value() ? *alpha_opt : Tensor{};
  auto alpha_ptr = &alpha;
  if (alpha_opt.has_value()) {
    if (alpha_tensor.numel() == 1) {
      alpha = alpha_tensor.item<float>();
    } else {
      alpha_ptr = static_cast<float*>(alpha_tensor.data_ptr());
    }
  }

  size_t workspace_size;
  TORCH_CUDASPARSE_CHECK(
      cusparseLtMatmulGetWorkspace(&handle, desc_ptrs->plan, &workspace_size));

  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto workspacePtr = allocator.allocate(workspace_size);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // create result tensor
  auto res_tensor_options =
  c10::TensorOptions().dtype(out_dtype).device(dense_B.device());
  at::Tensor res = (transpose_result) ? at::empty({n, m}, res_tensor_options)
                                    : at::empty({m, n}, res_tensor_options);

  // these are not used unless search_alg_id == true.
  cusparseLtSplitKMode_t splitKMode = (cusparseLtSplitKMode_t)0;
  int max_alg_id = 0;

  if (search_alg_id) {
    // run matmul search
    TORCH_CUDASPARSE_CHECK(cusparseLtMatmulSearch(
        &handle,
        desc_ptrs->plan,
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
        desc_ptrs->alg_sel,
        CUSPARSELT_MATMUL_ALG_CONFIG_ID,
        &alg_id,
        sizeof(alg_id)));

    TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgGetAttribute(
        &handle,
        desc_ptrs->alg_sel,
        CUSPARSELT_MATMUL_SPLIT_K,
        &split_k,
        sizeof(split_k)));

    TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgGetAttribute(
        &handle,
        desc_ptrs->alg_sel,
        CUSPARSELT_MATMUL_SPLIT_K_MODE,
        &splitKMode,
        sizeof(splitKMode)));

    TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgGetAttribute(
        &handle,
        desc_ptrs->alg_sel,
        CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID,
        &max_alg_id,
        sizeof(max_alg_id)));

  } else {
    // do normal matmul
    TORCH_CUDASPARSE_CHECK(cusparseLtMatmul(
        &handle,
        desc_ptrs->plan,
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
