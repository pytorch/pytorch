#include <ATen/Config.h>
#include <ATen/core/DimVector.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/CuFFTUtils.h>
#include <ATen/native/utils/ParamsHash.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>

#include <cufft.h>
#include <cufftXt.h>

#include <limits>
#include <list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace at { namespace native { namespace detail {

// Enum representing the FFT type
enum class CuFFTTransformType : int8_t {
  C2C,  // Complex-to-complex
  R2C,  // Real-to-complex
  C2R,  // Complex-to-real
};

// This struct is used to let us easily compute hashes of the
// parameters.
// It will be the **key** to the plan cache.
struct CuFFTParams
{
  int64_t signal_ndim_; // between 1 and max_rank, i.e., 1 <= signal_ndim <= 3
  // These include additional batch dimension as well.
  int64_t sizes_[max_rank + 1];
  int64_t input_strides_[max_rank + 1];
  int64_t output_strides_[max_rank + 1];
  CuFFTTransformType fft_type_;
  ScalarType value_type_;

  CuFFTParams() = default;

  CuFFTParams(IntArrayRef in_strides, IntArrayRef out_strides,
      IntArrayRef signal_sizes, CuFFTTransformType fft_type, ScalarType value_type) {
    // Padding bits must be zeroed for hashing
    memset(this, 0, sizeof(*this));
    signal_ndim_ = signal_sizes.size() - 1;
    fft_type_ = fft_type;
    value_type_ = value_type;

    TORCH_INTERNAL_ASSERT(in_strides.size() == signal_sizes.size());
    TORCH_INTERNAL_ASSERT(out_strides.size() == signal_sizes.size());
    TORCH_INTERNAL_ASSERT(1 <= signal_ndim_ && signal_ndim_ <= max_rank);

    std::copy(signal_sizes.cbegin(), signal_sizes.cend(), sizes_);
    std::copy(in_strides.cbegin(), in_strides.cend(), input_strides_);
    std::copy(out_strides.cbegin(), out_strides.cend(), output_strides_);
  }
};

static_assert(std::is_trivial_v<CuFFTParams>, "");

// Returns true if the transform type has complex input
inline bool cufft_complex_input(CuFFTTransformType type) {
  switch (type) {
    case CuFFTTransformType::C2C:
    case CuFFTTransformType::C2R:
      return true;

    case CuFFTTransformType::R2C:
      return false;
  }
  TORCH_INTERNAL_ASSERT(false);
}

// Returns true if the transform type has complex output
inline bool cufft_complex_output(CuFFTTransformType type) {
  switch (type) {
    case CuFFTTransformType::C2C:
    case CuFFTTransformType::R2C:
      return true;

    case CuFFTTransformType::C2R:
      return false;
  }
  TORCH_INTERNAL_ASSERT(false);
}

// Create transform type enum from bools representing if input and output are complex
inline CuFFTTransformType GetCuFFTTransformType(bool complex_input, bool complex_output) {
  if (complex_input && complex_output) {
    return CuFFTTransformType::C2C;
  } else if (complex_input && !complex_output) {
    return CuFFTTransformType::C2R;
  } else if (!complex_input && complex_output) {
    return CuFFTTransformType::R2C;
  }
  TORCH_INTERNAL_ASSERT(false, "Real to real FFTs are not supported");
}


class CuFFTHandle {
  ::cufftHandle handle_;
public:

  CuFFTHandle() {
    CUFFT_CHECK(cufftCreate(&handle_));
  }

  ::cufftHandle & get() { return handle_; }
  const ::cufftHandle & get() const { return handle_; }

  ~CuFFTHandle() {
// Not using fftDestroy() for rocFFT to work around double freeing of handles
#if !defined(USE_ROCM)
    cufftDestroy(handle_);
#endif
  }
};

__forceinline__
static bool is_pow_of_two(int64_t x) {
  return (x & (x - 1)) == 0;
}

using cufft_size_type = long long int;

using CuFFTDimVector = c10::SmallVector<cufft_size_type, at::kDimVectorStaticSize>;

// Struct representing a tensor in CuFFT's data layout for planning transforms
// See NOTE [ cuFFT Embedded Strides ].
struct CuFFTDataLayout {
  CuFFTDimVector embed;
  cufft_size_type stride, dist;
  bool must_clone, simple;
};

// Returns a cufft embedding for a contiguous signal of the given size.
// e.g. if the input is cloned, this will be the resulting data layout
// See NOTE [ cuFFT Embedded Strides ].
inline CuFFTDataLayout cufft_simple_embed(IntArrayRef sizes, bool onesided) {
  CuFFTDataLayout layout;
  layout.simple = true;
  layout.must_clone = false;
  layout.embed.assign(sizes.cbegin() + 1, sizes.cend());
  if (onesided) {
    layout.embed.back() = sizes.back() / 2 + 1;
  }
  layout.stride = 1;
  layout.dist = 1;
  for (const auto& len : layout.embed) {
    layout.dist *= len;
  }
  return layout;
}

// Convert strides to a CuFFT embedded representation.
// If strides cannot be embedded, returns a simple layout and sets must_clone flag
// See NOTE [ cuFFT Embedded Strides ].
inline CuFFTDataLayout as_cufft_embed(IntArrayRef strides, IntArrayRef sizes, bool onesided) {
  const auto signal_ndim = strides.size() - 1;
  CuFFTDataLayout layout;
  auto last_stride = strides[signal_ndim];
  layout.must_clone = (last_stride <= 0);

  const auto last_dim_size = onesided ?
      sizes[signal_ndim] / 2 + 1 : sizes[signal_ndim];
  const auto signal_numel = c10::multiply_integers(sizes.slice(1, sizes.size() - 2)) * last_dim_size;

  // Zero stides are not allowed, even if the batch size is one.
  // If that happens just set a dummy case
  if (sizes[0] == 1) {
    layout.dist = signal_numel;
  } else if (strides[0] == 0) {
    layout.must_clone = true;
  } else {
    layout.dist = strides[0];
  }

  // Calculate the embedding shape, or set must_clone if the strides cannot be embedded
  layout.embed.resize(signal_ndim);
  for (auto i = signal_ndim - 1; !layout.must_clone && i > 0; i--) {
    auto stride = strides[i];
    if (sizes[i] == 1) {
      layout.embed[i] = 1;
    } else if (stride > 0 && stride % last_stride == 0) {
      layout.embed[i] = stride / last_stride;
      last_stride = stride;
    } else {
      layout.must_clone = true;
    }
  }

  if (layout.must_clone) {
    // If the input needs to be cloned, assume it will be contiguous
    layout = cufft_simple_embed(sizes, onesided);
    layout.must_clone = true;
  } else {
    layout.embed[0] = sizes[1];
    layout.stride = strides[signal_ndim];
    // Determine if layout represents a simple embedding (contiguous data)
    layout.simple = [&] {
      for (const auto i : c10::irange(1, signal_ndim - 1)) {
        if (layout.embed[i] != sizes[i + 1]) {
          return false;
        }
      }

      return (layout.stride == 1 && layout.dist == signal_numel &&
          layout.embed.back() == last_dim_size);
    }();
  }
  return layout;
}

// This class contains all the information needed to execute a cuFFT plan:
//   1. the plan
//   2. whether to clone input before executing the plan
//   3. the workspace size needed
//
// This class will be the **value** in the plan cache.
// It **owns** the raw plan via a unique_ptr.
class CuFFTConfig {
public:

  // Only move semantics is enought for this class. Although we already use
  // unique_ptr for the plan, still remove copy constructor and assignment op so
  // we don't accidentally copy and take perf hit.
  CuFFTConfig(const CuFFTConfig&) = delete;
  CuFFTConfig& operator=(CuFFTConfig const&) = delete;

  explicit CuFFTConfig(const CuFFTParams& params):
      CuFFTConfig(
          IntArrayRef(params.input_strides_, params.signal_ndim_ + 1),
          IntArrayRef(params.output_strides_, params.signal_ndim_ + 1),
          IntArrayRef(params.sizes_, params.signal_ndim_ + 1),
          params.fft_type_,
          params.value_type_) {}

  // For complex types, strides are in units of 2 * element_size(dtype)
  // sizes are for the full signal, including batch size and always two-sided
  CuFFTConfig(IntArrayRef in_strides, IntArrayRef out_strides,
      IntArrayRef sizes, CuFFTTransformType fft_type, ScalarType dtype):
        fft_type_(fft_type), value_type_(dtype) {

    // signal sizes (excluding batch dim)
    CuFFTDimVector signal_sizes(sizes.begin() + 1, sizes.end());

    // input batch size
    const int64_t batch = sizes[0];
    const int64_t signal_ndim = sizes.size() - 1;

    // Since cuFFT has limited non-unit stride support and various constraints, we
    // use a flag to keep track throughout this function to see if we need to
    // input = input.clone();

#if defined(USE_ROCM)
    // clone input to avoid issues with hipfft clobering the input and failing tests
    clone_input = true;
#else
    clone_input = false;
#endif

    // For half, base strides on the real part of real-to-complex and
    // complex-to-real transforms are not supported. Since our output is always
    // contiguous, only need to check real-to-complex case.
    if (dtype == ScalarType::Half) {
      // cuFFT on half requires compute capability of at least SM_53
      auto dev_prop = at::cuda::getCurrentDeviceProperties();
      TORCH_CHECK(dev_prop->major >= 5 && !(dev_prop->major == 5 && dev_prop->minor < 3),
               "cuFFT doesn't support signals of half type with compute "
               "capability less than SM_53, but the device containing input half "
               "tensor only has SM_", dev_prop->major, dev_prop->minor);
      for (const auto i : c10::irange(signal_ndim)) {
        TORCH_CHECK(is_pow_of_two(sizes[i + 1]),
            "cuFFT only supports dimensions whose sizes are powers of two when"
            " computing in half precision, but got a signal size of",
            sizes.slice(1));
      }
      clone_input |= in_strides.back() != 1;
    }

    CuFFTDataLayout in_layout;
    if (clone_input) {
      in_layout = cufft_simple_embed(sizes, fft_type == CuFFTTransformType::C2R);
    } else {
      in_layout = as_cufft_embed(in_strides, sizes, fft_type == CuFFTTransformType::C2R);
    }
    auto out_layout = as_cufft_embed(out_strides, sizes, fft_type == CuFFTTransformType::R2C);
    TORCH_INTERNAL_ASSERT(!out_layout.must_clone, "Out strides cannot be represented as CuFFT embedding");
    clone_input |= in_layout.must_clone;

    // Check if we can take advantage of simple data layout.
    //
    // See NOTE [ cuFFT Embedded Strides ] in native/cuda/SpectralOps.cu.

    const bool simple_layout = in_layout.simple && out_layout.simple;
    cudaDataType itype, otype, exec_type;
    const auto complex_input = cufft_complex_input(fft_type);
    const auto complex_output = cufft_complex_output(fft_type);
    if (dtype == ScalarType::Float) {
      itype = complex_input ? CUDA_C_32F : CUDA_R_32F;
      otype = complex_output ? CUDA_C_32F : CUDA_R_32F;
      exec_type = CUDA_C_32F;
    } else if (dtype == ScalarType::Double) {
      itype = complex_input ? CUDA_C_64F : CUDA_R_64F;
      otype = complex_output ? CUDA_C_64F : CUDA_R_64F;
      exec_type = CUDA_C_64F;
    } else if (dtype == ScalarType::Half) {
      itype = complex_input ? CUDA_C_16F : CUDA_R_16F;
      otype = complex_output ? CUDA_C_16F : CUDA_R_16F;
      exec_type = CUDA_C_16F;
    } else {
      TORCH_CHECK(false, "cuFFT doesn't support tensor of type: ", dtype);
    }

    // disable auto allocation of workspace to use THC allocator
    CUFFT_CHECK(cufftSetAutoAllocation(plan(), /* autoAllocate */ 0));

    size_t ws_size_t;

    // make plan
    if (simple_layout) {
      // If with unit-stride, we tell cuFFT by setting inembed == onembed == NULL.
      // In such case, cuFFT ignores istride, ostride, idist, and odist
      // by assuming istride = ostride = 1.
      //
      // See NOTE [ cuFFT Embedded Strides ] in native/cuda/SpectralOps.cu.
      CUFFT_CHECK(cufftXtMakePlanMany(plan(), signal_ndim, signal_sizes.data(),
        /* inembed */ nullptr, /* base_istride */ 1, /* idist */ 1, itype,
        /* onembed */ nullptr, /* base_ostride */ 1, /* odist */ 1, otype,
        batch, &ws_size_t, exec_type));
    } else {
      CUFFT_CHECK(cufftXtMakePlanMany(plan(), signal_ndim, signal_sizes.data(),
            in_layout.embed.data(), in_layout.stride, in_layout.dist, itype,
            out_layout.embed.data(), out_layout.stride, out_layout.dist, otype,
            batch, &ws_size_t, exec_type));
    }
    ws_size = static_cast<int64_t>(ws_size_t);
  }

  const cufftHandle &plan() const { return plan_ptr.get(); }

  CuFFTTransformType transform_type() const { return fft_type_; }
  ScalarType data_type() const { return value_type_; }
  bool should_clone_input() const { return clone_input; }
  int64_t workspace_size() const { return ws_size; }

private:
  CuFFTHandle plan_ptr;
  bool clone_input;
  int64_t ws_size;
  CuFFTTransformType fft_type_;
  ScalarType value_type_;
};

#if defined(USE_ROCM)
  // Note that the max plan number for CUDA version < 10 has to be 1023
  // due to a bug that fails on the 1024th plan
  constexpr int64_t CUFFT_MAX_PLAN_NUM = 1023;
  constexpr int64_t CUFFT_DEFAULT_CACHE_SIZE = CUFFT_MAX_PLAN_NUM;
#else
  constexpr int64_t CUFFT_MAX_PLAN_NUM = std::numeric_limits<int64_t>::max();
  // The default max cache size chosen for CUDA version > 10 is arbitrary.
  // This number puts a limit on how big of a plan cache should we maintain by
  // default. Users can always configure it via cufft_set_plan_cache_max_size.
  constexpr int64_t CUFFT_DEFAULT_CACHE_SIZE = 4096;
#endif
static_assert(0 <= CUFFT_MAX_PLAN_NUM && CUFFT_MAX_PLAN_NUM <= std::numeric_limits<int64_t>::max(),
              "CUFFT_MAX_PLAN_NUM not in size_t range");
static_assert(CUFFT_DEFAULT_CACHE_SIZE >= 0 && CUFFT_DEFAULT_CACHE_SIZE <= CUFFT_MAX_PLAN_NUM,
              "CUFFT_DEFAULT_CACHE_SIZE not in [0, CUFFT_MAX_PLAN_NUM] range");

// This cache assumes that the mapping from key to value never changes.
// This is **NOT** thread-safe. Please use a mutex when using it **AND** the
// value returned from try_emplace_value.
// The contract of using this cache is that try_emplace_value should only be
// used when the max_size is positive.
class CuFFTParamsLRUCache {
public:
  using kv_t = typename std::pair<CuFFTParams, CuFFTConfig>;
  using map_t = typename std::unordered_map<std::reference_wrapper<CuFFTParams>,
                                            typename std::list<kv_t>::iterator,
                                            ParamsHash<CuFFTParams>,
                                            ParamsEqual<CuFFTParams>>;
  using map_kkv_iter_t = typename map_t::iterator;


  CuFFTParamsLRUCache() : CuFFTParamsLRUCache(CUFFT_DEFAULT_CACHE_SIZE) {}

  CuFFTParamsLRUCache(int64_t max_size) {
    _set_max_size(max_size);
  }

  CuFFTParamsLRUCache(CuFFTParamsLRUCache&& other) noexcept :
    _usage_list(std::move(other._usage_list)),
    _cache_map(std::move(other._cache_map)),
    _max_size(other._max_size) {}

  CuFFTParamsLRUCache& operator=(CuFFTParamsLRUCache&& other) noexcept {
    _usage_list = std::move(other._usage_list);
    _cache_map = std::move(other._cache_map);
    _max_size = other._max_size;
    return *this;
  }

  // If key is in this cache, return the cached config. Otherwise, emplace the
  // config in this cache and return it.
  // Return const reference because CuFFTConfig shouldn't be tampered with once
  // created.
  const CuFFTConfig &lookup(CuFFTParams params) {
    AT_ASSERT(_max_size > 0);

    map_kkv_iter_t map_it = _cache_map.find(params);
    // Hit, put to list front
    if (map_it != _cache_map.end()) {
      _usage_list.splice(_usage_list.begin(), _usage_list, map_it->second);
      return map_it->second->second;
    }

    // Miss
    // remove if needed
    if (_usage_list.size() >= _max_size) {
      auto last = _usage_list.end();
      last--;
      _cache_map.erase(last->first);
      _usage_list.pop_back();
    }

    // construct new plan at list front, then insert into _cache_map
    _usage_list.emplace_front(std::piecewise_construct,
                       std::forward_as_tuple(params),
                       std::forward_as_tuple(params));
    auto kv_it = _usage_list.begin();
    _cache_map.emplace(std::piecewise_construct,
                std::forward_as_tuple(kv_it->first),
                std::forward_as_tuple(kv_it));
    return kv_it->second;
  }

  void clear() {
    _cache_map.clear();
    _usage_list.clear();
  }

  void resize(int64_t new_size) {
    _set_max_size(new_size);
    auto cur_size = _usage_list.size();
    if (cur_size > _max_size) {
      auto delete_it = _usage_list.end();
      for (size_t i = 0; i < cur_size - _max_size; i++) {
        delete_it--;
        _cache_map.erase(delete_it->first);
      }
      _usage_list.erase(delete_it, _usage_list.end());
    }
  }

  size_t size() const { return _cache_map.size(); }

  size_t max_size() const noexcept { return _max_size; }

  std::mutex mutex;

private:
  // Only sets size and does value check. Does not resize the data structures.
  void _set_max_size(int64_t new_size) {
    // We check that 0 <= new_size <= CUFFT_MAX_PLAN_NUM here. Since
    // CUFFT_MAX_PLAN_NUM is of type size_t, we need to do non-negativity check
    // first.
    TORCH_CHECK(new_size >= 0,
             "cuFFT plan cache size must be non-negative, but got ", new_size);
    TORCH_CHECK(new_size <= CUFFT_MAX_PLAN_NUM,
             "cuFFT plan cache size can not be larger than ", CUFFT_MAX_PLAN_NUM, ", but got ", new_size);
    _max_size = static_cast<size_t>(new_size);
  }

  std::list<kv_t> _usage_list;
  map_t _cache_map;
  size_t _max_size;
};

// Since ATen is separated into CPU build and CUDA build, we need a way to call
// these functions only when CUDA is loaded. We use CUDA hooks for this purpose
// (at cuda/detail/CUDAHooks.cpp), and call the hooked functions from the actual
// native function counterparts (at native/SpectralOps.cpp), i.e.,
// _cufft_get_plan_cache_max_size, _cufft_set_plan_cache_max_size
// _cufft_get_plan_cache_size, and _cufft_clear_plan_cache.
int64_t cufft_get_plan_cache_max_size_impl(DeviceIndex device_index);
void cufft_set_plan_cache_max_size_impl(DeviceIndex device_index, int64_t max_size);
int64_t cufft_get_plan_cache_size_impl(DeviceIndex device_index);
void cufft_clear_plan_cache_impl(DeviceIndex device_index);

}}} // namespace at::native::detail
