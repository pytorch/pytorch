#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Config.h>
#include <ATen/native/cuda/CuFFTUtils.h>
#include <ATen/native/utils/ParamsHash.h>

#include <list>
#include <unordered_map>
#include <string>
#include <stdexcept>
#include <sstream>
#include <limits>
#include <cufft.h>
#include <cufftXt.h>

namespace at { namespace native { namespace detail {

// This POD struct is used to let us easily compute hashes of the
// parameters.
// It will be the **key** to the plan cache.
struct CuFFTParams
{
  at::ScalarType scalar_type_;
  int64_t input_sizes_[max_rank + 2];
  int64_t input_strides_[max_rank + 2];
  uint8_t signal_ndim_;  // between 1 and max_rank, i.e., 1 <= signal_ndim <= 3
  bool complex_input_;
  bool complex_output_;
  int64_t signal_sizes_[max_rank];
  bool onesided_;
};

// NB: This can't be a constructor, because then CuFFTParams
// would not be a POD anymore.
static inline void setCuFFTParams(CuFFTParams* params,
    const Tensor& input, int64_t signal_ndim, bool complex_input,
    bool complex_output, IntArrayRef checked_signal_sizes, bool onesided) {

  memset(params, 0, sizeof(CuFFTParams));
  params->scalar_type_ = input.type().scalarType();
  for (int i = 0; i != input.dim(); ++i) {
    params->input_sizes_[i] = input.size(i);
    if (input.size(i) != 1) {
      params->input_strides_[i] = input.stride(i);
    }
  }
  params->signal_ndim_ = (uint8_t) signal_ndim;
  params->complex_input_ = complex_input;
  params->complex_output_ = complex_output;
  for (size_t i = 0; i != checked_signal_sizes.size(); ++i) {
    params->signal_sizes_[i] = checked_signal_sizes[i];
  }
  params->onesided_ = onesided;
}

struct CuFFTHandleDeleter {
  void operator()(cufftHandle* x) {
    if (x != nullptr) {
      CUFFT_CHECK(cufftDestroy(*x));
    }
  }
};

__forceinline__
static bool is_pow_of_two(int64_t x) {
  return (x & (x - 1)) == 0;
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

  explicit CuFFTConfig(Tensor& input, int64_t signal_ndim, bool complex_input,
    bool complex_output, IntArrayRef checked_signal_sizes, bool onesided,
    IntArrayRef output_sizes) {

    // signal sizes
#ifdef __HIP_PLATFORM_HCC__
    std::vector<int> signal_sizes(checked_signal_sizes.begin(),
                                  checked_signal_sizes.end());
#else
    std::vector<long long int> signal_sizes(checked_signal_sizes.begin(),
                                            checked_signal_sizes.end());
#endif

    // input batch size
    long long int batch = input.size(0);

    // Since cuFFT has limited non-unit stride support and various constraints, we
    // use a flag to keep track throughout this function to see if we need to
    // input = input.clone();
    clone_input = false;

    // For half, base strides on the real part of real-to-complex and
    // complex-to-real transforms are not supported. Since our output is always
    // contiguous, only need to check real-to-complex case.
    if (input.type().scalarType() == ScalarType::Half) {
      // cuFFT on half requires compute capability of at least SM_53
      auto dev_prop = at::cuda::getCurrentDeviceProperties();
      AT_CHECK(dev_prop->major >= 5 && !(dev_prop->major == 5 && dev_prop->minor < 3),
               "cuFFT doesn't support signals of half type with compute "
               "capability less than SM_53, but the device containing input half "
               "tensor only has SM_", dev_prop->major, dev_prop->minor);
      for (int64_t i = 0; i < signal_ndim; i++) {
        auto signal_size = checked_signal_sizes[i];
        AT_CHECK(is_pow_of_two(signal_size),
                 "cuFFT doesn't support signals of half type with size at any ",
                 "dimension that is not a power of two, but got a signal size of ",
                 checked_signal_sizes);
      }
      clone_input |= input.stride(signal_ndim) != 1;
    }

    // check the input sizes and strides to see if we need to make it contiguous
    // cuFFT doesn't support batch dim with stride 0
    clone_input |= input.stride(0) == 0;

    if (complex_input) {
      // Real/imag dimension must be like complex type.
      clone_input |= input.stride(-1) != 1;
      // Strides of other dimensions needs to be aligned when viewed as of complex
      // type, i.e., multiples of 2. We check the batch dim and last signal dim
      // here. If the input can be viewed as having embedded strides, the other
      // signal dims will also satisfy this.
      // See NOTE [ cuFFT Embedded Strides ] in native/cuda/SpectralOps.cu.
      clone_input |= (batch > 0 && input.stride(0) % 2 != 0) ||
                      input.stride(signal_ndim) % 2 != 0;
    }

    // Checks if input strides can be viewed as embedded.
    // See NOTE [ cuFFT Embedded Strides ].
    //
    // TODO: Figure out why windows fails to compile
    //         c10::optional<std::vector<long long int>> inembed_opt =
    //         c10::nullopt;
    //       Then move the following to a helper function.
#ifdef __HIP_PLATFORM_HCC__
    std::vector<int> inembed(signal_ndim);
#else
    std::vector<long long int> inembed(signal_ndim);
#endif
    if (!clone_input) {
      auto istrides = input.strides();
      auto last_istride = istrides[signal_ndim];
      clone_input = last_istride <= 0;
      for (auto i = signal_ndim - 1; !clone_input && i > 0 /* inembed[0] doesn't matteer */; i--) {
        auto istride = istrides[i];
        if (istride > 0 && istride % last_istride == 0) {
          inembed[i] = istride / last_istride;
          last_istride = istride;
        } else {
          clone_input = true;
        }
      }
    }

    // Check if we can take advantage of simple data layout.
    //
    // Note that this is before the actual cloning. This is intentional so we can
    // check for advanced data layout with complex-to-real transform. cuFFT
    // out-of-place complex-to-real transforms with advanced layout may overwrite
    // input, and we need to clone the input.
    //
    // This just needs contiguity in cases except for twosided real-to-complex
    // transform where we won't have simple data layout as output is two sided.
    //
    // See NOTE [ cuFFT Embedded Strides ] in native/cuda/SpectralOps.cu.

    bool simple_layout = !(!complex_input && complex_output && !onesided) &&  // not twosided R2C
                         (clone_input || input.is_contiguous());              // contiguous
    if (!simple_layout && complex_input && !complex_output) {
      clone_input = true;
      simple_layout = true;
    }

    // if input should be cloned but simple layout can't be used (e.g. twosided R2C)
    if (clone_input && !simple_layout) {
      auto input_size = input.sizes();
      std::copy(input_size.begin() + 1,                // begin of signal dim in input
                input_size.begin() + signal_ndim + 1,  // end of signal dim in input
                inembed.begin());                      // begin of output
    }

#ifdef __HIP_PLATFORM_HCC__

    hipfftType exec_type;
    if (input.type().scalarType() == ScalarType::Float) {
      if (complex_input && complex_output) {
        exec_type = HIPFFT_C2C;
      } else if (complex_input && !complex_output) {
        exec_type = HIPFFT_C2R;
      } else if (!complex_input && complex_output) {
        exec_type = HIPFFT_R2C;
      } else {
        AT_ERROR("hipFFT doesn't support r2r (float)");
      }
    } else if (input.type().scalarType() == ScalarType::Double) {
      if (complex_input && complex_output) {
        exec_type = HIPFFT_Z2Z;
      } else if (complex_input && !complex_output) {
        exec_type = HIPFFT_Z2D;
      } else if (!complex_input && complex_output) {
        exec_type = HIPFFT_D2Z;
      } else {
        AT_ERROR("hipFFT doesn't support r2r (double)");
      }
    } else {
      std::ostringstream ss;
      ss << "hipFFT doesn't support tensor of type: "
         << toString(input.type().scalarType());
      AT_ERROR(ss.str());
    }

#else
    cudaDataType itype, otype, exec_type;
    if (input.type().scalarType() == ScalarType::Float) {
      itype = complex_input ? CUDA_C_32F : CUDA_R_32F;
      otype = complex_output ? CUDA_C_32F : CUDA_R_32F;
      exec_type = CUDA_C_32F;
    } else if (input.type().scalarType() == ScalarType::Double) {
      itype = complex_input ? CUDA_C_64F : CUDA_R_64F;
      otype = complex_output ? CUDA_C_64F : CUDA_R_64F;
      exec_type = CUDA_C_64F;
    } else if (input.type().scalarType() == ScalarType::Half) {
      itype = complex_input ? CUDA_C_16F : CUDA_R_16F;
      otype = complex_output ? CUDA_C_16F : CUDA_R_16F;
      exec_type = CUDA_C_16F;
    } else {
      std::ostringstream ss;
      ss << "cuFFT doesn't support tensor of type: "
         << toString(input.type().scalarType());
      AT_ERROR(ss.str());
    }
#endif

    // create plan
    auto raw_plan_ptr = new cufftHandle();
    CUFFT_CHECK(cufftCreate(raw_plan_ptr));
    plan_ptr.reset(raw_plan_ptr);

    // disable auto allocation of workspace to use THC allocator
    CUFFT_CHECK(cufftSetAutoAllocation(plan(), /* autoAllocate */ 0));

    size_t ws_size_t;

    // make plan
    if (simple_layout) {
      // If with unit-stride, we tell cuFFT by setting inembed == onembed == NULL.
      // In such case, cuFFT ignores base_istride, base_ostride, idist, and odist
      // by assuming base_istride = base_ostride = 1.
      //
      // See NOTE [ cuFFT Embedded Strides ] in native/cuda/SpectralOps.cu.
#ifdef __HIP_PLATFORM_HCC__
      CUFFT_CHECK(hipfftMakePlanMany(plan(), signal_ndim, signal_sizes.data(),
        /* inembed */ nullptr, /* base_istride */ 1, /* idist */ 1,
        /* onembed */ nullptr, /* base_ostride */ 1, /* odist */ 1,
	exec_type, batch, &ws_size_t));
#else
      CUFFT_CHECK(cufftXtMakePlanMany(plan(), signal_ndim, signal_sizes.data(),
        /* inembed */ nullptr, /* base_istride */ 1, /* idist */ 1, itype,
        /* onembed */ nullptr, /* base_ostride */ 1, /* odist */ 1, otype,
        batch, &ws_size_t, exec_type));
#endif
    } else {
      // set idist (stride at batch dim)
      // set base_istride (stride at innermost dim of signal)
      long long int idist, base_istride;
      if (clone_input) {
        idist = at::prod_intlist(input.sizes().slice(1, signal_ndim));
        base_istride = 1;
      } else if (complex_input) {
        idist = input.stride(0) >> 1;
        base_istride = input.stride(signal_ndim) >> 1;
      } else {
        idist = input.stride(0);
        base_istride = input.stride(signal_ndim);
      }
      // Even if batch dimension is one and idist (stride(0)) doesn't matter,
      // cuFFT errors if idist = 0. This is hack to make it succeed.
      if (idist == 0 && batch == 1) {
        idist = 1;
      }

      // set odist, onembed, base_ostride
#ifdef __HIP_PLATFORM_HCC__
      int odist = at::prod_intlist(output_sizes.slice(1, signal_ndim));
      std::vector<int> onembed(output_sizes.data() + 1, output_sizes.data() + signal_ndim + 1);
      int base_ostride = 1;

      int istride = base_istride;
      int iidist = idist;
      CUFFT_CHECK(hipfftMakePlanMany(plan(), signal_ndim, signal_sizes.data(),
        inembed.data(), istride, iidist,
        onembed.data(), base_ostride, odist,
        exec_type, batch, &ws_size_t));
#else
      long long int odist = at::prod_intlist(output_sizes.slice(1, signal_ndim));
      std::vector<long long int> onembed(output_sizes.data() + 1, output_sizes.data() + signal_ndim + 1);
      long long int base_ostride = 1;

      CUFFT_CHECK(cufftXtMakePlanMany(plan(), signal_ndim, signal_sizes.data(),
            inembed.data(), base_istride, idist, itype,
            onembed.data(), base_ostride, odist, otype,
            batch, &ws_size_t, exec_type));
#endif
      }
    ws_size = static_cast<int64_t>(ws_size_t);
  }

#ifdef __HIP_PLATFORM_HCC__
  cufftHandle &plan() const { return *plan_ptr.get(); }
#else
  const cufftHandle &plan() const { return *plan_ptr.get(); }
#endif

  bool should_clone_input() const { return clone_input; }

  int64_t workspace_size() const { return ws_size; }

private:
  std::unique_ptr<cufftHandle, CuFFTHandleDeleter> plan_ptr;
  bool clone_input;
  int64_t ws_size;
};

#if CUDA_VERSION < 10000
  // Note that the max plan number for CUDA version < 10 has to be 1023
  // due to a bug that fails on the 1024th plan
  constexpr size_t CUFFT_MAX_PLAN_NUM = 1023;
  constexpr size_t CUFFT_DEFAULT_CACHE_SIZE = CUFFT_MAX_PLAN_NUM;
#else
  constexpr size_t CUFFT_MAX_PLAN_NUM = std::numeric_limits<size_t>::max();
  // The default max cache size chosen for CUDA version > 10 is arbitrary.
  // This number puts a limit on how big of a plan cache should we maintain by
  // default. Users can always configure it via cufft_set_plan_cache_max_size.
  constexpr size_t CUFFT_DEFAULT_CACHE_SIZE = 4096;
#endif
static_assert(CUFFT_MAX_PLAN_NUM >= 0 && CUFFT_MAX_PLAN_NUM <= std::numeric_limits<size_t>::max(),
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

  // If key is in this cache, return the cached config. Otherwise, emplace the
  // config in this cache using value_args and return it.
  // Return const reference because CuFFTConfig shouldn't be tampered with once
  // created.
  // This is similar to c++ 17 try_emplace.
  template<typename K, class ...VArgs>
  const CuFFTConfig &try_emplace_value(K&& key, VArgs&&... value_args) {
    AT_ASSERT(_max_size > 0);

    map_kkv_iter_t map_it = _cache_map.find(key);
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
                       std::forward_as_tuple(key),
                       std::forward_as_tuple(value_args...));
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

private:
  // Only sets size and does value check. Does not resize the data structures.
  void _set_max_size(int64_t new_size) {
    // We check that 0 <= new_size <= CUFFT_MAX_PLAN_NUM here. Since
    // CUFFT_MAX_PLAN_NUM is of type size_t, we need to do non-negativity check
    // first.
    AT_CHECK(new_size >= 0,
             "cuFFT plan cache size must be non-negative, but got ", new_size);
    AT_CHECK(new_size <= CUFFT_MAX_PLAN_NUM,
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
int64_t cufft_get_plan_cache_max_size_impl();
void cufft_set_plan_cache_max_size_impl(int64_t max_size);
int64_t cufft_get_plan_cache_size_impl();
void cufft_clear_plan_cache_impl();

}}} // namespace at::native::detail
