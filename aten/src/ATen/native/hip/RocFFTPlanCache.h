#pragma once

// Plan cache for the rocFFT path used by stft/istft. See RocFFTSpectralOps.cpp
// for why this exists alongside the hipFFT path in native/cuda/SpectralOps.cpp.
//
// This file is hand-written for ROCm and is not produced by hipify: everything
// under */hip/* is in the ignore list of tools/amd_build/build_amd.py.

#include <ATen/hip/HIPContext.h>
#include <ATen/native/hip/RocFFTCallbacks.h>
#include <ATen/native/utils/ParamsHash.h>
#include <c10/util/CallOnce.h>
#include <c10/util/Exception.h>
#include <c10/util/ScopeExit.h>

#include <rocfft/rocfft.h>

#include <cstring>
#include <limits>
#include <list>
#include <mutex>
#include <type_traits>
#include <unordered_map>

namespace at::native::detail {

inline const char* _rocfftGetErrorEnum(rocfft_status error) {
  switch (error) {
    case rocfft_status_success:
      return "rocfft_status_success";
    case rocfft_status_failure:
      return "rocfft_status_failure";
    case rocfft_status_invalid_arg_value:
      return "rocfft_status_invalid_arg_value";
    case rocfft_status_invalid_dimensions:
      return "rocfft_status_invalid_dimensions";
    case rocfft_status_invalid_array_type:
      return "rocfft_status_invalid_array_type";
    case rocfft_status_invalid_strides:
      return "rocfft_status_invalid_strides";
    case rocfft_status_invalid_distance:
      return "rocfft_status_invalid_distance";
    case rocfft_status_invalid_offset:
      return "rocfft_status_invalid_offset";
    case rocfft_status_invalid_work_buffer:
      return "rocfft_status_invalid_work_buffer";
    default:
      return "unrecognized rocfft_status";
  }
}

inline void ROCFFT_CHECK(rocfft_status error) {
  TORCH_CHECK(error == rocfft_status_success, "rocFFT error: ", _rocfftGetErrorEnum(error));
}

// rocfft_setup() has to run before any other rocFFT call. There is a matching
// rocfft_cleanup(), but like the rest of PyTorch we leave device libraries
// standing at exit rather than racing teardown against live tensors.
inline void lazy_init_rocfft() {
  static c10::once_flag flag;
  c10::call_once(flag, [] { ROCFFT_CHECK(rocfft_setup()); });
}

enum class RocFFTTransformType : int8_t {
  C2C,
  R2C,
  C2R,
};

// Key of the plan cache. stft/istft only ever issue a batched 1-D transform
// over the last dimension, so a single length plus one stride/distance pair per
// side fully describes the layout.
struct RocFFTParams {
  int64_t signal_size_;  // length of the full (two-sided, real for R2C/C2R) signal
  int64_t batch_;
  int64_t in_stride_, in_distance_;
  int64_t out_stride_, out_distance_;
  // Folded into the transform instead of run as a separate pass over the output.
  double scale_factor_;
  RocFFTTransformType fft_type_;
  // Unlike cuFFT, a rocFFT plan bakes in the transform direction, so it is part
  // of the key.
  bool forward_;
  ScalarType value_type_;
  // rocFFT compiles the callback into the plan's kernels, so it keys the cache
  // too.
  RocFFTCallbackKind callback_kind_;

  RocFFTParams() = default;

  RocFFTParams(int64_t signal_size, int64_t batch, int64_t in_stride, int64_t in_distance,
      int64_t out_stride, int64_t out_distance, RocFFTTransformType fft_type, bool forward,
      ScalarType value_type, RocFFTCallbackKind callback_kind = RocFFTCallbackKind::None,
      double scale_factor = 1.0) {
    // Padding bits must be zeroed for hashing
    std::memset(this, 0, sizeof(*this));
    signal_size_ = signal_size;
    batch_ = batch;
    in_stride_ = in_stride;
    in_distance_ = in_distance;
    out_stride_ = out_stride;
    out_distance_ = out_distance;
    scale_factor_ = scale_factor;
    fft_type_ = fft_type;
    forward_ = forward;
    value_type_ = value_type;
    callback_kind_ = callback_kind;
  }
};

static_assert(std::is_trivial_v<RocFFTParams>);

class RocFFTHandle {
  rocfft_plan plan_ = nullptr;

 public:
  RocFFTHandle() = default;
  RocFFTHandle(const RocFFTHandle&) = delete;
  RocFFTHandle& operator=(const RocFFTHandle&) = delete;

  rocfft_plan& get() { return plan_; }
  const rocfft_plan& get() const { return plan_; }

  ~RocFFTHandle() {
    if (plan_ != nullptr) {
      rocfft_plan_destroy(plan_);
    }
  }
};

// Owns a planned transform and the size of the work buffer it needs. Value type
// of the plan cache.
class RocFFTConfig {
 public:
  RocFFTConfig(const RocFFTConfig&) = delete;
  RocFFTConfig& operator=(RocFFTConfig const&) = delete;

  explicit RocFFTConfig(const RocFFTParams& params) {
    auto in_array_type = rocfft_array_type_complex_interleaved;
    auto out_array_type = rocfft_array_type_complex_interleaved;
    auto transform_type = rocfft_transform_type_complex_forward;
    switch (params.fft_type_) {
      case RocFFTTransformType::C2C:
        transform_type = params.forward_ ? rocfft_transform_type_complex_forward
                                         : rocfft_transform_type_complex_inverse;
        break;
      case RocFFTTransformType::R2C:
        in_array_type = rocfft_array_type_real;
        out_array_type = rocfft_array_type_hermitian_interleaved;
        transform_type = rocfft_transform_type_real_forward;
        break;
      case RocFFTTransformType::C2R:
        in_array_type = rocfft_array_type_hermitian_interleaved;
        out_array_type = rocfft_array_type_real;
        transform_type = rocfft_transform_type_real_inverse;
        break;
    }

    rocfft_precision precision = rocfft_precision_single;
    if (params.value_type_ == ScalarType::Double) {
      precision = rocfft_precision_double;
    } else {
      TORCH_CHECK(params.value_type_ == ScalarType::Float,
          "rocFFT doesn't support tensor of type: ", params.value_type_);
    }

    rocfft_plan_description desc = nullptr;
    ROCFFT_CHECK(rocfft_plan_description_create(&desc));
    auto desc_guard = c10::make_scope_exit([&] { rocfft_plan_description_destroy(desc); });

    const size_t in_strides[1] = {static_cast<size_t>(params.in_stride_)};
    const size_t out_strides[1] = {static_cast<size_t>(params.out_stride_)};
    ROCFFT_CHECK(rocfft_plan_description_set_data_layout(desc, in_array_type, out_array_type,
        /*in_offsets=*/nullptr, /*out_offsets=*/nullptr,
        /*in_strides_size=*/1, in_strides, static_cast<size_t>(params.in_distance_),
        /*out_strides_size=*/1, out_strides, static_cast<size_t>(params.out_distance_)));

    if (params.scale_factor_ != 1.0) {
      ROCFFT_CHECK(rocfft_plan_description_set_scale_factor(desc, params.scale_factor_));
    }

    if (params.callback_kind_ != RocFFTCallbackKind::None) {
      rocfft_register_callback(desc, params.callback_kind_);
    }

    const size_t length = static_cast<size_t>(params.signal_size_);
    ROCFFT_CHECK(rocfft_plan_create(&plan_.get(), rocfft_placement_notinplace, transform_type,
        precision, /*dimensions=*/1, &length,
        /*number_of_transforms=*/static_cast<size_t>(params.batch_), desc));
    ROCFFT_CHECK(rocfft_plan_get_work_buffer_size(plan_.get(), &ws_size_));
  }

  const rocfft_plan& plan() const { return plan_.get(); }
  size_t workspace_size() const { return ws_size_; }

 private:
  RocFFTHandle plan_;
  size_t ws_size_ = 0;
};

constexpr int64_t ROCFFT_DEFAULT_CACHE_SIZE = 4096;

// Mirrors CuFFTParamsLRUCache. Not thread-safe; callers must hold `mutex`.
class RocFFTParamsLRUCache {
 public:
  using kv_t = typename std::pair<RocFFTParams, RocFFTConfig>;
  using map_t = typename std::unordered_map<std::reference_wrapper<RocFFTParams>,
                                            typename std::list<kv_t>::iterator,
                                            ParamsHash<RocFFTParams>,
                                            ParamsEqual<RocFFTParams>>;

  RocFFTParamsLRUCache() : _max_size(ROCFFT_DEFAULT_CACHE_SIZE) {}

  const RocFFTConfig& lookup(RocFFTParams params) {
    AT_ASSERT(_max_size > 0);

    auto map_it = _cache_map.find(params);
    if (map_it != _cache_map.end()) {
      _usage_list.splice(_usage_list.begin(), _usage_list, map_it->second);
      return map_it->second->second;
    }

    if (_usage_list.size() >= _max_size) {
      auto last = _usage_list.end();
      last--;
      _cache_map.erase(last->first);
      _usage_list.pop_back();
    }

    _usage_list.emplace_front(std::piecewise_construct, std::forward_as_tuple(params),
                              std::forward_as_tuple(params));
    auto kv_it = _usage_list.begin();
    _cache_map.emplace(std::piecewise_construct, std::forward_as_tuple(kv_it->first),
                       std::forward_as_tuple(kv_it));
    return kv_it->second;
  }

  void clear() {
    _cache_map.clear();
    _usage_list.clear();
  }

  std::mutex mutex;

 private:
  std::list<kv_t> _usage_list;
  map_t _cache_map;
  size_t _max_size;
};

} // namespace at::native::detail
