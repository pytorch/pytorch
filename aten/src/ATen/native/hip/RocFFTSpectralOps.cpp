#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// rocFFT path for the transforms issued by stft and istft.
//
// Both of those only ever ask for a batched 1-D transform over the last
// dimension, so this file deliberately skips all of the multi-dimensional
// machinery in native/cuda/SpectralOps.cpp: no dimension permutation, no
// splitting a transform across several plans, no embedded-stride model. It runs
// alongside the hipFFT path rather than replacing it, and is selected by
// use_rocfft_path() from the _fft_*_cufft entry points.
//
// This file is hand-written for ROCm and is not produced by hipify: everything
// under */hip/* is in the ignore list of tools/amd_build/build_amd.py.

#include <ATen/native/hip/RocFFTSpectralOps.h>

#include <ATen/core/DimVector.h>
#include <ATen/core/Tensor.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/native/SpectralOpsUtils.h>
#include <ATen/native/hip/RocFFTPlanCache.h>
#include <c10/core/GradMode.h>
#include <c10/util/ScopeExit.h>
#include <c10/util/env.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/ones.h>
#endif

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace at::native {

using namespace at::native::detail;

namespace {

std::vector<std::unique_ptr<RocFFTParamsLRUCache>> plan_caches;
std::mutex plan_caches_mutex;

RocFFTParamsLRUCache& rocfft_get_plan_cache(DeviceIndex device_index) {
  std::lock_guard<std::mutex> guard(plan_caches_mutex);
  AT_ASSERT(device_index >= 0);

  if (device_index >= static_cast<int64_t>(plan_caches.size())) {
    plan_caches.resize(device_index + 1);
  }
  if (!plan_caches[device_index]) {
    plan_caches[device_index] = std::make_unique<RocFFTParamsLRUCache>();
  }
  return *plan_caches[device_index];
}

// Distance between batches, in elements. A single batch never indexes by
// distance, but rocFFT still validates it, so report the packed value.
int64_t batch_distance(const Tensor& t) {
  return t.size(0) == 1 ? t.size(1) * t.stride(1) : t.stride(0);
}

double fft_normalization_scale(int64_t normalization, IntArrayRef sizes, IntArrayRef dims) {
  auto norm = static_cast<fft_norm_mode>(normalization);
  if (norm == fft_norm_mode::none) {
    return 1.0;
  }

  int64_t signal_numel = 1;
  for (auto dim : dims) {
    signal_numel *= sizes[dim];
  }
  const double scale_denom = (norm == fft_norm_mode::by_root_n) ?
    std::sqrt(signal_numel) : static_cast<double>(signal_numel);
  return 1.0 / scale_denom;
}

const Tensor& apply_normalization(const Tensor& self, int64_t normalization, IntArrayRef sizes, IntArrayRef dims) {
  auto scale = fft_normalization_scale(normalization, sizes, dims);
  return (scale == 1.0) ? self : self.mul_(scale);
}

// Looks the plan up and runs it. cb_data is the device pointer the plan's
// callback receives, and must be null exactly when the plan has no callback.
void run_rocfft_plan(const RocFFTParams& params, const void* in, void* out, void* cb_data,
                     DeviceIndex device_index) {
  auto& plan_cache = rocfft_get_plan_cache(device_index);
  std::lock_guard<std::mutex> guard(plan_cache.mutex);
  const auto& config = plan_cache.lookup(params);

  rocfft_execution_info info = nullptr;
  ROCFFT_CHECK(rocfft_execution_info_create(&info));
  auto info_guard = c10::make_scope_exit([&] { rocfft_execution_info_destroy(info); });
  ROCFFT_CHECK(rocfft_execution_info_set_stream(info, at::cuda::getCurrentCUDAStream().stream()));

  Tensor workspace;
  if (config.workspace_size() > 0) {
    workspace = at::empty({static_cast<int64_t>(config.workspace_size())},
                          at::device(at::kCUDA).dtype(at::kByte));
    ROCFFT_CHECK(rocfft_execution_info_set_work_buffer(
        info, workspace.mutable_data_ptr(), config.workspace_size()));
  }

  if (cb_data != nullptr) {
    rocfft_set_callback_data(info, params.callback_kind_, cb_data);
  }

  void* in_buffers[1] = {const_cast<void*>(in)};
  void* out_buffers[1] = {out};
  ROCFFT_CHECK(rocfft_execute(config.plan(), in_buffers, out_buffers, info));
}

// signal_size is the length of the full signal: for R2C and C2R that is the
// real side, not the hermitian one.
void exec_fft_rocfft(const Tensor& out, const Tensor& self, int64_t signal_size,
                     RocFFTTransformType fft_type, bool forward) {
  // Nothing to transform, and rocFFT rejects a zero batch count.
  if (out.numel() == 0) {
    return;
  }

  // Collapse the leading dimensions into a single batch dimension. This
  // materializes a copy when self is not viewable as (batch, n), which is also
  // what the hipFFT path does.
  auto input = self.reshape({-1, self.size(-1)});
  // A broadcast batch dimension cannot be described by a batch distance.
  if (input.size(0) > 1 && input.stride(0) <= 0) {
    input = input.contiguous();
  }
  // out is always freshly allocated and contiguous, so this is a view.
  auto output = out.view({-1, out.size(-1)});

  lazy_init_rocfft();

  RocFFTParams params(signal_size, input.size(0),
      input.stride(1), batch_distance(input),
      output.stride(1), batch_distance(output),
      fft_type, forward, c10::toRealValueType(input.scalar_type()));

  run_rocfft_plan(params, input.const_data_ptr(), output.data_ptr(), /*cb_data=*/nullptr,
                  input.device().index());
}

// Gathering inside the transform costs a fixed setup per call and a little per
// element, and saves writing and re-reading the framed tensor. The two break
// even around 2^24 framed elements on gfx1030: below that stft runs in well
// under a millisecond and the unfused path is quicker, above it the fused one
// reaches 2.3x. The balance is bandwidth dependent, hence the override. istft
// has no matching floor: its store callback leaves the access pattern alone, so
// it wins at every size.
int64_t rocfft_stft_min_elements() {
  static const int64_t value = [] {
    const auto override = c10::utils::get_env("TORCH_ROCM_ROCFFT_STFT_MIN_ELEMENTS");
    return override.has_value() ? std::stoll(*override) : int64_t{1} << 24;
  }();
  return value;
}

// The gather callback reads its parameters from device memory. Stage through
// pinned memory so the upload is asynchronous; a pageable source would
// serialize the copy against the stream and undo the fusion's savings.
Tensor upload_callback_data(const RocFFTCallbackData& data, Device device) {
  constexpr int64_t nbytes = sizeof(RocFFTCallbackData);
  auto host = at::empty({nbytes}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
  std::memcpy(host.mutable_data_ptr(), &data, nbytes);
  auto result = at::empty({nbytes}, at::device(device).dtype(at::kByte));
  result.copy_(host, /*non_blocking=*/true);
  return result;
}

} // namespace (anonymous)

bool use_rocfft_path(const Tensor& self, IntArrayRef dim) {
  static const bool enabled = c10::utils::check_env("TORCH_ROCM_PREFER_ROCFFT") == true;
  if (!enabled || dim.size() != 1 || dim[0] != self.dim() - 1) {
    return false;
  }
  // rocFFT has a half precision mode too, but it carries the same power-of-two
  // signal size restriction as hipFFT; leave those on the hipFFT path.
  const auto value_type = c10::toRealValueType(self.scalar_type());
  return value_type == ScalarType::Float || value_type == ScalarType::Double;
}

Tensor _fft_r2c_rocfft(const Tensor& self, IntArrayRef dim, int64_t normalization, bool onesided) {
  TORCH_CHECK(self.is_floating_point());
  const auto input_sizes = self.sizes();
  const auto last_dim = dim.back();
  const auto last_dim_halfsize = input_sizes[last_dim] / 2 + 1;

  DimVector onesided_sizes(input_sizes.begin(), input_sizes.end());
  onesided_sizes[last_dim] = last_dim_halfsize;
  IntArrayRef out_sizes = onesided ? IntArrayRef(onesided_sizes) : input_sizes;

  const auto out_options = self.options().dtype(c10::toComplexType(self.scalar_type()));
  auto output = at::empty(out_sizes, out_options);

  // Same over-alignment requirement as the hipFFT path: the real input is read
  // as if it were complex.
  const auto complex_size = 2 * self.element_size();
  auto working_tensor = self;
  if (reinterpret_cast<std::uintptr_t>(self.const_data_ptr()) % complex_size != 0) {
    working_tensor = self.clone(MemoryFormat::Contiguous);
  }

  exec_fft_rocfft(output, working_tensor, input_sizes[last_dim], RocFFTTransformType::R2C,
                  /*forward=*/true);

  // Only the onesided slice holds data; the rest is overwritten below.
  auto out_slice = output.slice(last_dim, 0, last_dim_halfsize);
  apply_normalization(out_slice, normalization, input_sizes, dim);

  if (!onesided) {
    at::native::_fft_fill_with_conjugate_symmetry_(output, dim);
  }
  return output;
}

Tensor _fft_c2r_rocfft(const Tensor& self, IntArrayRef dim, int64_t normalization, int64_t lastdim) {
  TORCH_CHECK(self.is_complex());
  DimVector out_sizes(self.sizes().begin(), self.sizes().end());
  out_sizes[dim.back()] = lastdim;

  auto output = at::empty(out_sizes, self.options().dtype(c10::toRealValueType(self.scalar_type())));
  // Complex to real transforms may overwrite their input even when they are not
  // in-place (gh-34551), so always hand rocFFT a private copy.
  auto temp = self.clone(MemoryFormat::Contiguous);
  exec_fft_rocfft(output, temp, lastdim, RocFFTTransformType::C2R, /*forward=*/false);

  return apply_normalization(output, normalization, out_sizes, dim);
}

Tensor _fft_c2c_rocfft(const Tensor& self, IntArrayRef dim, int64_t normalization, bool forward) {
  TORCH_CHECK(self.is_complex());
  const auto out_sizes = self.sizes();
  auto output = at::empty(out_sizes, self.options());

  exec_fft_rocfft(output, self, out_sizes[dim.back()], RocFFTTransformType::C2C, forward);

  return apply_normalization(output, normalization, out_sizes, dim);
}

Tensor stft_r2c_rocfft(const Tensor& self, int64_t n_fft, int64_t hop_length, int64_t n_frames,
                       const Tensor& window, bool onesided, int64_t normalization) {
  static const bool enabled = c10::utils::check_env("TORCH_ROCM_PREFER_ROCFFT") == true;
  // Single precision only: rocFFT takes the callback's element type from the
  // plan, and the SPIR-V module only defines the float entry points.
  if (!enabled || self.dim() != 2 || hop_length <= 0 ||
      self.scalar_type() != ScalarType::Float) {
    return {};
  }
  // The callback indexes the window out to n_fft, so nothing shorter can be
  // broadcast the way the unfused multiply would.
  if (window.defined() &&
      (window.scalar_type() != ScalarType::Float || window.dim() != 1 || window.numel() != n_fft)) {
    return {};
  }

  // _stft_r2c has no derivative; the unfused decomposition is differentiable
  // only because autograd sees through to the mul and the transform. stft
  // already keeps graph-recording calls away from here, so this only catches
  // direct callers, for which silently dropping the graph would be worse.
  if (at::GradMode::is_enabled() &&
      (self.requires_grad() || (window.defined() && window.requires_grad()))) {
    return {};
  }

  const int64_t batch = self.size(0);
  if (batch <= 0 || n_frames <= 0) {
    return {};
  }

  const int64_t framed_elements = batch * n_frames * n_fft;
  if (framed_elements < rocfft_stft_min_elements()) {
    return {};
  }

  lazy_init_rocfft();
  if (!rocfft_callbacks_available()) {
    return {};
  }

  auto signal = self;
  // The callback walks the signal with unit stride, stepping between channels by
  // a single positive offset.
  if (signal.stride(1) != 1 || (batch > 1 && signal.stride(0) <= 0)) {
    signal = signal.contiguous();
  }
  // Same over-alignment requirement as the unfused R2C path: the real input is
  // read as if it were complex.
  const int64_t complex_size = 2 * self.element_size();
  if (reinterpret_cast<std::uintptr_t>(signal.const_data_ptr()) % complex_size != 0) {
    signal = signal.clone(MemoryFormat::Contiguous);
  }

  // The callbacks do their index arithmetic in 32 bits: both the offset rocFFT
  // hands out and the signal index it is mapped to have to fit.
  const int64_t max_index = (batch - 1) * signal.stride(0) + (n_frames - 1) * hop_length + n_fft;
  const int64_t index_limit = std::numeric_limits<uint32_t>::max();
  if (framed_elements > index_limit || max_index > index_limit) {
    return {};
  }

  // Folding an all-ones window in is exact, and keeps this to one code path.
  const auto window_ = window.defined() ? window.contiguous() : at::ones({n_fft}, self.options());

  const int64_t out_last = onesided ? n_fft / 2 + 1 : n_fft;
  auto output = at::empty({batch, n_frames, out_last},
                          self.options().dtype(c10::toComplexType(self.scalar_type())));

  uint32_t n_fft_log2 = 0;
  while ((int64_t{1} << n_fft_log2) < n_fft) {
    ++n_fft_log2;
  }
  const bool pow2 = (int64_t{1} << n_fft_log2) == n_fft;

  RocFFTCallbackData cb_data = {};
  cb_data.window = window_.const_data_ptr<float>();
  cb_data.n_fft = static_cast<uint32_t>(n_fft);
  cb_data.n_frames = static_cast<uint32_t>(n_frames);
  cb_data.hop = static_cast<uint32_t>(hop_length);
  cb_data.signal_stride = static_cast<uint32_t>(signal.stride(0));
  cb_data.n_fft_log2 = n_fft_log2;
  const auto cb_dev = upload_callback_data(cb_data, self.device());

  // Normalization rides along in the plan rather than as a second pass over the
  // output, which for stft is as much data as the transform itself writes.
  RocFFTParams params(n_fft, batch * n_frames, /*in_stride=*/1, /*in_distance=*/n_fft,
      /*out_stride=*/1, /*out_distance=*/output.stride(1), RocFFTTransformType::R2C,
      /*forward=*/true, ScalarType::Float,
      pow2 ? RocFFTCallbackKind::LoadGatherPow2 : RocFFTCallbackKind::LoadGather,
      fft_normalization_scale(normalization, {n_fft}, {0}));

  run_rocfft_plan(params, signal.const_data_ptr(), output.data_ptr(), cb_dev.data_ptr(),
                  self.device().index());

  if (!onesided) {
    at::native::_fft_fill_with_conjugate_symmetry_(output, {2});
  }
  return output;
}

Tensor istft_c2r_rocfft(const Tensor& self, int64_t n_fft, const Tensor& window,
                        int64_t normalization) {
  static const bool enabled = c10::utils::check_env("TORCH_ROCM_PREFER_ROCFFT") == true;
  // Single precision only, for the same reason as the stft side: the SPIR-V
  // module only defines the float entry points.
  if (!enabled || self.dim() != 3 || self.scalar_type() != ScalarType::ComplexFloat ||
      window.scalar_type() != ScalarType::Float || window.dim() != 1 ||
      window.numel() != n_fft) {
    return {};
  }

  // _istft_c2r has no derivative. istft keeps graph-recording calls away from
  // here; this only catches direct callers.
  if (at::GradMode::is_enabled() && (self.requires_grad() || window.requires_grad())) {
    return {};
  }

  const int64_t channels = self.size(0);
  const int64_t n_frames = self.size(1);
  if (channels <= 0 || n_frames <= 0 || self.size(2) != n_fft / 2 + 1) {
    return {};
  }

  // The callback does its index arithmetic in 32 bits.
  if (channels * n_frames * n_fft > std::numeric_limits<uint32_t>::max()) {
    return {};
  }

  lazy_init_rocfft();
  if (!rocfft_callbacks_available()) {
    return {};
  }

  auto output = at::empty({channels, n_frames, n_fft},
                          self.options().dtype(c10::toRealValueType(self.scalar_type())));
  // Complex to real transforms may overwrite their input even when they are not
  // in-place (gh-34551), so always hand rocFFT a private copy. The copy also
  // settles the layout, since istft arrives here on a transpose.
  const auto input = self.clone(MemoryFormat::Contiguous);
  const auto window_ = window.contiguous();

  RocFFTCallbackData cb_data = {};
  cb_data.window = window_.const_data_ptr<float>();
  cb_data.n_fft = static_cast<uint32_t>(n_fft);
  const auto cb_dev = upload_callback_data(cb_data, self.device());

  RocFFTParams params(n_fft, channels * n_frames, /*in_stride=*/1,
      /*in_distance=*/input.stride(1), /*out_stride=*/1, /*out_distance=*/n_fft,
      RocFFTTransformType::C2R, /*forward=*/false, ScalarType::Float,
      RocFFTCallbackKind::StoreWindow,
      fft_normalization_scale(normalization, {n_fft}, {0}));

  run_rocfft_plan(params, input.const_data_ptr(), output.data_ptr(), cb_dev.data_ptr(),
                  self.device().index());
  return output;
}

} // namespace at::native
