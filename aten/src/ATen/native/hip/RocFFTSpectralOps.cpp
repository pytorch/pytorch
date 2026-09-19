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
#include <c10/util/CallOnce.h>
#include <c10/util/ScopeExit.h>
#include <c10/util/env.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
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

// rocfft_setup() has to run before any other rocFFT call. There is a matching
// rocfft_cleanup(), but like the rest of PyTorch we leave device libraries
// standing at exit rather than racing teardown against live tensors.
void lazy_init_rocfft() {
  static c10::once_flag flag;
  c10::call_once(flag, [] { ROCFFT_CHECK(rocfft_setup()); });
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

  auto& plan_cache = rocfft_get_plan_cache(input.device().index());
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

  void* in_buffers[1] = {const_cast<void*>(input.const_data_ptr())};
  void* out_buffers[1] = {output.data_ptr()};
  ROCFFT_CHECK(rocfft_execute(config.plan(), in_buffers, out_buffers, info));
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

} // namespace at::native
