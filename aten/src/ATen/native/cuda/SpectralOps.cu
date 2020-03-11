#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Config.h>
#include <ATen/Dispatch.h>
#include <ATen/Utils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/native/SpectralOpsUtils.h>
#include <ATen/native/cuda/CuFFTUtils.h>
#include <ATen/native/cuda/CuFFTPlanCache.h>
#include <THC/THCTensorSort.cuh>
#include <THC/THCThrustAllocator.cuh>

#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include <cufft.h>
#include <cufftXt.h>
#include <vector>
#include <cmath>

namespace at { namespace native {

using namespace at::native::detail;

// In real-to-complex transform, cuFFT only fills half of the values due to
// conjugate symmetry. See native/SpectralUtils.h for more details.
// The following structs are used to fill in the other half with symmetry in
// case of real-to-complex transform with onesided=False flag.
// See NOTE [ Fourier Transform Conjugate Symmetry ] in native/SpectralOpsUtils.h.

// counting_iterator => index to fill
struct cnt_to_dst_idx_functor : public thrust::unary_function<int64_t, int64_t>
{
  int64_t last_dim_size;
  int64_t last_dim_start_slice;
  int64_t last_dim_to_fill_size;

  cnt_to_dst_idx_functor(int64_t last_dim_size, int64_t last_dim_start_slice) :
    last_dim_size(last_dim_size), last_dim_start_slice(last_dim_start_slice),
    last_dim_to_fill_size(last_dim_size - last_dim_start_slice) {}

  // HIP wants __host__ __device__ tag, CUDA does not
#ifdef __HIP_PLATFORM_HCC__
  __host__ __device__
#endif
  cnt_to_dst_idx_functor & operator=(const cnt_to_dst_idx_functor&) = default;

  __host__ __device__ __forceinline__
  int64_t operator()(const int64_t& i) const
  {
    int64_t imag = i % 2;
    int64_t idx = i / 2;
    int64_t num_dim = idx / last_dim_to_fill_size;
    int64_t slice_idx = idx % last_dim_to_fill_size;
    return (num_dim * last_dim_size + last_dim_start_slice + slice_idx) * 2 + imag;
  }
};

// index to fill => index to read from
template <typename scalar_t>
struct dst_idx_to_src_functor : public thrust::unary_function<int64_t, scalar_t>
{
  // output can have at most dim 5 (batch + 3 signal dim + real/imag)
  int64_t sizes[max_rank + 2], strides[max_rank + 2];
  const int64_t signal_ndim;
  scalar_t *data;  // device ptr

  dst_idx_to_src_functor(const Tensor& batched_complex_signal)
    : signal_ndim(batched_complex_signal.dim() - 1),
      data(batched_complex_signal.data_ptr<scalar_t>()) {
    for (int64_t i = 0; i < signal_ndim; i++) {
      sizes[i] = batched_complex_signal.size(i);
      strides[i] = batched_complex_signal.stride(i);
    }
  }

  __device__ __forceinline__
  scalar_t operator()(const int64_t& write_idx_with_imag) const
  {
    int64_t imag = write_idx_with_imag % 2;
    // all but first (batch) and last (real/imag) dims need to be reflected
    int64_t read_idx = 0;
    int64_t remainder = write_idx_with_imag - imag;
    int64_t dim_idx, dim_stride;
    for (int64_t i = 0; i < signal_ndim; i++) {
      dim_stride = strides[i];
      dim_idx = remainder / dim_stride;
      if (i == 0) {
        read_idx += dim_idx * dim_stride;
      } else if (dim_idx != 0) {
        read_idx += (sizes[i] - dim_idx) * dim_stride;
      }
      remainder = remainder % dim_stride;
    }
    if (imag) {
      return -data[read_idx + 1];
    } else {
      return data[read_idx];
    }
  }
};

// input should be a contiguous batched tensor of same size as full (twosided)
// signals, but only contains half (onesided) of the values.
// This function modifies inplace.
__forceinline__
static void _fft_fill_with_conjugate_symmetry_(Tensor& input,
                      int64_t size_last_dim, int64_t last_dim_start_slice) {
  if (last_dim_start_slice >= size_last_dim) {
    return;
  }

  // copy
  int64_t n = input.numel() / size_last_dim * (size_last_dim - last_dim_start_slice);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
  auto policy = thrust::cuda::par(allocator).on(stream);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "_fft_fill_with_conjugate_symmetry_", [&] {
    typedef thrust::device_ptr<scalar_t> device_ptr;
    typedef thrust::counting_iterator<int64_t> counter;
    typedef thrust::transform_iterator<cnt_to_dst_idx_functor, counter> dst_idx_iterator;
    typedef thrust::permutation_iterator<device_ptr, dst_idx_iterator> dst_iterator;
    typedef thrust::transform_iterator<dst_idx_to_src_functor<scalar_t>, dst_idx_iterator> src_iterator;

    dst_idx_iterator dst_idxs(counter(0), cnt_to_dst_idx_functor(size_last_dim, last_dim_start_slice));

    auto data = device_ptr(input.data_ptr<scalar_t>());
    dst_iterator dsts(data, dst_idxs);
    src_iterator srcs(dst_idxs, dst_idx_to_src_functor<scalar_t>(input));
    thrust::copy_n(policy, srcs, n, dsts);
  });
}

// NOTE [ cuFFT Embedded Strides ]
//
// cuFFT supports a subset of arbitrary strides via their "advanced data layout"
// option (http://docs.nvidia.com/cuda/cufft/index.html#advanced-data-layout).
// Specifically, these are tensors that can be viewed as subtensors resulted
// from slicing a larger contiguous tensors. For such input tensors, let the
// sizes of the enclosing tensor be `inembed`, and we can have in 3d case:
//
//     input[x, y, z] = input[((x * inembed[1] + y) * inembed[2] + z)]
//
// Above is the simplified formula ignoring the batch dimension. In fact, the
// last dimension of the enclosing tensor doesn't have to be contiguous, i.e.,
// it can be greater than 1. Then one can set the base stride for the enclosing
// tensor with `istride`. Then we have
//
//     input[x, y, z] = input[((x * inembed[1] + y) * inembed[2] + z) * istride]
//
// For example, consider
//
//     enclosing = torch.zeros(6, 8, 10)  # contiguous
//     input = enclosing[:4, 2:6, 6:]
//     input.size()                       # [ 4,  4,  4]
//     input.stride()                     # [80, 10,  1]
//     # inembed = [6, 8, 10]
//     input[2, 1, 3] = input[((2 * 8) + 1) * 10 + 3]   # using above formula
//                    = input[173]
//                    = input[2 * 80 + 1 * 10 + 1 * 3]  # using strides directly
//
// Generally, the embedded strides can be computed as
//
//     embed[i] = stride[i - 1] / stride[i].
//
// Note that the value of embed[0] isn't used to compute indices and doesn't
// matter.
//
// Contrary to advanced data layout, simple layout means that *embeds have
// unit-strides. In particular, unit-stride refers to that the input and output
// tensors being contiguous, and that the strides at the innermost signal
// dimension being unit (1) w.r.t. the corresponding data type.

static inline Tensor _run_cufft(
    const CuFFTConfig &config, Tensor& input, int64_t signal_ndim,
    bool complex_input, bool complex_output, bool inverse,
    IntArrayRef checked_signal_sizes, bool normalized, bool onesided,
    IntArrayRef output_sizes, bool input_was_cloned
) {
  if (config.should_clone_input() && !input_was_cloned) {
    input = input.clone(at::MemoryFormat::Contiguous);
  }

  auto& plan = config.plan();
  auto& ctx = at::globalContext();

  // set output
  auto output = at::empty(output_sizes, input.options());

  // set to current stream
  CUFFT_CHECK(cufftSetStream(plan, at::cuda::getCurrentCUDAStream()));

  auto ws = at::empty({ config.workspace_size() }, at::device(at::kCUDA).dtype(at::kByte));
  CUFFT_CHECK(cufftSetWorkArea(plan, ws.data_ptr()));

  // run
#ifdef __HIP_PLATFORM_HCC__
  if (input.scalar_type() == ScalarType::Float) {
      if (complex_input && complex_output) {
        CUFFT_CHECK(hipfftExecC2C(plan, static_cast<hipfftComplex*>(input.data_ptr()),
          static_cast<hipfftComplex*>(output.data_ptr()),
          inverse ? HIPFFT_BACKWARD : HIPFFT_FORWARD));
      } else if (complex_input && !complex_output) {
        CUFFT_CHECK(hipfftExecC2R(plan, static_cast<hipfftComplex*>(input.data_ptr()),
          static_cast<hipfftReal*>(output.data_ptr())));
      } else if (!complex_input && complex_output) {
        CUFFT_CHECK(hipfftExecR2C(plan, static_cast<hipfftReal*>(input.data_ptr()),
          static_cast<hipfftComplex*>(output.data_ptr())));
      } else {
        AT_ERROR("hipFFT doesn't support r2r (float)");
      }
    } else if (input.scalar_type() == ScalarType::Double) {
      if (complex_input && complex_output) {
        CUFFT_CHECK(hipfftExecZ2Z(plan, static_cast<hipfftDoubleComplex*>(input.data_ptr()),
          static_cast<hipfftDoubleComplex*>(output.data_ptr()),
          inverse ? HIPFFT_BACKWARD : HIPFFT_FORWARD));
      } else if (complex_input && !complex_output) {
        CUFFT_CHECK(hipfftExecZ2D(plan, static_cast<hipfftDoubleComplex*>(input.data_ptr()),
          static_cast<hipfftDoubleReal*>(output.data_ptr())));
      } else if (!complex_input && complex_output) {
        CUFFT_CHECK(hipfftExecD2Z(plan, static_cast<hipfftDoubleReal*>(input.data_ptr()),
          static_cast<hipfftDoubleComplex*>(output.data_ptr())));
      } else {
        AT_ERROR("hipFFT doesn't support r2r (double)");
      }
    } else {
      std::ostringstream ss;
      ss << "hipFFT doesn't support tensor of type: "
         << toString(input.scalar_type());
      AT_ERROR(ss.str());
    }
#else
  CUFFT_CHECK(cufftXtExec(plan, input.data_ptr(), output.data_ptr(),
    inverse ? CUFFT_INVERSE : CUFFT_FORWARD));
#endif

  // rescale if needed by normalized flag or inverse transform
  auto size_last_signal_dim = checked_signal_sizes[signal_ndim - 1];
  if (normalized || inverse) {
    auto signal_numel = at::prod_intlist(checked_signal_sizes);
    double scale_denom;
    if (normalized) {
      scale_denom = std::sqrt(static_cast<double>(signal_numel));
    } else {
      scale_denom = static_cast<double>(signal_numel);
    }
    if (!complex_input && complex_output && !onesided) {
      auto end_data_slice = infer_ft_real_to_complex_onesided_size(size_last_signal_dim);
      output.narrow(signal_ndim, 0, end_data_slice).div_(scale_denom);
    } else {
      output.div_(scale_denom);
    }
  }

  // if needed, fill out the other half using conjugate symmetry
  if (!complex_input && complex_output && !onesided) {
    auto start_slice = infer_ft_real_to_complex_onesided_size(size_last_signal_dim);
    _fft_fill_with_conjugate_symmetry_(output, size_last_signal_dim, start_slice);
  }
  return output;
}

// The cuFFT plan cache
// unique_ptr for nullability and to avoid reference invalidation on vector resize
static std::vector<std::unique_ptr<CuFFTParamsLRUCache>> plan_caches;
static std::mutex plan_caches_mutex;

static inline
CuFFTParamsLRUCache &cufft_get_plan_cache(int64_t device_index) {
  std::lock_guard<std::mutex> guard(plan_caches_mutex);

  AT_ASSERT(device_index >= 0);

  if (device_index >= plan_caches.size()) {
    plan_caches.resize(device_index + 1);
  }

  if (!plan_caches[device_index]) {
    plan_caches[device_index] = std::make_unique<CuFFTParamsLRUCache>();
  }

  return *plan_caches[device_index];
}


namespace detail {

int64_t cufft_get_plan_cache_max_size_impl(int64_t device_index) {
  TORCH_CHECK(0 <= device_index && device_index < at::detail::getCUDAHooks().getNumGPUs(),
    "cufft_get_plan_cache_max_size: expected 0 <= device_index < ",
    at::detail::getCUDAHooks().getNumGPUs(), "], but got device_index=",
    device_index);
  return cufft_get_plan_cache(device_index).max_size();
}

void cufft_set_plan_cache_max_size_impl(int64_t device_index, int64_t max_size) {
  TORCH_CHECK(0 <= device_index && device_index < at::detail::getCUDAHooks().getNumGPUs(),
    "cufft_set_plan_cache_max_size: expected 0 <= device_index < ",
    at::detail::getCUDAHooks().getNumGPUs(), "], but got device_index=",
    device_index);
  return cufft_get_plan_cache(device_index).resize(max_size);
}

int64_t cufft_get_plan_cache_size_impl(int64_t device_index) {
  TORCH_CHECK(0 <= device_index && device_index < at::detail::getCUDAHooks().getNumGPUs(),
    "cufft_get_plan_cache_size: expected 0 <= device_index < ",
    at::detail::getCUDAHooks().getNumGPUs(), "], but got device_index=",
    device_index);
  return cufft_get_plan_cache(device_index).size();
}

void cufft_clear_plan_cache_impl(int64_t device_index) {
  TORCH_CHECK(0 <= device_index && device_index < at::detail::getCUDAHooks().getNumGPUs(),
    "cufft_clear_plan_cache: expected 0 <= device_index < ",
    at::detail::getCUDAHooks().getNumGPUs(), "], but got device_index=",
    device_index);
  return cufft_get_plan_cache(device_index).clear();
}

} // namespace at::native::detail

// cuFFT
// Currently not utilizing multi GPUs so this can be potentially sped up.
Tensor _fft_cufft(const Tensor& self, int64_t signal_ndim,
                  bool complex_input, bool complex_output, bool inverse,
                  IntArrayRef checked_signal_sizes, bool normalized, bool onesided,
                  IntArrayRef output_sizes) {

  CuFFTParamsLRUCache& plan_cache = cufft_get_plan_cache(self.device().index());

  Tensor input = self;
  bool input_was_cloned = false;

  // Slice when twosided complex-to-real. This is not always needed because we
  // calculate the inembed. But it will benefit us in certain cases where we
  // clone the input tensor.
  //
  // See NOTE [ cuFFT Embedded Strides ].
  // See NOTE [ Fourier Transform Conjugate Symmetry ] in native/SpectralOpsUtils.h.
  if (complex_input && !complex_output && !onesided) {
    auto onesided_size = infer_ft_real_to_complex_onesided_size(checked_signal_sizes[signal_ndim - 1]);
    input = input.narrow(signal_ndim, 0, onesided_size);
  }

  // cuFFT requires input and output data pointers to complex type aligned.
  // Our newly allocated output tensor is always 512 bytes aligned so it is fine
  // (see kRoundSmall and kRoundLarge in THCCachingAllocator.cpp), but we do
  // need to check input tensor to make sure that it is not unaligned, e.g.,
  // from a slicing.
  auto complex_size_bytes = 2 * input.element_size();
  if (reinterpret_cast<std::uintptr_t>(input.data_ptr()) % complex_size_bytes != 0) {
    input = input.clone(at::MemoryFormat::Contiguous);
    input_was_cloned = true;
  }

  // Now that we have done error check and data_ptr checks, we delegate all
  // further cuFFT parameter computation and plan creation to the helper class
  // CuFFTConfig in CuFFTUtils.h.

  // If plan caching is enabled, we check the cache. Note that this accesses
  // plan_cache.max_size() and thus makes this function less functional.
  // However, integrating additional arguments into the "public" level c++ APIs,
  // e.g., irfft, is difficult as we have a long call sequence looking like
  //   irfft --> _fft --> _fft_with_size --dispatching-to-> _fft_cufft

  // This read is not locked for perf reason. Shouldn't matter too much because
  // we check again after acquiring the lock.
  if (plan_cache.max_size() > 0) {
    CuFFTParams params;
    setCuFFTParams(&params, input, signal_ndim, complex_input,
      complex_output, checked_signal_sizes, onesided);
    std::lock_guard<std::mutex> guard(plan_cache.mutex);
    if (plan_cache.max_size() > 0) {  // check again after acquiring the lock
      const CuFFTConfig &config = plan_cache.try_emplace_value(std::move(params),
                                             input, signal_ndim, complex_input,
                                             complex_output, checked_signal_sizes,
                                             onesided, output_sizes);
      return _run_cufft(config, input, signal_ndim, complex_input,
                        complex_output, inverse, checked_signal_sizes, normalized,
                        onesided, output_sizes, input_was_cloned);
    }
  }
  CuFFTConfig config(input, signal_ndim, complex_input, complex_output,
                     checked_signal_sizes, onesided, output_sizes);
  return _run_cufft(config, input, signal_ndim, complex_input,
                    complex_output, inverse, checked_signal_sizes, normalized,
                    onesided, output_sizes, input_was_cloned);
}

}} // at::native
