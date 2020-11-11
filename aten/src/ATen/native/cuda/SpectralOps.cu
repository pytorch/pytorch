#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Config.h>
#include <ATen/Dispatch.h>
#include <ATen/Utils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/native/TensorIterator.h>
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

// Offset calculator for indexing in Hermitian mirrored order.
// In mirrored dims, maps linear index i to (n - i) % n
template <typename index_t>
struct HermitianSymmetryOffsetCalculator {
  using offset_type = at::detail::Array<index_t, 1>;
  using dim_type = std::remove_cv_t<decltype(MAX_DIMS)>;
  dim_type dims;
  IntDivider<index_t> sizes_[MAX_DIMS];
  index_t strides_[MAX_DIMS];
  uint32_t mirror_dim_;  // bit mask
  static_assert(MAX_DIMS < 32, "Need a bigger mask type");

  HermitianSymmetryOffsetCalculator(
      IntArrayRef sizes, IntArrayRef strides, IntArrayRef dim,
      const int64_t element_size){
    TORCH_INTERNAL_ASSERT(sizes.size() == strides.size());
    TORCH_INTERNAL_ASSERT(sizes.size() <= MAX_DIMS);
    dims = sizes.size();

    for (dim_type i = 0; i < MAX_DIMS; ++i) {
      if (i < dims) {
        sizes_[i] = IntDivider<index_t>(sizes[i]);
        strides_[i] = strides[i] / element_size;
      } else {
        sizes_[i] = IntDivider<index_t>(1);
        strides_[i] = 0;
      }
    }

    mirror_dim_ = 0;
    for (int64_t i = 0; i < dim.size(); ++i) {
      mirror_dim_ |= (uint32_t{1} << dim[i]);
    }
  }

  C10_HOST_DEVICE offset_type get(index_t linear_idx) const {
    index_t offset = 0;

    for (dim_type dim = 0; dim < dims; ++dim) {
      auto divmod = sizes_[dim].divmod(linear_idx);
      linear_idx = divmod.div;

      if ((mirror_dim_ & (uint32_t{1} << dim)) == 0) {
        offset += divmod.mod * strides_[dim];
      } else if (divmod.mod != 0) {
        offset += (sizes_[dim].divisor - divmod.mod) * strides_[dim];
      }
    }
    offset_type offsets;
    offsets[0] = offset;
    return offsets;
  }
};

// out[:] = conj(in[:]) where in and out ordering is generalized by offset calculators
template <typename scalar_t, typename inp_calc_t, typename out_calc_t>
C10_LAUNCH_BOUNDS_1(cuda::detail::CUDA_NUM_THREADS)
__global__ void _fft_conjugate_copy_kernel(
    int64_t numel, scalar_t * out_data, const scalar_t * in_data,
    inp_calc_t ic, out_calc_t oc) {
  CUDA_KERNEL_LOOP_TYPE(index, numel, int64_t) {
    auto in_offset = ic.get(index)[0];
    auto out_offset = oc.get(index)[0];
    out_data[out_offset] = std::conj(in_data[in_offset]);
  }
}

// In real-to-complex transform, cuFFT only fills half of the values due to
// conjugate symmetry. See native/SpectralUtils.h for more details.
// The following function fills in the other half with symmetry in
// case of real-to-complex transform with onesided=False flag.
// See NOTE [ Fourier Transform Conjugate Symmetry ] in native/SpectralOpsUtils.h.

// input should be a tensor of same size as full (twosided)
// signals, but only contains half (onesided) of the values.
// This function modifies inplace.
void _fft_fill_with_conjugate_symmetry_cuda_(
    ScalarType dtype, IntArrayRef mirror_dims, IntArrayRef signal_half_sizes,
    IntArrayRef in_strides, const void * in_data,
    IntArrayRef out_strides, void * out_data) {
  // Do the actual conjugate mirroring.
  // TODO: consider adding a 32bit indexed kernel for improved performance
  auto* in_strides_ptr = in_strides.data();
  const int ndim = in_strides.size();
  const int64_t element_size = scalarTypeToTypeMeta(dtype).itemsize();
  OffsetCalculator<1, int64_t> input_offset_calculator(
      ndim, signal_half_sizes.data(), &in_strides_ptr, &element_size);
  HermitianSymmetryOffsetCalculator<int64_t> output_offset_calculator(
      signal_half_sizes, out_strides, mirror_dims, element_size);

  const auto numel = at::prod_intlist(signal_half_sizes);
  AT_DISPATCH_COMPLEX_TYPES(dtype, "_fft_fill_with_conjugate_symmetry", [&] {
        using namespace cuda::detail;
        _fft_conjugate_copy_kernel<<<
          GET_BLOCKS(numel), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
              numel,
              static_cast<scalar_t*>(out_data),
              static_cast<const scalar_t*>(in_data),
              input_offset_calculator,
              output_offset_calculator);
      });
}

REGISTER_DISPATCH(fft_fill_with_conjugate_symmetry_stub, &_fft_fill_with_conjugate_symmetry_cuda_);

// Execute a pre-planned tranform
static void exec_cufft_plan(
    const CuFFTConfig &config, void* in_data, void* out_data, bool forward) {
  auto& plan = config.plan();
#ifdef __HIP_PLATFORM_HCC__
  auto value_type = config.data_type();
  if (value_type == kFloat) {
    switch (config.transform_type()) {
      case CuFFTTransformType::C2C: {
        CUFFT_CHECK(hipfftExecC2C(plan, static_cast<hipfftComplex*>(in_data),
                                  static_cast<hipfftComplex*>(out_data),
                                  forward ? HIPFFT_FORWARD : HIPFFT_BACKWARD));
        return;
      }
      case CuFFTTransformType::R2C: {
        CUFFT_CHECK(hipfftExecC2R(plan, static_cast<hipfftComplex*>(in_data),
                                  static_cast<hipfftReal*>(out_data)));
        return;
      }
      case CuFFTTransformType::C2R: {
        CUFFT_CHECK(hipfftExecR2C(plan, static_cast<hipfftReal*>(in_data),
                                  static_cast<hipfftComplex*>(out_data)));
        return;
      }
    }
  } else if (value_type == kDouble) {
    switch (config.transform_type()) {
      case CuFFTTransformType::C2C: {
        CUFFT_CHECK(hipfftExecZ2Z(plan, static_cast<hipfftDoubleComplex*>(in_data),
                                  static_cast<hipfftDoubleComplex*>(out_data),
                                  forward ? HIPFFT_FORWARD : HIPFFT_BACKWARD));
        return;
      }
      case CuFFTTransformType::R2C: {
        CUFFT_CHECK(hipfftExecD2Z(plan, static_cast<hipfftDoubleReal*>(in_data),
                                  static_cast<hipfftDoubleComplex*>(out_data)));
        return;
      }
      case CuFFTTransformType::C2R: {
        CUFFT_CHECK(hipfftExecZ2D(plan, static_cast<hipfftDoubleComplex*>(in_data),
                                  static_cast<hipfftDoubleReal*>(out_data)));
        return;
      }
    }
  }
  TORCH_CHECK(false, "hipFFT doesn't support transforms on type: ", value_type);
#else
  CUFFT_CHECK(cufftXtExec(plan, in_data, out_data,
                          forward ? CUFFT_FORWARD : CUFFT_INVERSE));
#endif
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
    IntArrayRef checked_signal_sizes, fft_norm_mode norm, bool onesided,
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
  exec_cufft_plan(config, input.data_ptr(), output.data_ptr(), !inverse);

  // rescale if requested
  auto size_last_signal_dim = checked_signal_sizes[signal_ndim - 1];
  if (norm != fft_norm_mode::none) {
    auto signal_numel = at::prod_intlist(checked_signal_sizes);
    double scale_denom;
    if (norm == fft_norm_mode::by_root_n) {
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
    DimVector signal_dims(signal_ndim);
    std::iota(signal_dims.begin(), signal_dims.end(), 1);
    auto out_as_complex = at::view_as_complex(output);
    at::native::_fft_fill_with_conjugate_symmetry_(out_as_complex, signal_dims);
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
                  IntArrayRef checked_signal_sizes, int64_t normalization, bool onesided,
                  IntArrayRef output_sizes) {

  CuFFTParamsLRUCache& plan_cache = cufft_get_plan_cache(self.device().index());

  Tensor input = self;
  const auto fft_type = GetCuFFTTransformType(complex_input, complex_output);

  if (complex_input) {
    TORCH_CHECK(input.size(-1) == 2, "Expected a complex (size 2) last dimension");
  }


  // Slice when twosided complex-to-real. This is not always needed because we
  // calculate the inembed. But it will benefit us in certain cases where we
  // clone the input tensor.
  //
  // See NOTE [ cuFFT Embedded Strides ].
  // See NOTE [ Fourier Transform Conjugate Symmetry ] in native/SpectralOpsUtils.h.
  if (fft_type == CuFFTTransformType::C2R && !onesided) {
    auto onesided_size = infer_ft_real_to_complex_onesided_size(checked_signal_sizes[signal_ndim - 1]);
    input = input.narrow(signal_ndim, 0, onesided_size);
  }

  // cuFFT requires input and output data pointers to complex type aligned.
  // Our newly allocated output tensor is always 512 bytes aligned so it is fine
  // (see kRoundSmall and kRoundLarge in THCCachingAllocator.cpp), but we do
  // need to check input tensor to make sure that it is not unaligned, e.g.,
  // from a slicing.
  bool must_clone = false;
  auto complex_size_bytes = 2 * input.element_size();
  if (reinterpret_cast<std::uintptr_t>(input.data_ptr()) % complex_size_bytes != 0) {
    must_clone = true;
  }

  if (complex_input) {
    auto strides = input.strides();
    // Real/imag dimension must be like complex type.
    must_clone |= strides.back() != 1;
    // Strides of other dimensions needs to be aligned when viewed as complex
    // type, i.e., multiples of 2.
    must_clone |= std::any_of(strides.begin(), strides.end() - 1,
                              [&](int64_t stride) { return stride % 2 != 0; });

    // Complex to real FFTs may overwrite the input buffer (gh-34551)
    must_clone |= !complex_output;
  }

  if (must_clone) {
    input = input.clone(MemoryFormat::Contiguous);
  }

  // Now that we have done error check and data_ptr checks, we delegate all
  // further cuFFT parameter computation and plan creation to the helper class
  // CuFFTConfig in CuFFTPlanCache.h.

  // If plan caching is enabled, we check the cache. Note that this accesses
  // plan_cache.max_size() and thus makes this function less functional.
  // However, integrating additional arguments into the "public" level c++ APIs,
  // e.g., irfft, is difficult as we have a long call sequence looking like
  //   irfft --> _fft --> _fft_with_size --dispatching-to-> _fft_cufft

  DimVector in_strides(signal_ndim + 1);
  auto input_strides = input.strides();
  for (int64_t i = signal_ndim; i >= 0; --i) {
    in_strides[i] = complex_input ? input_strides[i] / 2 : input_strides[i];
  }

  DimVector out_strides(signal_ndim + 1);
  out_strides[signal_ndim] = 1;
  if (fft_type == CuFFTTransformType::R2C && onesided) {
    out_strides[signal_ndim - 1] = checked_signal_sizes[signal_ndim - 1] / 2 + 1;
  } else {
    out_strides[signal_ndim - 1] = checked_signal_sizes[signal_ndim - 1];
  }
  for (int64_t i = signal_ndim - 2; i >= 0; --i) {
    out_strides[i] = out_strides[i + 1] * checked_signal_sizes[i];
  }

  DimVector full_sizes(signal_ndim + 1);
  full_sizes[0] = self.size(0);
  std::copy(checked_signal_sizes.begin(), checked_signal_sizes.end(), full_sizes.begin() + 1);
  CuFFTParams Params(in_strides, out_strides, full_sizes, fft_type,
                     c10::toValueType(input.scalar_type()));

  // This read is not locked for perf reason. Shouldn't matter too much because
  // we check again after acquiring the lock.
  if (plan_cache.max_size() > 0) {
    std::lock_guard<std::mutex> guard(plan_cache.mutex);
    if (plan_cache.max_size() > 0) {  // check again after acquiring the lock
      const CuFFTConfig &config = plan_cache.lookup(Params);
      return _run_cufft(config, input, signal_ndim, complex_input,
                        complex_output, inverse, checked_signal_sizes,
                        static_cast<fft_norm_mode>(normalization),
                        onesided, output_sizes, must_clone);
    }
  }
  CuFFTConfig config(Params);
  return _run_cufft(config, input, signal_ndim, complex_input,
                    complex_output, inverse, checked_signal_sizes,
                    static_cast<fft_norm_mode>(normalization),
                    onesided, output_sizes, must_clone);
}

}} // at::native
