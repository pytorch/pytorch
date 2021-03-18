#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Config.h>
#include <ATen/Dispatch.h>
#include <ATen/Utils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/SpectralOpsUtils.h>
#include <ATen/native/cuda/CuFFTUtils.h>
#include <ATen/native/cuda/CuFFTPlanCache.h>
#include <c10/util/accumulate.h>
#include <THC/THCTensorSort.cuh>
#include <THC/THCThrustAllocator.cuh>

#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include <cufft.h>
#include <cufftXt.h>

#include <cmath>
#include <vector>


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

  const auto numel = c10::multiply_integers(signal_half_sizes);
  AT_DISPATCH_COMPLEX_TYPES(dtype, "_fft_fill_with_conjugate_symmetry", [&] {
        using namespace cuda::detail;
        _fft_conjugate_copy_kernel<<<
          GET_BLOCKS(numel), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
              numel,
              static_cast<scalar_t*>(out_data),
              static_cast<const scalar_t*>(in_data),
              input_offset_calculator,
              output_offset_calculator);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
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
        CUFFT_CHECK(hipfftExecR2C(plan, static_cast<hipfftReal*>(in_data),
                                  static_cast<hipfftComplex*>(out_data)));
        return;
      }
      case CuFFTTransformType::C2R: {
        CUFFT_CHECK(hipfftExecC2R(plan, static_cast<hipfftComplex*>(in_data),
                                  static_cast<hipfftReal*>(out_data)));
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
    auto signal_numel = c10::multiply_integers(checked_signal_sizes);
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

namespace {
constexpr int64_t cufft_max_ndim = 3;

// Execute a general fft operation (can be c2c, onesided r2c or onesided c2r)
static Tensor& _exec_fft(Tensor& out, const Tensor& self, IntArrayRef out_sizes,
                         IntArrayRef dim, bool forward) {
  const auto ndim = self.dim();
  const int64_t signal_ndim = dim.size();
  const auto batch_dims = ndim - signal_ndim;

  // Permute dimensions so batch dimensions come first, and in stride order
  // This maximizes data locality when collapsing to a single batch dimension
  DimVector dim_permute(ndim);
  std::iota(dim_permute.begin(), dim_permute.end(), int64_t{0});

  c10::SmallVector<bool, kDimVectorStaticSize> is_transformed_dim(ndim);
  for (const auto& d : dim) {
    is_transformed_dim[d] = true;
  }
  auto batch_end = std::partition(dim_permute.begin(), dim_permute.end(),
                                  [&](int64_t d) {return !is_transformed_dim[d]; });
  auto self_strides = self.strides();
  std::sort(dim_permute.begin(), batch_end,
            [&](int64_t a, int64_t b) { return self_strides[a] > self_strides[b]; });
  std::copy(dim.cbegin(), dim.cend(), batch_end);
  auto input = self.permute(dim_permute);

  // Collapse batch dimensions into a single dimension
  DimVector batched_sizes(signal_ndim + 1);
  batched_sizes[0] = -1;
  std::copy(input.sizes().cbegin() + batch_dims, input.sizes().cend(), batched_sizes.begin() + 1);
  input = input.reshape(batched_sizes);

  const auto batch_size = input.sizes()[0];
  DimVector signal_size(signal_ndim + 1);
  signal_size[0] = batch_size;
  for (int64_t i = 0; i < signal_ndim; ++i) {
    auto in_size = input.sizes()[i + 1];
    auto out_size = out_sizes[dim[i]];
    signal_size[i + 1] = std::max(in_size, out_size);
    TORCH_INTERNAL_ASSERT(in_size == signal_size[i + 1] ||
                          in_size == (signal_size[i + 1] / 2) + 1);
    TORCH_INTERNAL_ASSERT(out_size == signal_size[i + 1] ||
                          out_size == (signal_size[i + 1] / 2) + 1);
  }

  batched_sizes[0] = batch_size;
  DimVector batched_out_sizes(batched_sizes.begin(), batched_sizes.end());
  for (size_t i = 0; i < dim.size(); ++i) {
    batched_out_sizes[i + 1] = out_sizes[dim[i]];
  }
  out.resize_(batched_out_sizes, MemoryFormat::Contiguous);

  // Create the transform plan (either from cache or locally)
  const auto value_type = c10::toValueType(input.scalar_type());
  auto fft_type = GetCuFFTTransformType(input.is_complex(), out.is_complex());
  CuFFTParams Params(input.strides(), out.strides(), signal_size, fft_type, value_type);
  CuFFTParamsLRUCache& plan_cache = cufft_get_plan_cache(input.device().index());
  std::unique_lock<std::mutex> guard(plan_cache.mutex, std::defer_lock);
  c10::optional<CuFFTConfig> uncached_plan;
  const CuFFTConfig * config = nullptr;

  if (plan_cache.max_size() > 0) {
    guard.lock();
    if (plan_cache.max_size() > 0) {  // check again after acquiring the lock
      config = &plan_cache.lookup(Params);
    }
  }

  if (config == nullptr) {
    uncached_plan.emplace(Params);
    config = &uncached_plan.value();
  }

  auto & plan = config->plan();

  if (config->should_clone_input()) {
    input = input.clone(MemoryFormat::Contiguous);
  }

  // prepare cufft for execution
  CUFFT_CHECK(cufftSetStream(plan, at::cuda::getCurrentCUDAStream()));
  auto workspace = at::empty({ config->workspace_size() }, at::device(at::kCUDA).dtype(at::kByte));
  CUFFT_CHECK(cufftSetWorkArea(plan, workspace.data_ptr()));

  // execute transform plan
  exec_cufft_plan(*config, input.data_ptr(), out.data_ptr(), forward);

  // Inplace reshaping to original batch shape and inverting the dimension permutation
  DimVector out_strides(ndim);
  int64_t batch_numel = 1;
  for (int64_t i = batch_dims - 1; i >= 0; --i) {
    out_strides[dim_permute[i]] = batch_numel * out.strides()[0];
    batch_numel *= out_sizes[dim_permute[i]];
  }
  for (int64_t i = batch_dims; i < ndim; ++i) {
    out_strides[dim_permute[i]] = out.strides()[1 + (i - batch_dims)];
  }
  return out.as_strided_(out_sizes, out_strides, out.storage_offset());
}

// Calculates the normalization constant and applies it in-place to self
// sizes is the sizes of a twosided tensor and dims are all transformed dims
double _fft_normalization_scale(int64_t normalization, IntArrayRef sizes, IntArrayRef dims) {
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

const Tensor& _fft_apply_normalization(const Tensor& self, int64_t normalization, IntArrayRef sizes, IntArrayRef dims) {
  auto scale = _fft_normalization_scale(normalization, sizes, dims);
  return (scale == 1.0) ? self : self.mul_(scale);
}

Tensor& _fft_apply_normalization_out(Tensor& out, const Tensor& self, int64_t normalization, IntArrayRef sizes, IntArrayRef dims) {
  auto scale = _fft_normalization_scale(normalization, sizes, dims);
  return at::mul_out(out, self, c10::scalar_to_tensor(scale));
}

}  // namespace (anonymous)

// n-dimensional real to complex FFT
Tensor _fft_r2c_cufft(const Tensor& self, IntArrayRef dim, int64_t normalization, bool onesided) {
  TORCH_CHECK(self.is_floating_point());
  auto input_sizes = self.sizes();
  DimVector onesided_sizes(input_sizes.begin(), input_sizes.end());
  auto last_dim = dim.back();
  auto last_dim_halfsize = (input_sizes[last_dim]) / 2 + 1;
  onesided_sizes[last_dim] = last_dim_halfsize;
  IntArrayRef out_sizes = onesided ? onesided_sizes : input_sizes;

  const auto out_options = self.options().dtype(c10::toComplexType(self.scalar_type()));
  auto output = at::empty(out_sizes, out_options);

  // CuFFT requires real input to be over-aligned, as if it were complex
  const auto complex_size = 2 * self.element_size();
  const bool complex_aligned = (
      reinterpret_cast<std::uintptr_t>(self.data_ptr()) % complex_size == 0);
  auto working_tensor = self;
  if (!complex_aligned) {
    working_tensor = self.movedim(last_dim, -1)
                         .clone(MemoryFormat::Contiguous)
                         .movedim(-1, last_dim);
  }

  // First do the R2C transform on the last dimension
  {
    auto target_sizes = dim.size() == 1 ? out_sizes : onesided_sizes;
    _exec_fft(output, working_tensor, target_sizes, last_dim, /*forward=*/true);
    if (dim.size() > 1) {
      working_tensor = at::empty(out_sizes, out_options);
    }
  }

  // Then any remaining C2C transforms
  DimVector sorted_dims(dim.begin(), dim.end() - 1);
  while (!sorted_dims.empty()) {
    std::swap(output, working_tensor);

    // Resort dimensions every time as _exec_fft re-strides the output
    auto strides = working_tensor.strides();
    std::sort(sorted_dims.begin(), sorted_dims.end(),
              [&](int64_t a, int64_t b) { return strides[a] > strides[b]; });

    const auto max_dims = std::min(static_cast<size_t>(cufft_max_ndim), sorted_dims.size());
    auto last_dims = IntArrayRef(sorted_dims).slice(sorted_dims.size() - max_dims, max_dims);

    // Intermediate results are always onesided
    _exec_fft(output, working_tensor, onesided_sizes, last_dims, /*forward=*/true);
    sorted_dims.resize(sorted_dims.size() - max_dims);
  }

  // Only need to normalize the onesided slice since data in the other half is overwritten
  auto out_slice = output.slice(last_dim, 0, last_dim_halfsize);
  _fft_apply_normalization(out_slice, normalization, input_sizes, dim);

  if (!onesided) {
    if (output.sizes()[last_dim] != out_sizes[last_dim]) {
      working_tensor.resize_(out_sizes, MemoryFormat::Contiguous);
      working_tensor.slice(last_dim, 0, last_dim_halfsize).copy_(output);
      output = std::move(working_tensor);
    }
    at::native::_fft_fill_with_conjugate_symmetry_(output, dim);
  }
  return output;
}

Tensor& _fft_r2c_cufft_out(const Tensor& self, IntArrayRef dim,
                           int64_t normalization, bool onesided, Tensor& out) {
  auto result = _fft_r2c_cufft(self, dim, static_cast<int64_t>(fft_norm_mode::none), /*onesided=*/true);
  if (onesided) {
    return _fft_apply_normalization_out(out, result, normalization, self.sizes(), dim);
  }

  resize_output(out, self.sizes());

  auto last_dim = dim.back();
  auto last_dim_halfsize = result.sizes()[last_dim];
  auto out_slice = out.slice(last_dim, 0, last_dim_halfsize);
  _fft_apply_normalization_out(out_slice, result, normalization, self.sizes(), dim);
  at::native::_fft_fill_with_conjugate_symmetry_(out, dim);
  return out;
}

// n-dimensional complex to real IFFT
Tensor _fft_c2r_cufft(const Tensor& self, IntArrayRef dim, int64_t normalization, int64_t lastdim) {
  TORCH_CHECK(self.is_complex());
  auto in_sizes = self.sizes();
  DimVector out_sizes(in_sizes.begin(), in_sizes.end());
  out_sizes[dim.back()] = lastdim;

  // First complete any C2C transforms
  Tensor temp;
  if (dim.size() > 1) {
    temp = _fft_c2c_cufft(
        self, dim.slice(0, dim.size() - 1),
        static_cast<int64_t>(fft_norm_mode::none), /*forward=*/false);
  } else {
    // Complex to real FFTs may overwrite the input buffer, so must always clone (gh-34551)
    temp = self.clone(MemoryFormat::Contiguous);
  }

  // Finally, do a 1D C2R transform
  // TODO: could transform up to 2 other dims in the same cuFFT operation
  auto output = at::empty(out_sizes, self.options().dtype(c10::toValueType(self.scalar_type())));
  _exec_fft(output, temp, out_sizes, dim.back(), /*forward=*/false);
  return _fft_apply_normalization(output, normalization, out_sizes, dim);
}

Tensor& _fft_c2r_cufft_out(const Tensor& self, IntArrayRef dim,
                           int64_t normalization, int64_t lastdim, Tensor& out) {
  auto result = _fft_c2r_cufft(self, dim, static_cast<int64_t>(fft_norm_mode::none), lastdim);
  return _fft_apply_normalization_out(out, result, normalization, result.sizes(), dim);
}

// n-dimensional complex to complex FFT/IFFT
Tensor _fft_c2c_cufft(const Tensor& self, IntArrayRef dim, int64_t normalization, bool forward) {
  TORCH_CHECK(self.is_complex());
  if (dim.empty()) {
    return self.clone();
  }

  auto out_sizes = self.sizes();
  auto output = at::empty(out_sizes, self.options());

  // Perform any number of C2C transforms
  DimVector sorted_dims(dim.begin(), dim.end());
  auto self_strides = self.strides();
  auto working_tensor = self;
  while (true) {
    // Sort dimensions every time as _exec_fft re-strides the output
    auto strides = working_tensor.strides();
    std::sort(sorted_dims.begin(), sorted_dims.end(),
              [&](int64_t a, int64_t b) { return strides[a] > strides[b]; });

    const auto max_dims = std::min(static_cast<size_t>(cufft_max_ndim), sorted_dims.size());
    auto first_dims = IntArrayRef(sorted_dims).slice(sorted_dims.size() - max_dims, max_dims);

    _exec_fft(output, working_tensor, out_sizes, first_dims, forward);
    sorted_dims.resize(sorted_dims.size() - max_dims);

    if (sorted_dims.empty()) {
      break;
    }

    if (working_tensor.is_same(self)) {
      working_tensor = std::move(output);
      output = at::empty(out_sizes, self.options());
    } else {
      std::swap(output, working_tensor);
    }
  }

  return _fft_apply_normalization(output, normalization, out_sizes, dim);
}

Tensor& _fft_c2c_cufft_out(const Tensor& self, IntArrayRef dim,
                           int64_t normalization, bool forward, Tensor& out) {
  auto result = _fft_c2c_cufft(self, dim, static_cast<int64_t>(fft_norm_mode::none), forward);
  return _fft_apply_normalization_out(out, result, normalization, result.sizes(), dim);
}


}} // at::native
