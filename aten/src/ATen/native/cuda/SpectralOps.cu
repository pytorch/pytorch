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
  int64_t dims;
  IntDivider<index_t> sizes_[MAX_DIMS];
  index_t strides_[MAX_DIMS];
  uint32_t mirror_dim_;  // bit mask
  static_assert(MAX_DIMS < 32, "Need a bigger mask type");

  HermitianSymmetryOffsetCalculator(
      IntArrayRef sizes, IntArrayRef strides, IntArrayRef dim,
      const int64_t element_size){
    dims = sizes.size();
    TORCH_INTERNAL_ASSERT(sizes.size() == strides.size());
    TORCH_INTERNAL_ASSERT(dims <= MAX_DIMS);

    for (int i = 0; i < MAX_DIMS; ++i) {
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

    for (int dim = 0; dim < dims; ++dim) {
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

namespace {
constexpr int64_t cufft_max_ndim = 3;

// Execute a general fft operation (can be c2c, onesided r2c or onesided c2r)
void _exec_fft(const Tensor& input, const Tensor& output, IntArrayRef dims, bool forward) {
  // Use TensorIterator to coalesce batch dimensions
  auto iter = TensorIteratorConfig()
      .add_output(output)
      .add_input(input)
      .resize_outputs(false)
      .check_all_same_dtype(false)
      .declare_static_shape(input.sizes(), dims)
      .build();

  DimVector in_strides(dims.size() + 1);
  DimVector out_strides(dims.size() + 1);
  DimVector signal_size(dims.size() + 1);

  // Convert batch strides from byte stride to element stride
  const auto in_element_size = iter.element_size(1);
  const auto out_element_size = iter.element_size(0);
  in_strides[0] = iter.strides(1)[0] / in_element_size;
  out_strides[0] = iter.strides(0)[0] / out_element_size;
  const auto batch_size = iter.shape()[0];
  signal_size[0] = batch_size;

  const int64_t signal_ndim = dims.size();
  for (int64_t i = 0; i < signal_ndim; ++i) {
    auto idim = dims[signal_ndim - i - 1];
    in_strides[i + 1] = input.strides()[idim];
    out_strides[i + 1] = output.strides()[idim];

    auto in_size = input.sizes()[idim];
    auto out_size = output.sizes()[idim];
    signal_size[i + 1] = std::max(in_size, out_size);
    TORCH_INTERNAL_ASSERT(in_size == signal_size[i + 1] ||
                          in_size == (signal_size[i + 1] / 2) + 1);
    TORCH_INTERNAL_ASSERT(out_size == signal_size[i + 1] ||
                          out_size == (signal_size[i + 1] / 2) + 1);
  }

  // Create the transform plan (either from cache or locally)
  auto fft_type = GetCuFFTTransformType(input.is_complex(), output.is_complex());
  CuFFTParams Params(in_strides, out_strides, signal_size, fft_type,
                     c10::toValueType(input.scalar_type()));
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

  // dims should always be chosen so a valid embedding is possible
  TORCH_INTERNAL_ASSERT(!config->should_clone_input());

  // prepare cufft for execution
  CUFFT_CHECK(cufftSetStream(plan, at::cuda::getCurrentCUDAStream()));
  auto workspace = at::empty({ config->workspace_size() }, at::device(at::kCUDA).dtype(at::kByte));
  CUFFT_CHECK(cufftSetWorkArea(plan, workspace.data_ptr()));

  // execute transform plan (potentially many times)
  iter.serial_for_each([&](char** data, const int64_t* strides, int64_t size) {
        // Execute one batch of FFTs
        TORCH_INTERNAL_ASSERT(size == batch_size);
        TORCH_INTERNAL_ASSERT(strides[0] == out_strides[0] * out_element_size);
        TORCH_INTERNAL_ASSERT(strides[1] == in_strides[0] * in_element_size);
        exec_cufft_plan(*config, data[1], data[0], forward);
      }, {0, iter.numel()});
}

// Returns n s.t. the first n sorted_dims can be represented as a
// cufft embedding. See NOTE [ cuFFT Embedded Strides ]
int64_t num_embeddable_dims(IntArrayRef strides, IntArrayRef sorted_dims) {
  const int64_t max_dims = std::min(
      cufft_max_ndim, static_cast<int64_t>(sorted_dims.size()));
  for (int64_t i = 1; i < max_dims; ++i) {
    auto dim = sorted_dims[i];
    auto dim_prev = sorted_dims[i - 1];
    if (strides[dim] % strides[dim_prev] != 0) {
      return i;
    }
  }
  return max_dims;
}

// Calculates the normalization constant and applies it in-place to self
// sizes is the sizes of a twosided tensor and dims are all transformed dims
void _fft_apply_normalization(const Tensor& self, int64_t normalization, IntArrayRef sizes, IntArrayRef dims) {
  auto norm = static_cast<fft_norm_mode>(normalization);
  if (norm == fft_norm_mode::none) {
    return;
  }

  int64_t signal_numel = 1;
  for (auto dim : dims) {
    signal_numel *= sizes[dim];
  }
  const double scale_denom = (norm == fft_norm_mode::by_root_n) ?
    std::sqrt(signal_numel) : static_cast<double>(signal_numel);
  self.div_(scale_denom);
}

}  // namespace (anonymous)

Tensor _fft_r2c_cufft(const Tensor& self, IntArrayRef dim, int64_t normalization, bool onesided) {
  TORCH_CHECK(self.is_floating_point());
  auto input_sizes = self.sizes();
  DimVector out_sizes(input_sizes.begin(), input_sizes.end());
  auto last_dim = dim.back();
  auto last_dim_halfsize = (input_sizes[last_dim]) / 2 + 1;
  if (onesided) {
    out_sizes[last_dim] = last_dim_halfsize;
  }

  auto input_strides = self.strides();
  bool has_broadcasted_dims = false;
  for (int64_t i = 0; i < input_strides.size(); ++i) {
    if (input_strides[i] == 0 && input_sizes[i] > 1) {
      has_broadcasted_dims = true;
    }
  }

  auto input = self;
  if (has_broadcasted_dims) {
    input = self.clone(MemoryFormat::Contiguous);
  }

  auto output = at::empty(out_sizes, self.options().dtype(c10::toComplexType(self.scalar_type())));
  // Onesided slice view of output
  auto out_slice = output.slice(last_dim, 0, last_dim_halfsize);

  // Do a 1D R2C transform first
  // TODO: could transform up to 2 other dims in the same cuFFT operation
  _exec_fft(input, out_slice, dim.back(), /*forward=*/true);

  // Any subsequent C2C transforms are done in-place
  // Sort dimensions to make valid embeddings
  DimVector sorted_dims(dim.begin(), dim.end() - 1);
  auto output_strides = output.strides();
  std::sort(sorted_dims.begin(), sorted_dims.end(),
            [&](int64_t a, int64_t b) { return output_strides[a] < output_strides[b]; });

  IntArrayRef remaining_dims = sorted_dims;

  while (!remaining_dims.empty()) {
    auto ndim = num_embeddable_dims(output_strides, remaining_dims);
    _exec_fft(out_slice, out_slice, remaining_dims.slice(0, ndim), /*forward=*/true);
    remaining_dims = remaining_dims.slice(ndim);
  }

  _fft_apply_normalization(out_slice, normalization, input.sizes(), dim);

  if (!onesided) {
    at::native::_fft_fill_with_conjugate_symmetry_(output, dim);
  }
  return output;
}

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
        static_cast<int64_t>(fft_norm_mode::none), false);
  } else {
    // Complex to real FFTs may overwrite the input buffer, so must always clone (gh-34551)
    temp = self.clone(MemoryFormat::Contiguous);
  }

  // Finally, do a 1D C2R transform
  // TODO: could transform up to 2 other dims in the same cuFFT operation
  auto output = at::empty(out_sizes, self.options().dtype(c10::toValueType(self.scalar_type())));
  _exec_fft(temp, output, dim.back(), /*forward=*/false);

  _fft_apply_normalization(output, normalization, out_sizes, dim);
  return output;
}

Tensor _fft_c2c_cufft(const Tensor& self, IntArrayRef dims_, int64_t normalization, bool forward) {
  TORCH_CHECK(self.is_complex());
  if (dims_.empty()) {
    return self.clone();
  }

  Tensor input = self;
  auto input_strides = input.strides();
  auto input_sizes = input.sizes();
  bool has_broadcasted_dims = false;
  for (int64_t i = 0; i < input_strides.size(); ++i) {
    if (input_strides[i] == 0 && input_sizes[i] > 1) {
      has_broadcasted_dims = true;
    }
  }

  // Broadcasted dims cannot be represented as a cuFFT embedding so must clone
  Tensor output;
  if (has_broadcasted_dims) {
    input = input.clone(MemoryFormat::Contiguous);
    output = input;
  } else {
    output = at::empty_like(input);
  }

  DimVector sorted_dims(dims_.begin(), dims_.end());
  std::sort(sorted_dims.begin(), sorted_dims.end(),
            [&](int64_t a, int64_t b) { return input_strides[a] < input_strides[b]; });

  auto output_strides = output.strides();
  // First pass may be out-of-place, can only transform dimensions
  // that can be embedded in both input and output
  if (!input.is_same(output)) {
    const int64_t transform_ndims = std::min(num_embeddable_dims(input_strides, sorted_dims),
                                             num_embeddable_dims(output_strides, sorted_dims));
    _exec_fft(input, output, IntArrayRef{sorted_dims}.slice(0, transform_ndims), forward);
    sorted_dims.erase(sorted_dims.begin(), sorted_dims.begin() + transform_ndims);
  }

  // Any subsequent passes are done in-place in the output
  std::sort(sorted_dims.begin(), sorted_dims.end(),
            [&](int64_t a, int64_t b) { return output_strides[a] < output_strides[b]; });
  IntArrayRef dims = sorted_dims;


  // Any subsequent passes are in-place
  while (!dims.empty()) {
    auto ndim = num_embeddable_dims(output_strides, dims);
    _exec_fft(output, output, dims.slice(0, ndim), forward);
    dims = dims.slice(ndim);
  }

  _fft_apply_normalization(output, normalization, output.sizes(), dims_);
  return output;
}

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
