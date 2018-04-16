#include "ATen/ATen.h"
#include "ATen/Config.h"
#include "ATen/Dispatch.h"
#include "ATen/Utils.h"
#include <ATen/optional.h>
#include "ATen/NativeFunctions.h"
#include "ATen/native/SpectralOpsUtils.h"
#include "ATen/native/cuda/CuFFTUtils.h"

#include "ATen/cuda/AccumulateType.cuh"
#include "ATen/cuda/CUDATensorMethods.cuh"
#include "ATen/cuda/CUDATypeConversion.cuh"

#include <THC/THCDeviceUtils.cuh>
#include <THC/THCNumerics.cuh>
#include <THC/THCTensorMathReduce.cuh>
#include <THC/THCTensorSort.cuh>
#include <THC/THCThrustAllocator.cuh>
#include <THCUNN/THCHalfAutoNumerics.cuh>

#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cmath>

namespace at { namespace native {

__forceinline__
static bool is_pow_of_two(long long int  x) {
  return (x & (x - 1)) == 0;
}

// In real-to-complex transform, cuFFT only fills half of the values due to
// conjugate symmetry. See native/SpectralUtils.h for more details.
// The following structs are used to fill in the other half with symmetry in
// case of real-to-complex transform with onesided=False flag.
// See NOTE [ Fourier Transform Conjugate Symmetry ] in native/SpectralOpsUtils.h.

// counting_iterator => index to fill
struct cnt_to_dst_idx_functor : public thrust::unary_function<int64_t, int64_t>
{
  const int64_t last_dim_size;
  const int64_t last_dim_start_slice;
  const int64_t last_dim_to_fill_size;

  cnt_to_dst_idx_functor(int64_t last_dim_size, int64_t last_dim_start_slice) :
    last_dim_size(last_dim_size), last_dim_start_slice(last_dim_start_slice),
    last_dim_to_fill_size(last_dim_size - last_dim_start_slice) {}

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
  int64_t sizes[5], strides[5];
  const int64_t signal_ndim;
  scalar_t *data;  // device ptr

  dst_idx_to_src_functor(const Tensor& batched_complex_signal)
    : signal_ndim(batched_complex_signal.dim() - 1),
      data(batched_complex_signal.data<scalar_t>()) {
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

  cudaStream_t stream = globalContext().getCurrentCUDAStream();
  auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
  auto policy = thrust::cuda::par(allocator).on(stream);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "_fft_fill_with_conjugate_symmetry_", [&] {
    using cuda_scalar_t = cuda::type<scalar_t>;
    typedef thrust::device_ptr<cuda_scalar_t> device_ptr;
    typedef thrust::counting_iterator<int64_t> counter;
    typedef thrust::transform_iterator<cnt_to_dst_idx_functor, counter> dst_idx_iterator;
    typedef thrust::permutation_iterator<device_ptr, dst_idx_iterator> dst_iterator;
    typedef thrust::transform_iterator<dst_idx_to_src_functor<cuda_scalar_t>, dst_idx_iterator> src_iterator;

    dst_idx_iterator dst_idxs(counter(0), cnt_to_dst_idx_functor(size_last_dim, last_dim_start_slice));

    auto data = device_ptr(input.data<cuda_scalar_t>());
    dst_iterator dsts(data, dst_idxs);
    src_iterator srcs(dst_idxs, dst_idx_to_src_functor<cuda_scalar_t>(input));
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

// cuFFT
// Currently not utilizing multi GPUs so this potentially speed up.
Tensor _fft_cufft(const Tensor& self, int64_t signal_ndim,
                  bool complex_input, bool complex_output,
                  bool inverse, IntList checked_signal_sizes,
                  bool normalized, bool onesided,
                  IntList output_sizes) {
  Tensor input = self;

  // Slice when twosided complex-to-real. This is not necessarily needed because
  // we calculate the inembed. But it will benefit us in certain cases where we
  // clone the input tensor.
  //
  // See NOTE [ cuFFT Embedded Strides ].
  // See NOTE [ Fourier Transform Conjugate Symmetry ] in native/SpectralOpsUtils.h.
  if (complex_input && !complex_output && !onesided) {
    auto onesided_size = infer_ft_real_to_complex_onesided_size(checked_signal_sizes[signal_ndim - 1]);
    input = input.narrow(signal_ndim, 0, onesided_size);
  }

  // signal sizes
  std::vector<long long int> signal_sizes(checked_signal_sizes.begin(),
                                          checked_signal_sizes.end());

  // input batch size
  long long int batch = input.size(0);

  // Since cuFFT has limited non-unit stride support and various constraints, we
  // use a flag to keep track throughout this function to see if we need to
  // input = input.clone();
  bool clone_input = false;

  if (input.type().scalarType() == ScalarType::Half) {
    // cuFFT on half requires compute capability of at least SM_53
    auto dev_prop = at::globalContext().getCurrentDeviceProperties();
    if (dev_prop->major < 5 || (dev_prop->major == 5 && dev_prop->minor < 3)) {
      std::ostringstream ss;
      ss << "cuFFT doesn't support signals of half type with compute "
         << "capability less than SM_53, but the device containing input half "
         << "tensor only has SM_" << dev_prop->major << dev_prop->minor;
      throw std::runtime_error(ss.str());
    }
    for (int64_t i = 0; i < signal_ndim; i--) {
      auto signal_size = checked_signal_sizes[i];
      if (!is_pow_of_two(signal_size)) {
        std::ostringstream ss;
        ss << "cuFFT doesn't support signals of half type with size at any "
           << "dimension that is not a power of two, but got a signal size of "
           << checked_signal_sizes;
        throw std::runtime_error(ss.str());
      }
    }
    // For half, base strides on the real part of real-to-complex and
    // complex-to-real transforms are not supported. Since our output is always
    // contiguous, only need to check real-to-complex case.
    clone_input |= input.stride(signal_ndim) != 1;
  }

  // cuFFT requires input and output data pointers to complex type aligned
  // our allocated output tensor is always 256 bytes aligned so it is fine, but
  // we need to check input tensor to make sure that it is not unaligned, e.g.,
  // from a slicing.
  auto complex_size_bytes = 2 * input.type().elementSizeInBytes();
  clone_input |= reinterpret_cast<std::uintptr_t>(input.data_ptr()) % complex_size_bytes != 0;

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
    // See NOTE [ cuFFT Embedded Strides ].
    clone_input |= (batch > 0 && input.stride(0) % 2 != 0) ||
                    input.stride(signal_ndim) % 2 != 0;
  }

  // Checks if input strides can be viewed as embedded.
  // See NOTE [ cuFFT Embedded Strides ].
  //
  // TODO: Figure out why windows fails to compile
  //         at::optional<std::vector<long long int>> inembed_opt = at::nullopt;
  //       Then move the following to a helper function.
  std::vector<long long int>inembed(signal_ndim);
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
  // See NOTE [ cuFFT Embedded Strides ].

  bool simple_layout = !(!complex_input && complex_output && !onesided) &&  // not twosided R2C
                       (clone_input || input.is_contiguous());              // contiguous
  if (!simple_layout && complex_input && !complex_output) {
    clone_input = true;
    simple_layout = true;
  }

  // clone if needed
  if (clone_input) {
    input = input.clone();
    if (!simple_layout) {
      // If advanced layout, copy new input sizes to inemed
      auto input_size = input.sizes();
      std::copy(input_size.begin() + 1,                // begin of signal dim in input
                input_size.begin() + signal_ndim + 1,  // end of signal dim in input
                inembed.begin());                      // begin of output
    }
  }
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
       << at::toString(input.type().scalarType());
    throw std::runtime_error(ss.str());
  }

  // set output
  auto output = input.type().tensor(output_sizes);

  // create plan
  CufftHandle plan;
  size_t ws_size = 0;
  auto& ctx = at::globalContext();

  // set to current stream
  CUFFT_CHECK(cufftSetStream(plan.get(), ctx.getCurrentCUDAStream()));

  // disable auto allocation of workspace to use THC allocator
  CUFFT_CHECK(cufftSetAutoAllocation(plan.get(), /* autoAllocate */ 0));

  // make plan
  if (simple_layout) {
    // If with unit-stride, we tell cuFFT by setting inembed == onembed == NULL.
    // In such case, cuFFT ignores base_istride, base_ostride, idist, and odist
    // by assuming base_istride = base_ostride = 1.
    //
    // See NOTE [ cuFFT Embedded Strides ].
    CUFFT_CHECK(cufftXtMakePlanMany(plan.get(), signal_ndim, signal_sizes.data(),
      /* inembed */ nullptr, /* base_istride */ 1, /* idist */ 1, itype,
      /* onembed */ nullptr, /* base_ostride */ 1, /* odist */ 1, otype,
      batch, &ws_size, exec_type));
  } else {
    // set idist (stride at batch dim)
    long long int idist = complex_input ? input.stride(0) >> 1 : input.stride(0);
    // Even if batch dimension is one and idist (stride(0)) doesn't matter,
    // cuFFT errors if idist = 0. This is hack to make it succeed.
    if (idist == 0 && batch == 1) {
      idist = 1;
    }
    // set base_istride (stride at innermost dim of signal)
    long long int base_istride = complex_input ? input.stride(signal_ndim) >> 1
                                               : input.stride(signal_ndim);

    // set odist, onembed, base_ostride
    long long int odist = complex_output ? output.stride(0) >> 1 : output.stride(0);
    std::vector<long long int> onembed(output_sizes.data() + 1, output_sizes.data() + signal_ndim + 1);
    long long int base_ostride = 1;

    CUFFT_CHECK(cufftXtMakePlanMany(plan.get(), signal_ndim, signal_sizes.data(),
      inembed.data(), base_istride, idist, itype,
      onembed.data(), base_ostride, odist, otype,
      batch, &ws_size, exec_type));
  }

  auto ws = ctx.getType(at::Backend::CUDA, at::ScalarType::Byte).tensor({ static_cast<int64_t>(ws_size) });
  CUFFT_CHECK(cufftSetWorkArea(plan.get(), ws.data_ptr()));

  // run
  CUFFT_CHECK(cufftXtExec(plan.get(), input.data_ptr(), output.data_ptr(),
    inverse ? CUFFT_INVERSE : CUFFT_FORWARD));

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

}} // at::native
