#include "ATen/ATen.h"
#include "ATen/Config.h"
#include "ATen/Dispatch.h"
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
#include <numeric>
#include <iostream>

namespace at { namespace native {

__forceinline__
static bool is_pow_of_two(long long int  x) {
  return (x & (x - 1)) == 0;
}

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

// input should be a contiguous batched tensor of full signals (i.e. twosided)
static inline void _fft_fill_with_hermitian_symmetry_(Tensor& input,
                      int64_t size_last_dim, int64_t last_dim_start_slice) {
  if (last_dim_start_slice >= size_last_dim) {
    return;
  }

  // copy
  int64_t n = input.numel() / size_last_dim * (size_last_dim - last_dim_start_slice);

  cudaStream_t stream = globalContext().getCurrentCUDAStream();
  auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
  auto policy = thrust::cuda::par(allocator).on(stream);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "_fft_fill_with_hermitian_symmetry_", [&] {
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

// cufft
Tensor _fft_cufft(const Tensor& self, int64_t signal_ndim,
                  bool complex_input, bool complex_output,
                  bool inverse, IntList checked_signal_sizes,
                  bool normalized, bool onesided,
                  IntList output_sizes) {
  Tensor input = self;

  bool is_half = input.type().scalarType() == ScalarType::Half;

  // cuFFT requires input and output data pointers to complex type aligned
  // our allocated output tensor is always 256 bytes aligned so it is fine, but
  // we need to check input tensor to make sure that it is not unaligned, e.g.
  // from a slicing.
  auto complex_size_bytes = 2 * input.type().elementSizeInBytes();
  if (reinterpret_cast<std::uintptr_t>(input.data_ptr()) % complex_size_bytes != 0) {
    input = self.clone();
  }

  // check input batch size
  long long int batch = input.size(0);

  // check the input sizes and strides to see if we need to make it contiguous
  std::vector<long long int> signal_sizes(signal_ndim);
  std::vector<long long int> inembeds(signal_ndim);
  long long int istride, signal_size;
  bool need_contiguous = input.stride(signal_ndim) == 0;
  if (complex_input) {
    // real/imag dimension must be like complex type
    need_contiguous |= input.stride(-1) != 1;
  } else if (is_half) {
    // For half, base strides on the real part of real-to-complex and
    // complex-to-real transforms are not supported. Since our output is always
    // contiguous, only need to check real-to-complex case.
    need_contiguous |= input.stride(signal_ndim) != 1;
  }
  // store last tensor stride to infer inembeds array
  // complex input's last dim is size 2 and contiguous
  long long int ilast_stride = complex_input ? 2 : 1;
   // for each signal dim from innermost to outermost
  for (int64_t i = signal_ndim - 1; i >= 0; i--) {
    signal_size = checked_signal_sizes[i];
    istride = input.stride(i + 1);
    if (is_half && !is_pow_of_two(signal_size)) {
      std::ostringstream ss;
      ss << "cuFFT doesn't support signals of half type with size that is not "
         << "a power of two, but gets a signal size of " << signal_size << " at"
         << "signal dimension " << i;
      throw std::runtime_error(ss.str());
    }
    signal_sizes[i] = signal_size;
    if (!need_contiguous) {
      // set the inembeds for dim in last dimension
      if (ilast_stride == 0) {
        need_contiguous = istride != 0;
        if (i < signal_ndim - 1) {
          inembeds[i + 1] = 0;
        }
      } else {
        need_contiguous = istride % ilast_stride != 0;
        if (i < signal_ndim - 1) {
          inembeds[i + 1] = istride / ilast_stride;
        }
      }
      ilast_stride = istride;
    }
  }

  need_contiguous |= (input.size(0) > 1 && input.stride(0) == 0);
  if (need_contiguous) {
    input = input.contiguous();
    auto input_sizes = input.sizes();
    std::copy(&input_sizes[1], &input_sizes[signal_ndim] + 1, &inembeds[0]);
  }

  // set base_istride (stride at innermost dim of signal)
  // set idist (stride at batch dim)
  long long int idist = complex_input ? input.stride(0) >> 1 : input.stride(0);
  // Even if batch dimension is one, then idist (stride(0)) doesn't matter,
  // cuFFT errors if idist = 0. This is hack to make it succeed.
  if (idist == 0 && batch == 1) {
    idist = 1;
  }
  long long int base_istride = complex_input ? input.stride(signal_ndim) >> 1 : input.stride(signal_ndim);

  // output, base_ostride, odist
  auto output = input.type().tensor(output_sizes);
  std::vector<long long int> onembeds(&output_sizes[1], &output_sizes[signal_ndim] + 1);
  long long int odist = complex_output ? output.stride(0) >> 1 : output.stride(0);
  long long int base_ostride = 1;

  CufftHandle plan;
  size_t ws = 0;
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

  CUFFT_CHECK(cufftXtMakePlanMany(plan.get(), signal_ndim, &signal_sizes[0], &inembeds[0],
    base_istride, idist, itype, &onembeds[0], base_ostride, odist, otype, batch,
    &ws, exec_type));

  CUFFT_CHECK(cufftXtExec(plan.get(), input.data_ptr(), output.data_ptr(),
    inverse ? CUFFT_INVERSE : CUFFT_FORWARD));

  auto size_last_signal_dim = checked_signal_sizes[signal_ndim - 1];
  if (normalized || inverse) {
    auto signal_numel = std::accumulate(checked_signal_sizes.begin(), checked_signal_sizes.end(), 1, std::multiplies<int64_t>());
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
  // now if needed, fill out the other half using Hermitian symmetry dim
  if (!complex_input && complex_output && !onesided) {
    auto start_slice = infer_ft_real_to_complex_onesided_size(size_last_signal_dim);
    _fft_fill_with_hermitian_symmetry_(output, size_last_signal_dim, start_slice);
  }
  return output;
}

}} // at::native
