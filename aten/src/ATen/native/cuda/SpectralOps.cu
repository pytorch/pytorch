#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Config.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/native/SpectralOpsUtils.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>


namespace at::native {

// Offset calculator for indexing in Hermitian mirrored order.
// In mirrored dims, maps linear index i to (n - i) % n
template <typename index_t>
struct HermitianSymmetryOffsetCalculator {
  // Strides may be negative (the last transformed dim is mirrored with a
  // negative stride), so the stride/offset type must stay signed even when the
  // divmod index type is unsigned for faster division.
  using stride_t = std::make_signed_t<index_t>;
  using offset_type = std::array<stride_t, 1>;
  using dim_type = std::remove_cv_t<decltype(MAX_DIMS)>;
  dim_type dims;
  at::cuda::detail::IntDivider<index_t> sizes_[MAX_DIMS];
  stride_t strides_[MAX_DIMS];
  uint32_t mirror_dim_;  // bit mask
  static_assert(MAX_DIMS < 32, "Need a bigger mask type");

  HermitianSymmetryOffsetCalculator(
      IntArrayRef sizes, IntArrayRef strides, IntArrayRef dim,
      const int64_t element_size){
    TORCH_INTERNAL_ASSERT(sizes.size() == strides.size());
    TORCH_INTERNAL_ASSERT(sizes.size() <= MAX_DIMS);
    dims = sizes.size();

    using at::cuda::detail::IntDivider;
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
    for (const auto i: c10::irange(dim.size())) {
      mirror_dim_ |= (uint32_t{1} << dim[i]);
    }
  }

  C10_HOST_DEVICE offset_type get(index_t linear_idx) const {
    stride_t offset = 0;

    for (dim_type dim = 0; dim < dims; ++dim) {
      auto divmod = sizes_[dim].divmod(linear_idx);
      linear_idx = divmod.div;

      if ((mirror_dim_ & (uint32_t{1} << dim)) == 0) {
        offset += static_cast<stride_t>(divmod.mod) * strides_[dim];
      } else if (divmod.mod != 0) {
        offset += static_cast<stride_t>(sizes_[dim].divisor - divmod.mod) * strides_[dim];
      }
    }
    offset_type offsets;
    offsets[0] = offset;
    return offsets;
  }
};


// out[:] = conj(in[:]) where in and out ordering is generalized by offset calculators
template <typename scalar_t, typename index_t, typename inp_calc_t, typename out_calc_t>
C10_LAUNCH_BOUNDS_1(cuda::detail::CUDA_NUM_THREADS)
__global__ void _fft_conjugate_copy_kernel(
    int64_t numel, scalar_t * out_data, const scalar_t * in_data,
    inp_calc_t ic, out_calc_t oc) {
  CUDA_KERNEL_LOOP_TYPE(index, numel, index_t) {
    auto in_offset = ic.get(index)[0];
    auto out_offset = oc.get(index)[0];
    out_data[out_offset] = std::conj(in_data[in_offset]);
  }
}

template <typename index_t>
static void _fft_fill_with_conjugate_symmetry_launch(
    ScalarType dtype, IntArrayRef mirror_dims, IntArrayRef signal_half_sizes,
    IntArrayRef in_strides, const void * in_data,
    IntArrayRef out_strides, void * out_data,
    int64_t element_size, int64_t numel) {
  auto* in_strides_ptr = in_strides.data();
  const int ndim = in_strides.size();
  // signed_strides matches the signed offset type in the output calculator: the
  // divmod index type stays unsigned (fast IntDivider) while strides/offsets
  // remain signed, so a negative input stride cannot wrap in the 32-bit path.
  OffsetCalculator<1, index_t, /*signed_strides=*/true> input_offset_calculator(
      ndim, signal_half_sizes.data(), &in_strides_ptr, &element_size);
  HermitianSymmetryOffsetCalculator<index_t> output_offset_calculator(
      signal_half_sizes, out_strides, mirror_dims, element_size);

  AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, dtype, "_fft_fill_with_conjugate_symmetry", [&] {
      using namespace cuda::detail;
      _fft_conjugate_copy_kernel<scalar_t, index_t><<<
        GET_BLOCKS(numel), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            numel,
            static_cast<scalar_t*>(out_data),
            static_cast<const scalar_t*>(in_data),
            input_offset_calculator,
            output_offset_calculator);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
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
  const int ndim = in_strides.size();
  const int64_t element_size = scalarTypeToTypeMeta(dtype).itemsize();
  const auto numel = c10::multiply_integers(signal_half_sizes);

  // Prefer 32-bit index math when the element count and every element offset
  // fit in int32. The offset calculators then use magic-number division rather
  // than full 64-bit integer divmod, which dominates this offset-bound kernel.
  bool use_32bit_indexing = numel <= std::numeric_limits<int32_t>::max();
  if (use_32bit_indexing) {
    int64_t max_in_offset = 0;
    int64_t max_out_offset = 0;
    for (const auto i : c10::irange(ndim)) {
      const auto extent = signal_half_sizes[i] - 1;
      max_in_offset += extent * std::abs(in_strides[i] / element_size);
      max_out_offset += extent * std::abs(out_strides[i] / element_size);
    }
    use_32bit_indexing = max_in_offset <= std::numeric_limits<int32_t>::max() &&
        max_out_offset <= std::numeric_limits<int32_t>::max();
  }

  if (use_32bit_indexing) {
    _fft_fill_with_conjugate_symmetry_launch<uint32_t>(
        dtype, mirror_dims, signal_half_sizes, in_strides, in_data,
        out_strides, out_data, element_size, numel);
  } else {
    _fft_fill_with_conjugate_symmetry_launch<int64_t>(
        dtype, mirror_dims, signal_half_sizes, in_strides, in_data,
        out_strides, out_data, element_size, numel);
  }
}

REGISTER_DISPATCH(fft_fill_with_conjugate_symmetry_stub, &_fft_fill_with_conjugate_symmetry_cuda_)

} // at::native
