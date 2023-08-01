#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Config.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/native/SpectralOpsUtils.h>

#include <cmath>
#include <vector>


namespace at::native {

// Offset calculator for indexing in Hermitian mirrored order.
// In mirrored dims, maps linear index i to (n - i) % n
template <typename index_t>
struct HermitianSymmetryOffsetCalculator {
  using offset_type = at::detail::Array<index_t, 1>;
  using dim_type = std::remove_cv_t<decltype(MAX_DIMS)>;
  dim_type dims;
  at::cuda::detail::IntDivider<index_t> sizes_[MAX_DIMS];
  index_t strides_[MAX_DIMS];
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
  AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, dtype, "_fft_fill_with_conjugate_symmetry", [&] {
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

} // at::native
