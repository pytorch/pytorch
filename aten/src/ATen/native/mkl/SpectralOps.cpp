#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/SpectralOpsUtils.h>
#include <ATen/Config.h>

#if !AT_MKL_ENABLED()

namespace at { namespace native {

Tensor _fft_mkl(const Tensor& input, int64_t signal_ndim,
                bool complex_input, bool complex_output,
                bool inverse, IntArrayRef checked_signal_sizes,
                int64_t normalization, bool onesided,
                IntArrayRef output_sizes) {
  AT_ERROR("fft: ATen not compiled with MKL support");
}

}}

#else // AT_MKL_ENABLED

#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/Utils.h>

#include <algorithm>
#include <vector>
#include <numeric>
#include <cmath>

#include <mkl_dfti.h>
#include <ATen/mkl/Exceptions.h>
#include <ATen/mkl/Descriptors.h>
#include <ATen/mkl/Limits.h>


namespace at { namespace native {

// In real-to-complex transform, MKL FFT only fills half of the values due to
// conjugate symmetry. See native/SpectralUtils.h for more details.
// The following structs are used to fill in the other half with symmetry in
// case of real-to-complex transform with onesided=False flag.
// See NOTE [ Fourier Transform Conjugate Symmetry ] in native/SpectralOpsUtils.h.

template <typename scalar_t>
static inline void _fft_fill_with_conjugate_symmetry_slice(Tensor& output,
                       int64_t signal_ndim, int64_t size_last_dim,
                       int64_t start_last_dim_idx, int64_t i, int64_t num) {
  scalar_t *data = output.data_ptr<scalar_t>();

  // A slice means a slice of last dimension (of size size_last_dim)

  // This function iterates through the slices to fill, i.e. to_slice_data
  // (basically data_slices[i:i+num]), and keeps track of the slices it reads
  // data from, i.e., from_slice_data, using from_slice_indices, a vector
  // containing the index of the from_slice_data slice.

  // Compute the indices for the first from_slice_data
  std::vector<int64_t> from_slice_indices(signal_ndim);  // up to before last signal dim
  int64_t remainder = i;
  // set last signal dim values
  int64_t from_slice_offset = 0;
  for (int64_t d = signal_ndim - 1; d >= 0; d--) {
    int64_t dim_size = output.size(d);
    int64_t dim_idx = remainder % dim_size;
    remainder = remainder / dim_size;
    from_slice_indices[d] = dim_idx;
    if (d == 0) {
      from_slice_offset += dim_idx * output.stride(d);
    } else if (dim_idx != 0) {
      from_slice_offset += (dim_size - dim_idx) * output.stride(d);
    }
  }

  // First to_slice_data and from_slice_data
  scalar_t *to_slice_data = data + i * size_last_dim * 2;
  scalar_t *from_slice_data = data + from_slice_offset;

  while (num > 0) {
    // Fill to_slice_data from values in from_slice_data
    for (int64_t j = start_last_dim_idx; j < size_last_dim; j++) {
      // multiply index by 2 because of the last complex dim has size 2
      int64_t to_idx = j * 2;
      int64_t from_idx = (size_last_dim - j) * 2;
      to_slice_data[to_idx] = from_slice_data[from_idx];
      to_slice_data[to_idx + 1] = -from_slice_data[from_idx + 1];
    }
    // Compute the next to_slice_data and from_slice_data slices
    to_slice_data += size_last_dim * 2;
    for (int64_t d = signal_ndim - 1; d >= 0; d--) {
      // Compute the next index at this dimension using conjugate symmetry
      // Break out of this loop if nothing carries over
      from_slice_indices[d] = (from_slice_indices[d] + 1) % output.size(d);
      if (d > 0) {
        // At d > 0 nonbatch dim, to get next from_slice_data offset
        //   1. if this dim idx becomes 1, will need to add (size - 1) * stride
        //   2. otherwise, will need to subtract stride
        if (from_slice_indices[d] == 0) {
          // Subtract. Carries over to previous dimension
          from_slice_data -= output.stride(d);
        } else if (from_slice_indices[d] == 1) {
          // Dimension index becomes 1
          // Doesn't carry over to previous dimension
          from_slice_data += (output.size(d) - 1) * output.stride(d);
          break;
        } else {
          // Subtract. Doesn't carry over to previous dimension
          from_slice_data -= output.stride(d);
          break;
        }
      } else {
        // At d = 0 nonbatch dim, it means that to_slice_data ise now at a the
        // beginning of a data sample. It maps to itself by conjugate symmetry.
        from_slice_data = to_slice_data;
      }
    }
    num--;
  }
}

// input should be a contiguous batched tensor of same size as full (twosided)
// signals, but only contains half (onesided) of the values.
// This function modifies inplace.
static inline void _fft_fill_with_conjugate_symmetry_(Tensor& input,
                      int64_t signal_ndim, int64_t size_last_dim,
                      int64_t last_dim_start_slice) {
  if (last_dim_start_slice >= size_last_dim) {
    return;
  }

  int64_t num = 1;
  for (int64_t d = 0; d < signal_ndim; d++) {
    num *= input.size(d);
  }

  at::parallel_for(0, num, 500, [&](int64_t start, int64_t end) {
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "_fft_fill_with_conjugate_symmetry", [&] {
      _fft_fill_with_conjugate_symmetry_slice<scalar_t>(input, signal_ndim, size_last_dim,
          last_dim_start_slice, start, (end - start));
    });
  });
}

// MKL DFTI
Tensor _fft_mkl(const Tensor& self, int64_t signal_ndim,
                bool complex_input, bool complex_output,
                bool inverse, IntArrayRef checked_signal_sizes,
                int64_t normalization, bool onesided,
                IntArrayRef output_sizes) {
  int64_t batch = self.size(0);
  Tensor input = self;
  // real/imag dimension must aligned when viewed as of complex type
  if (complex_input) {
    bool need_contiguous = input.stride(-1) != 1;
    for (int64_t i = 0; !need_contiguous && i <= signal_ndim; i++) {
      need_contiguous |= input.stride(i) % 2 != 0;
    }
    if (need_contiguous) {
      input = input.contiguous();
    }
  }

  // check if we can use MKL because MKL_LONG is 32bit on some OS, e.g. Windows
  // need to check input and output size and strides
  // be careful about complex domain, where the stride needs to be divided by 2
  // only need to test upper bound MKL_LONG_MAX as these values are non-negative
  if (sizeof(MKL_LONG) < sizeof(int64_t)) {
    bool need_contiguous = false;
    int64_t inumel = 1 /* istride if we contiguous-fy */, onumel = 1;
    int64_t isize, osize, istride, ostride;
    for (int64_t i = signal_ndim; i >= 0; i--) {
      isize = input.size(i);
      osize = output_sizes[i];
      istride = complex_input ? input.stride(i) >> 1 : input.stride(i);
      ostride = onumel;
      TORCH_CHECK(isize <= MKL_LONG_MAX && osize <= MKL_LONG_MAX && ostride <= MKL_LONG_MAX,
               "MKL FFT: input signal numel exceeds allowed range [1 ~ ", MKL_LONG_MAX, "]");
      if (!need_contiguous && istride > MKL_LONG_MAX) {
        // If we didn't plan to contiguous-fy but the `istride` exceeds bound,
        // check if we can stride (equal to `inumel`) get back within bound if
        // we contiguous-fy. If so, then we need to always check `inumel`
        // instead for the remaining iterations. The iterations before this are
        // fine as `inumel` is non-decreasing.
        need_contiguous = true;
      }
      TORCH_CHECK(!need_contiguous || inumel <= MKL_LONG_MAX,
               "MKL FFT: input signal numel exceeds allowed range [1 ~ ", MKL_LONG_MAX, "]");
      inumel *= isize;
      onumel *= osize;
    }
  }
  Tensor output = at::empty(output_sizes, input.options());

  // precision
  DFTI_CONFIG_VALUE prec;
  if (input.scalar_type() == ScalarType::Float) {
    prec = DFTI_SINGLE;
  } else if (input.scalar_type() == ScalarType::Double) {
    prec = DFTI_DOUBLE;
  } else {
    std::ostringstream ss;
    ss << "MKL FFT doesn't support tensor of type: "
       << toString(input.scalar_type());
    AT_ERROR(ss.str());
  }
  // signal type
  DFTI_CONFIG_VALUE signal_type;
  if (!inverse) {
    signal_type = complex_input ? DFTI_COMPLEX : DFTI_REAL;
  } else {
    signal_type = complex_output ? DFTI_COMPLEX : DFTI_REAL;
  }
  // create descriptor with signal size
  std::vector<MKL_LONG> mkl_signal_sizes(checked_signal_sizes.begin(), checked_signal_sizes.end());
  DftiDescriptor descriptor;
  descriptor.init(prec, signal_type, signal_ndim, mkl_signal_sizes.data());
  // out of place FFT
  MKL_DFTI_CHECK(DftiSetValue(descriptor.get(), DFTI_PLACEMENT, DFTI_NOT_INPLACE));
  // batch mode
  MKL_DFTI_CHECK(DftiSetValue(descriptor.get(), DFTI_NUMBER_OF_TRANSFORMS, batch));

  auto istrides = input.strides();
  auto ostrides = output.strides();
  // batch dim stride, i.e., dist between each data
  MKL_LONG idist = complex_input ? istrides[0] >> 1 : istrides[0];
  MKL_LONG odist = complex_output ? ostrides[0] >> 1 : ostrides[0];
  MKL_DFTI_CHECK(DftiSetValue(descriptor.get(), DFTI_INPUT_DISTANCE, idist));
  MKL_DFTI_CHECK(DftiSetValue(descriptor.get(), DFTI_OUTPUT_DISTANCE, odist));
  // signal strides
  // first val is offset, set to zero (ignored)
  std::vector<MKL_LONG> mkl_istrides(1 + signal_ndim, 0), mkl_ostrides(1 + signal_ndim, 0);
  for (int64_t i = 1; i <= signal_ndim; i++) {
    mkl_istrides[i] = complex_input ? istrides[i] >> 1 : istrides[i];
    mkl_ostrides[i] = complex_output ? ostrides[i] >> 1 : ostrides[i];
  }
  MKL_DFTI_CHECK(DftiSetValue(descriptor.get(), DFTI_INPUT_STRIDES, mkl_istrides.data()));
  MKL_DFTI_CHECK(DftiSetValue(descriptor.get(), DFTI_OUTPUT_STRIDES, mkl_ostrides.data()));
  // if conjugate domain of real is involved, set standard CCE storage type
  // this will become default in MKL in future
  if (!complex_input || !complex_output) {
    MKL_DFTI_CHECK(DftiSetValue(descriptor.get(), DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX));
  }
  // rescale if requested
  const auto norm = static_cast<fft_norm_mode>(normalization);
  if (norm != fft_norm_mode::none) {
    auto signal_numel = at::prod_intlist(checked_signal_sizes);
    double double_scale;
    if (norm == fft_norm_mode::by_root_n) {
      double_scale = 1.0 / std::sqrt(static_cast<double>(signal_numel));
    } else {
      double_scale = 1.0 / static_cast<double>(signal_numel);
    }
    MKL_DFTI_CHECK(DftiSetValue(descriptor.get(),
      inverse ? DFTI_BACKWARD_SCALE : DFTI_FORWARD_SCALE,
      prec == DFTI_DOUBLE ? double_scale : static_cast<float>(double_scale)));
  }
  // finalize
  MKL_DFTI_CHECK(DftiCommitDescriptor(descriptor.get()));
  // run
  if (!inverse) {
    MKL_DFTI_CHECK(DftiComputeForward(descriptor.get(), input.data_ptr(), output.data_ptr()));
  } else {
    MKL_DFTI_CHECK(DftiComputeBackward(descriptor.get(), input.data_ptr(), output.data_ptr()));
  }
  // now if needed, fill out the other half using Hermitian symmetry dim
  if (!complex_input && complex_output && !onesided) {
    auto size_last_signal_dim = checked_signal_sizes[signal_ndim - 1];
    auto start_slice = infer_ft_real_to_complex_onesided_size(size_last_signal_dim);
    _fft_fill_with_conjugate_symmetry_(output, signal_ndim, size_last_signal_dim, start_slice);
  }
  return output;
}

}} // namespace at::native

#endif
