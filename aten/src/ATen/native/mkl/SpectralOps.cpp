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

const Tensor& _fft_fill_with_conjugate_symmetry_cpu_(const Tensor& input, IntArrayRef dim) {
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

#include <ATen/native/TensorIterator.h>

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
static void _fft_fill_with_conjugate_symmetry_slice(
    const scalar_t * data_in, scalar_t * data_out,
    IntArrayRef signal_half_sizes, IntArrayRef signal_strides,
    int64_t lastdim) {
  // This function simply assigns out[:] = conj(in)
  // in has strides signal_strides and out has the same strides but negated
  // Thus it does conjugated mirroring in all dimensions

  const int64_t signal_ndim = signal_half_sizes.size();
  TORCH_INTERNAL_ASSERT(signal_ndim == signal_strides.size());

  const int64_t firstsize = signal_half_sizes[0];
  const int64_t firststride = signal_strides[0];

  const scalar_t * in_ptr = data_in;
  scalar_t * out_ptr = data_out;
  DimVector iter_index(signal_ndim - 1);
  do {
    if (lastdim == 0) {
      for (int64_t i = 0; i < firstsize; ++i) {
        out_ptr[-i * firststride] = std::conj(in_ptr[i * firststride]);
      }
    } else {
      out_ptr[0] = std::conj(in_ptr[0]);
      for (int64_t i = 1; i < firstsize; ++i) {
        out_ptr[(firstsize - i) * firststride] = std::conj(in_ptr[i * firststride]);
      }
    }

    for (int64_t i = 0; i < iter_index.size(); ++i) {
      if (iter_index[i] == 0 && i + 1 != lastdim) {
        ++iter_index[i];
        in_ptr += signal_strides[i + 1];
        out_ptr += (signal_half_sizes[i + 1] - 1) * signal_strides[i + 1];
        break;
      } else if (iter_index[i] + 1 < signal_half_sizes[i + 1]) {
        ++iter_index[i];
        in_ptr += signal_strides[i + 1];
        out_ptr -= signal_strides[i + 1];
        break;
      } else if (i + 1 == iter_index.size()) {
        return;
      }

      in_ptr -= signal_strides[i + 1] * iter_index[i];
      out_ptr -= signal_strides[i + 1];
      iter_index[i] = 0;
    }
  } while (!iter_index.empty());
}

// input should be a tensor of same size as full (twosided)
// signals, but only contains half (onesided) of the values.
// This function modifies inplace.
void _fft_fill_with_conjugate_symmetry_cpu_(const Tensor& input, IntArrayRef dim_) {
  const auto input_sizes = input.sizes();
  const auto input_strides = input.strides();
  TORCH_CHECK(dim_.size() > 0);
  DimVector dim(dim_.begin(), dim_.end());
  at::maybe_wrap_dims(dim, input_strides.size());

  if (input_sizes[dim.back()] <= 2) {
    return;
  }

  // Small dimensions may be treated as batch dims since they don't get mirrored
  dim.erase(
      std::remove_if(dim.begin(), dim.end(), [&](int64_t dim) {
        return (input_sizes[dim] <= 2);
      }),
      dim.end());

  const int64_t in_data_offset = input_strides[dim.back()];
  const int64_t out_data_offset = input_strides[dim.back()] * (input_sizes[dim.back()] - 1);

  // Sort dims by data stride to maximize data locality when iterating
  DimVector dim_permute(dim.size());
  std::iota(dim_permute.begin(), dim_permute.end(), 0);
  std::sort(dim_permute.begin(), dim_permute.end(),
      [&](auto dim1, auto dim2) {
        return input_strides[dim[dim1]] < input_strides[dim[dim1]];
      });

  DimVector signal_half_sizes(dim.size());
  DimVector signal_strides(dim.size());
  int64_t lastdim = 0;
  for (int64_t i = 0; i < dim.size(); ++i) {
    auto idim = dim[dim_permute[i]];
    signal_strides[i] = input_strides[idim];
    if (dim_permute[i] < dim.size() - 1) {
      signal_half_sizes[i] = input_sizes[idim];
    } else {
      signal_half_sizes[i] = (input_sizes[idim] - 1) / 2;
      lastdim = i;
    }
  }

  const auto numiter = at::prod_intlist(signal_half_sizes);
  const auto grain_size = std::max(int64_t{1}, at::internal::GRAIN_SIZE / numiter);
  if (numiter == 0) {
    return;
  }

  auto iter = TensorIteratorConfig()
      .add_output(input)
      .add_input(input)
      .resize_outputs(false)
      .declare_static_shape(input_sizes, dim)
      .build();

  AT_DISPATCH_COMPLEX_TYPES(input.scalar_type(), "_fft_fill_with_conjugate_symmetry", [&] {
    iter.for_each(
      [&](char** data, const int64_t* strides, int64_t size){
        for (int64_t i = 0; i < size; ++i) {
          scalar_t * data_scalar = reinterpret_cast<scalar_t*>(data[0] + strides[0] * i);
          const scalar_t * in_data = data_scalar + in_data_offset;
          scalar_t * out_data = data_scalar + out_data_offset;
          _fft_fill_with_conjugate_symmetry_slice<scalar_t>(
              in_data, out_data, signal_half_sizes, signal_strides, lastdim);
        }
      }, grain_size);
  });
  return;
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
    DimVector signal_dims(signal_ndim);
    std::iota(signal_dims.begin(), signal_dims.end(), 1);
    auto out_as_complex = at::view_as_complex(output);
    _fft_fill_with_conjugate_symmetry_cpu_(out_as_complex, signal_dims);
  }
  return output;
}

}} // namespace at::native

#endif
