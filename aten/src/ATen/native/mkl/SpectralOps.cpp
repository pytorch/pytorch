#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/native/Resize.h>
#include <ATen/native/SpectralOpsUtils.h>
#include <ATen/NativeFunctions.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>

#if AT_MKL_ENABLED() || AT_POCKETFFT_ENABLED()
#include <ATen/Parallel.h>

namespace at { namespace native {
// In real-to-complex transform, MKL FFT only fills half of the values due to
// conjugate symmetry. See native/SpectralUtils.h for more details.
// The following structs are used to fill in the other half with symmetry in
// case of real-to-complex transform with onesided=False flag.
// See NOTE [ Fourier Transform Conjugate Symmetry ] in native/SpectralOpsUtils.h.

template <typename scalar_t>
static __ubsan_ignore_undefined__  // UBSAN gives false positives on using negative indexes with a pointer
void _fft_fill_with_conjugate_symmetry_slice(
    Range range, at::ArrayRef<bool> is_mirrored_dim, IntArrayRef signal_half_sizes,
    IntArrayRef in_strides, const scalar_t * in_ptr,
    IntArrayRef out_strides, scalar_t * out_ptr) {
  const auto ndim = signal_half_sizes.size();
  DimVector iter_index(ndim, 0);

  // We explicitly loop over one row, then use this lambda to iterate over
  // n-dimensions. This advances iter_index by one row, while updating in_ptr
  // and out_ptr to point to the new row of data.
  auto advance_index = [&] () __ubsan_ignore_undefined__ {
    for (size_t i = 1; i < iter_index.size(); ++i) {
      if (iter_index[i] + 1 < signal_half_sizes[i]) {
        ++iter_index[i];
        in_ptr += in_strides[i];
        if (is_mirrored_dim[i]) {
          if (iter_index[i] == 1) {
            out_ptr += (signal_half_sizes[i] - 1) * out_strides[i];
          } else {
            out_ptr -= out_strides[i];
          }
        } else {
          out_ptr += out_strides[i];
        }
        return;
      }

      in_ptr -= in_strides[i] * iter_index[i];
      if (is_mirrored_dim[i]) {
        out_ptr -= out_strides[i];
      } else {
        out_ptr -= out_strides[i] * iter_index[i];
      }
      iter_index[i] = 0;
    }
  };

  // The data slice we operate on may start part-way into the data
  // Update iter_index and pointers to reference the start of the slice
  if (range.begin > 0) {
    iter_index[0] = range.begin % signal_half_sizes[0];
    auto linear_idx = range.begin / signal_half_sizes[0];

    for (size_t i = 1; i < ndim && linear_idx > 0; ++i) {
      iter_index[i] = linear_idx % signal_half_sizes[i];
      linear_idx = linear_idx / signal_half_sizes[i];

      if (iter_index[i] > 0) {
        in_ptr += in_strides[i] * iter_index[i];
        if (is_mirrored_dim[i]) {
          out_ptr += out_strides[i] * (signal_half_sizes[i] - iter_index[i]);
        } else {
          out_ptr += out_strides[i] * iter_index[i];
        }
      }
    }
  }

  auto numel_remaining = range.end - range.begin;

  if (is_mirrored_dim[0]) {
    // Explicitly loop over a Hermitian mirrored dimension
    if (iter_index[0] > 0) {
      auto end = std::min(signal_half_sizes[0], iter_index[0] + numel_remaining);
      for (int64_t i = iter_index[0]; i < end; ++i) {
        out_ptr[(signal_half_sizes[0] - i) * out_strides[0]] = std::conj(in_ptr[i * in_strides[0]]);
      }
      numel_remaining -= (end - iter_index[0]);
      iter_index[0] = 0;
      advance_index();
    }

    while (numel_remaining > 0) {
      auto end = std::min(signal_half_sizes[0], numel_remaining);
      out_ptr[0] = std::conj(in_ptr[0]);
      for (int64_t i = 1; i < end; ++i) {
        out_ptr[(signal_half_sizes[0] - i) * out_strides[0]] = std::conj(in_ptr[i * in_strides[0]]);
      }
      numel_remaining -= end;
      advance_index();
    }
  } else {
    // Explicit loop over a non-mirrored dimension, so just a simple conjugated copy
    while (numel_remaining > 0) {
      auto end = std::min(signal_half_sizes[0], iter_index[0] + numel_remaining);
      for (int64_t i = iter_index[0]; i != end; ++i) {
        out_ptr[i * out_strides[0]] = std::conj(in_ptr[i * in_strides[0]]);
      }
      numel_remaining -= (end - iter_index[0]);
      iter_index[0] = 0;
      advance_index();
    }
  }
}

static void _fft_fill_with_conjugate_symmetry_cpu_(
    ScalarType dtype, IntArrayRef mirror_dims, IntArrayRef signal_half_sizes,
    IntArrayRef in_strides_bytes, const void * in_data,
    IntArrayRef out_strides_bytes, void * out_data) {

  // Convert strides from bytes to elements
  const auto element_size = scalarTypeToTypeMeta(dtype).itemsize();
  const auto ndim = signal_half_sizes.size();
  DimVector in_strides(ndim), out_strides(ndim);
  for (const auto i : c10::irange(ndim)) {
    TORCH_INTERNAL_ASSERT(in_strides_bytes[i] % element_size == 0);
    in_strides[i] = in_strides_bytes[i] / element_size;
    TORCH_INTERNAL_ASSERT(out_strides_bytes[i] % element_size == 0);
    out_strides[i] = out_strides_bytes[i] / element_size;
  }

  // Construct boolean mask for mirrored dims
  c10::SmallVector<bool, at::kDimVectorStaticSize> is_mirrored_dim(ndim, false);
  for (const auto& dim : mirror_dims) {
    is_mirrored_dim[dim] = true;
  }

  const auto numel = c10::multiply_integers(signal_half_sizes);
  AT_DISPATCH_COMPLEX_TYPES(dtype, "_fft_fill_with_conjugate_symmetry", [&] {
    at::parallel_for(0, numel, at::internal::GRAIN_SIZE,
        [&](int64_t begin, int64_t end) {
          _fft_fill_with_conjugate_symmetry_slice(
              {begin, end}, is_mirrored_dim, signal_half_sizes,
              in_strides, static_cast<const scalar_t*>(in_data),
              out_strides, static_cast<scalar_t*>(out_data));
        });
  });
}

// Register this one implementation for all cpu types instead of compiling multiple times
REGISTER_ARCH_DISPATCH(fft_fill_with_conjugate_symmetry_stub, DEFAULT, &_fft_fill_with_conjugate_symmetry_cpu_)
REGISTER_AVX_DISPATCH(fft_fill_with_conjugate_symmetry_stub, &_fft_fill_with_conjugate_symmetry_cpu_)
REGISTER_AVX2_DISPATCH(fft_fill_with_conjugate_symmetry_stub, &_fft_fill_with_conjugate_symmetry_cpu_)

// _out variants can be shared between PocketFFT and MKL
Tensor& _fft_r2c_mkl_out(const Tensor& self, IntArrayRef dim, int64_t normalization,
                         bool onesided, Tensor& out) {
  auto result = _fft_r2c_mkl(self, dim, normalization, /*onesided=*/true);
  if (onesided) {
    resize_output(out, result.sizes());
    return out.copy_(result);
  }

  resize_output(out, self.sizes());

  auto last_dim = dim.back();
  auto last_dim_halfsize = result.sizes()[last_dim];
  auto out_slice = out.slice(last_dim, 0, last_dim_halfsize);
  out_slice.copy_(result);
  at::native::_fft_fill_with_conjugate_symmetry_(out, dim);
  return out;
}

Tensor& _fft_c2r_mkl_out(const Tensor& self, IntArrayRef dim, int64_t normalization,
                         int64_t last_dim_size, Tensor& out) {
  auto result = _fft_c2r_mkl(self, dim, normalization, last_dim_size);
  resize_output(out, result.sizes());
  return out.copy_(result);
}

Tensor& _fft_c2c_mkl_out(const Tensor& self, IntArrayRef dim, int64_t normalization,
                         bool forward, Tensor& out) {
  auto result = _fft_c2c_mkl(self, dim, normalization, forward);
  resize_output(out, result.sizes());
  return out.copy_(result);
}

}} // namespace at::native
#endif /* AT_MKL_ENALED() || AT_POCKETFFT_ENABLED() */

#if AT_POCKETFFT_ENABLED()
#include <pocketfft_hdronly.h>

namespace at { namespace native {

namespace {
using namespace pocketfft;

stride_t stride_from_tensor(const Tensor& t) {
  stride_t stride(t.strides().begin(), t.strides().end());
  for(auto& s: stride) {
   s *= t.element_size();
  }
  return stride;
}

inline shape_t shape_from_tensor(const Tensor& t) {
  return shape_t(t.sizes().begin(), t.sizes().end());
}

template<typename T>
inline std::complex<T> *tensor_cdata(Tensor& t) {
  return reinterpret_cast<std::complex<T>*>(t.data<c10::complex<T>>());
}

template<typename T>
inline const std::complex<T> *tensor_cdata(const Tensor& t) {
  return reinterpret_cast<const std::complex<T>*>(t.data<c10::complex<T>>());
}

template<typename T>
T compute_fct(int64_t size, int64_t normalization) {
  constexpr auto one = static_cast<T>(1);
  switch (static_cast<fft_norm_mode>(normalization)) {
    case fft_norm_mode::none: return one;
    case fft_norm_mode::by_n: return one / static_cast<T>(size);
    case fft_norm_mode::by_root_n: return one / std::sqrt(static_cast<T>(size));
  }
  AT_ERROR("Unsupported normalization type", normalization);
}

template<typename T>
T compute_fct(const Tensor& t, IntArrayRef dim, int64_t normalization) {
  if (static_cast<fft_norm_mode>(normalization) == fft_norm_mode::none) {
    return static_cast<T>(1);
  }
  const auto& sizes = t.sizes();
  int64_t n = 1;
  for(auto idx: dim) {
    n *= sizes[idx];
  }
  return compute_fct<T>(n, normalization);
}

} // anonymous namespace

Tensor _fft_c2r_mkl(const Tensor& self, IntArrayRef dim, int64_t normalization, int64_t last_dim_size) {
  auto in_sizes = self.sizes();
  DimVector out_sizes(in_sizes.begin(), in_sizes.end());
  out_sizes[dim.back()] = last_dim_size;
  auto out = at::empty(out_sizes, self.options().dtype(c10::toValueType(self.scalar_type())));
  pocketfft::shape_t axes(dim.begin(), dim.end());
  if (self.scalar_type() == kComplexFloat) {
    pocketfft::c2r(shape_from_tensor(out), stride_from_tensor(self), stride_from_tensor(out), axes, false,
                   tensor_cdata<float>(self),
                   out.data<float>(), compute_fct<float>(out, dim, normalization));
  } else {
    pocketfft::c2r(shape_from_tensor(out), stride_from_tensor(self), stride_from_tensor(out), axes, false,
                   tensor_cdata<double>(self),
                   out.data<double>(), compute_fct<double>(out, dim, normalization));
    }
  return out;
}


Tensor _fft_r2c_mkl(const Tensor& self, IntArrayRef dim, int64_t normalization, bool onesided) {
  TORCH_CHECK(self.is_floating_point());
  auto input_sizes = self.sizes();
  DimVector out_sizes(input_sizes.begin(), input_sizes.end());
  auto last_dim = dim.back();
  auto last_dim_halfsize = (input_sizes[last_dim]) / 2 + 1;
  if (onesided) {
    out_sizes[last_dim] = last_dim_halfsize;
  }

  auto out = at::empty(out_sizes, self.options().dtype(c10::toComplexType(self.scalar_type())));
  pocketfft::shape_t axes(dim.begin(), dim.end());
  if (self.scalar_type() == kFloat) {
    pocketfft::r2c(shape_from_tensor(self), stride_from_tensor(self), stride_from_tensor(out), axes, true,
                   self.data<float>(),
                   tensor_cdata<float>(out), compute_fct<float>(self, dim, normalization));
  } else {
    pocketfft::r2c(shape_from_tensor(self), stride_from_tensor(self), stride_from_tensor(out), axes, true,
                   self.data<double>(),
                   tensor_cdata<double>(out), compute_fct<double>(self, dim, normalization));
  }

  if (!onesided) {
    at::native::_fft_fill_with_conjugate_symmetry_(out, dim);
  }
  return out;
}

Tensor _fft_c2c_mkl(const Tensor& self, IntArrayRef dim, int64_t normalization, bool forward) {
  TORCH_CHECK(self.is_complex());
  auto out = at::empty(self.sizes(), self.options());
  pocketfft::shape_t axes(dim.begin(), dim.end());
  if (self.scalar_type() == kComplexFloat) {
    pocketfft::c2c(shape_from_tensor(self), stride_from_tensor(self), stride_from_tensor(out), axes, forward,
                   tensor_cdata<float>(self),
                   tensor_cdata<float>(out), compute_fct<float>(self, dim, normalization));
  } else {
    pocketfft::c2c(shape_from_tensor(self), stride_from_tensor(self), stride_from_tensor(out), axes, forward,
                   tensor_cdata<double>(self),
                   tensor_cdata<double>(out), compute_fct<double>(self, dim, normalization));
  }

  return out;
}

}}

#elif AT_MKL_ENABLED()
#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
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

// Constructs an mkl-fft plan descriptor representing the desired transform
// For complex types, strides are in units of 2 * element_size(dtype)
// sizes are for the full signal, including batch size and always two-sided
static DftiDescriptor _plan_mkl_fft(
    IntArrayRef in_strides, IntArrayRef out_strides, IntArrayRef sizes,
    bool complex_input, bool complex_output,
    int64_t normalization, bool forward, ScalarType dtype) {
  const int64_t signal_ndim = sizes.size() - 1;
  TORCH_INTERNAL_ASSERT(in_strides.size() == sizes.size());
  TORCH_INTERNAL_ASSERT(out_strides.size() == sizes.size());

  // precision
  const DFTI_CONFIG_VALUE prec = [&]{
    switch (c10::toValueType(dtype)) {
      case ScalarType::Float: return DFTI_SINGLE;
      case ScalarType::Double: return DFTI_DOUBLE;
      default: TORCH_CHECK(false, "MKL FFT doesn't support tensors of type: ", dtype);
    }
  }();
  // signal type
  const DFTI_CONFIG_VALUE signal_type = [&]{
    if (forward) {
      return complex_input ? DFTI_COMPLEX : DFTI_REAL;
    } else {
      return complex_output ? DFTI_COMPLEX : DFTI_REAL;
    }
  }();
  // create descriptor with signal size
  using MklDimVector = c10::SmallVector<MKL_LONG, at::kDimVectorStaticSize>;
  MklDimVector mkl_signal_sizes(sizes.begin() + 1, sizes.end());
  DftiDescriptor descriptor;
  descriptor.init(prec, signal_type, signal_ndim, mkl_signal_sizes.data());
  // out of place FFT
  MKL_DFTI_CHECK(DftiSetValue(descriptor.get(), DFTI_PLACEMENT, DFTI_NOT_INPLACE));
  // batch mode
  MKL_LONG mkl_batch_size = sizes[0];
  MKL_DFTI_CHECK(DftiSetValue(descriptor.get(), DFTI_NUMBER_OF_TRANSFORMS, mkl_batch_size));

  // batch dim stride, i.e., dist between each data
  TORCH_CHECK(in_strides[0] <= MKL_LONG_MAX && out_strides[0] <= MKL_LONG_MAX);
  MKL_LONG idist = in_strides[0];
  MKL_LONG odist = out_strides[0];
  MKL_DFTI_CHECK(DftiSetValue(descriptor.get(), DFTI_INPUT_DISTANCE, idist));
  MKL_DFTI_CHECK(DftiSetValue(descriptor.get(), DFTI_OUTPUT_DISTANCE, odist));

  // signal strides
  // first val is offset, set to zero (ignored)
  MklDimVector mkl_istrides(1 + signal_ndim, 0), mkl_ostrides(1 + signal_ndim, 0);
  for (int64_t i = 1; i <= signal_ndim; i++) {
    TORCH_CHECK(in_strides[i] <= MKL_LONG_MAX && out_strides[i] <= MKL_LONG_MAX);
    mkl_istrides[i] = in_strides[i];
    mkl_ostrides[i] = out_strides[i];
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
  int64_t signal_numel = c10::multiply_integers(IntArrayRef(sizes.data() + 1, signal_ndim));
  if (norm != fft_norm_mode::none) {
    const double scale = (
      (norm == fft_norm_mode::by_root_n) ?
      1.0 / std::sqrt(static_cast<double>(signal_numel)) :
      1.0 / static_cast<double>(signal_numel));
    const auto scale_direction = forward ? DFTI_FORWARD_SCALE : DFTI_BACKWARD_SCALE;
    MKL_DFTI_CHECK(DftiSetValue(descriptor.get(), scale_direction, scale));
  }

  if (sizeof(MKL_LONG) < sizeof(int64_t)) {
    TORCH_CHECK(signal_numel <= MKL_LONG_MAX,
                "MKL FFT: input signal numel exceeds allowed range [1, ", MKL_LONG_MAX, "]");
  }

  // finalize
  MKL_DFTI_CHECK(DftiCommitDescriptor(descriptor.get()));

  return descriptor;
}

// Execute a general fft operation (can be c2c, onesided r2c or onesided c2r)
static Tensor& _exec_fft(Tensor& out, const Tensor& self, IntArrayRef out_sizes,
                         IntArrayRef dim, int64_t normalization, bool forward) {
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

  const auto value_type = c10::toValueType(input.scalar_type());
  out.resize_(batched_out_sizes, MemoryFormat::Contiguous);

  auto descriptor = _plan_mkl_fft(
      input.strides(), out.strides(), signal_size, input.is_complex(),
      out.is_complex(), normalization, forward, value_type);

  // run the FFT
  if (forward) {
    MKL_DFTI_CHECK(DftiComputeForward(descriptor.get(), input.data_ptr(), out.data_ptr()));
  } else {
    MKL_DFTI_CHECK(DftiComputeBackward(descriptor.get(), input.data_ptr(), out.data_ptr()));
  }

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
  out.as_strided_(out_sizes, out_strides, out.storage_offset());
  return out;
}

// Sort transform dimensions by input layout, for best performance
// exclude_last is for onesided transforms where the last dimension cannot be reordered
static DimVector _sort_dims(const Tensor& self, IntArrayRef dim, bool exclude_last=false) {
  DimVector sorted_dims(dim.begin(), dim.end());
  auto self_strides = self.strides();
  std::sort(sorted_dims.begin(), sorted_dims.end() - exclude_last,
            [&](int64_t a, int64_t b) { return self_strides[a] > self_strides[b]; });
  return sorted_dims;
}

// n-dimensional complex to real IFFT
Tensor _fft_c2r_mkl(const Tensor& self, IntArrayRef dim, int64_t normalization, int64_t last_dim_size) {
  TORCH_CHECK(self.is_complex());
  // NOTE: Multi-dimensional C2R transforms don't agree with numpy in cases
  // where the input isn't strictly Hermitian-symmetric. Instead, we use a
  // multi-dim C2C transform followed by a 1D C2R transform.
  //
  // Such inputs are technically out of contract though, so maybe a disagreement
  // is okay.
  auto input = self;
  if (dim.size() > 1) {
    auto c2c_dims = dim.slice(0, dim.size() - 1);
    input = _fft_c2c_mkl(self, c2c_dims, normalization, /*forward=*/false);
    dim = dim.slice(dim.size() - 1);
  }

  auto in_sizes = input.sizes();
  DimVector out_sizes(in_sizes.begin(), in_sizes.end());
  out_sizes[dim.back()] = last_dim_size;
  auto out = at::empty(out_sizes, self.options().dtype(c10::toValueType(self.scalar_type())));
  return _exec_fft(out, input, out_sizes, dim, normalization, /*forward=*/false);
}

// n-dimensional real to complex FFT
Tensor _fft_r2c_mkl(const Tensor& self, IntArrayRef dim, int64_t normalization, bool onesided) {
  TORCH_CHECK(self.is_floating_point());
  auto input_sizes = self.sizes();
  DimVector out_sizes(input_sizes.begin(), input_sizes.end());
  auto last_dim = dim.back();
  auto last_dim_halfsize = (input_sizes[last_dim]) / 2 + 1;
  if (onesided) {
    out_sizes[last_dim] = last_dim_halfsize;
  }

  auto sorted_dims = _sort_dims(self, dim, /*exclude_last=*/true);
  auto out = at::empty(out_sizes, self.options().dtype(c10::toComplexType(self.scalar_type())));
  _exec_fft(out, self, out_sizes, sorted_dims, normalization, /*forward=*/true);

  if (!onesided) {
    at::native::_fft_fill_with_conjugate_symmetry_(out, dim);
  }
  return out;
}

// n-dimensional complex to complex FFT/IFFT
Tensor _fft_c2c_mkl(const Tensor& self, IntArrayRef dim, int64_t normalization, bool forward) {
  TORCH_CHECK(self.is_complex());
  const auto sorted_dims = _sort_dims(self, dim);
  auto out = at::empty(self.sizes(), self.options());
  return _exec_fft(out, self, self.sizes(), sorted_dims, normalization, forward);
}

}} // namespace at::native

#else

namespace at { namespace native {
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_NO_CPU_DISPATCH(fft_fill_with_conjugate_symmetry_stub, fft_fill_with_conjugate_symmetry_fn);

Tensor _fft_c2r_mkl(const Tensor& self, IntArrayRef dim, int64_t normalization, int64_t last_dim_size) {
  AT_ERROR("fft: ATen not compiled with FFT support");
}

Tensor _fft_r2c_mkl(const Tensor& self, IntArrayRef dim, int64_t normalization, bool onesided) {
  AT_ERROR("fft: ATen not compiled with FFT support");
}

Tensor _fft_c2c_mkl(const Tensor& self, IntArrayRef dim, int64_t normalization, bool forward) {
  AT_ERROR("fft: ATen not compiled with FFT support");
}

Tensor& _fft_r2c_mkl_out(const Tensor& self, IntArrayRef dim, int64_t normalization,
                         bool onesided, Tensor& out) {
  AT_ERROR("fft: ATen not compiled with FFT support");
}

Tensor& _fft_c2r_mkl_out(const Tensor& self, IntArrayRef dim, int64_t normalization,
                         int64_t last_dim_size, Tensor& out) {
  AT_ERROR("fft: ATen not compiled with FFT support");
}

Tensor& _fft_c2c_mkl_out(const Tensor& self, IntArrayRef dim, int64_t normalization,
                         bool forward, Tensor& out) {
  AT_ERROR("fft: ATen not compiled with FFT support");
}

}} // namespace at::native
#endif
