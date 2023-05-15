// Basic functions on sparse tensors
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/Layout.h>
#include <ATen/Parallel.h>
#include <ATen/SparseCsrTensorImpl.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/native/LinearAlgebraUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_convert_indices_from_csr_to_coo.h>
#include <ATen/ops/_nnz_native.h>
#include <ATen/ops/_sparse_compressed_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_csr_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_csc_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_bsr_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_bsc_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_coo_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_coo_tensor_unsafe.h>
#include <ATen/ops/_validate_sparse_compressed_tensor_args_native.h>
#include <ATen/ops/_validate_sparse_csr_tensor_args_native.h>
#include <ATen/ops/_validate_sparse_csc_tensor_args_native.h>
#include <ATen/ops/_validate_sparse_bsr_tensor_args_native.h>
#include <ATen/ops/_validate_sparse_bsc_tensor_args_native.h>
#include <ATen/ops/aminmax.h>
#include <ATen/ops/ccol_indices_native.h>
#include <ATen/ops/clone_native.h>
#include <ATen/ops/col_indices_native.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/crow_indices_native.h>
#include <ATen/ops/dense_dim_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like_native.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/resize_as_sparse_native.h>
#include <ATen/ops/resize_native.h>
#include <ATen/ops/row_indices_native.h>
#include <ATen/ops/select_native.h>
#include <ATen/ops/select_copy.h>
#include <ATen/ops/select_copy_native.h>
#include <ATen/ops/sparse_compressed_tensor_native.h>
#include <ATen/ops/sparse_csr_tensor_native.h>
#include <ATen/ops/sparse_csc_tensor_native.h>
#include <ATen/ops/sparse_bsr_tensor_native.h>
#include <ATen/ops/sparse_bsc_tensor_native.h>
#include <ATen/ops/sparse_dim_native.h>
#include <ATen/ops/values_native.h>
#include <ATen/ops/_validate_compressed_sparse_indices.h>
#include <ATen/ops/where.h>
#endif

namespace at::native {

using namespace at::sparse_csr;

namespace {

bool solve_arange(const Tensor& input, int64_t& start, int64_t& end, int64_t& step) {
  /*
    This function solves the equation

      input == arange(start, end, step)

    for integers start, end, and step, if possible. If the solution
    exists, returns true.
  */
  int64_t n = input.numel();
  if (n == 0) {
    // a trivial solution
    start = end = 0;
    step = 1;
  } else if (n == 1) {
    // a simple solution
    start = input[0].item<int64_t>();
    end = start + 1;
    step = 1;
  } else {
    Tensor first_last = input.slice(0, 0, n, n - 1).cpu();
    int64_t start_candidate = first_last[0].item<int64_t>();
    int64_t end_candidate = first_last[1].item<int64_t>() + 1;
    if (end_candidate - start_candidate == n) {
      // a special solution
      start = start_candidate;
      end = end_candidate;
      step = 1;
    } else {
      // detect if general solution exists
      Tensor possible_steps = input.slice(0, 1).sub(input.slice(0, 0, n - 1));
      Tensor possible_step = possible_steps[0];
      if ((possible_steps.eq(possible_step)).all().item<bool>()) {
        start = start_candidate;
        end = end_candidate;
        step = possible_step.item<int64_t>();
      } else {
        // no solution
        return false;
      }
    }
  }
  return true;
}

} // end anonymous namespace

/*
  Validate the arguments to sparse compressed (CSR, CSC, BSR, and BSC)
  tensor factory functions.

  The CSR and BSR invariants for PyTorch are outlined in

    https://pearu.github.io/csr_tensor_invariants.html
    https://pearu.github.io/bsr_tensor_invariants.html

  that in what follows are generalized for all sparse compressed
  formats with support to batched and dense dimensions.
*/

void _validate_sparse_compressed_tensor_args_worker(const Tensor& compressed_indices, const Tensor& plain_indices, const Tensor& values, const IntArrayRef size, const Layout& layout) {
  // Layout must be Sparse Compressed, 2.4
  AT_DISPATCH_ALL_SPARSE_COMPRESSED_LAYOUTS(layout, "validate_sparse_compressed_tensor_args", [&]{});

  const std::string layout_name = layoutToString(layout, /*upper=*/ true);
  const std::string compressed_indices_name = compressedIndicesName(layout);
  const std::string plain_indices_name = plainIndicesName(layout);
  const std::string compressed_dim_name = compressedDimName(layout);
  const std::string plain_dim_name = plainDimName(layout);

  // Layout Invariants

  // Re 3.5 and 3.6: in the case of compressed/plain indices tensors,
  // we require contiguity per-patch basis, that is, the last stride
  // of these indices must be 1. The reasoning for this is that
  // indices tensors within a patch are "atomic" in the sense that
  // sliced compressed/plain indices would not represent the indices
  // of any sparse compressed tensor as the slicing would break the
  // description of the tensor index structure.

  // 2.1
  TORCH_CHECK(plain_indices.layout() == kStrided,
              "expected ", plain_indices_name, " to be a strided tensor but got ", plain_indices.layout(), " tensor");

  // 2.2
  TORCH_CHECK(compressed_indices.layout() == kStrided,
              "expected ", compressed_indices_name, " to be a strided tensor but got ", compressed_indices.layout(), " tensor");

  const int base_ndim = 2;  // corresponds to compressed and plain indices
  const int batch_ndim = compressed_indices.dim() - 1;
  const int block_ndim = AT_DISPATCH_PLAIN_SPARSE_COMPRESSED_LAYOUTS(
                           layout, "validate_sparse_compressed_tensor_args",
                           [&] { return 0; }, [&] { return 2; });
  const int dense_ndim = values.dim() - batch_ndim - block_ndim - 1;

  // 2.3
  TORCH_CHECK(values.layout() == kStrided,
              "expected values to be a strided tensor but got ", values.layout(), " tensor");

  // 3.7 is dropped, that is, values tensor does not need to be
  // contiguous, in general. Particular algorithms on sparse
  // compressed tensors may require contiguity though.

  // Shape and Strides invariants

  // 3.2
  TORCH_CHECK(
              batch_ndim >= 0,
              compressed_indices_name, " must have dimensionality >= 1 but got ", compressed_indices.dim());

  // 3.3
  TORCH_CHECK(
              compressed_indices.dim() == plain_indices.dim(),
              compressed_indices_name, " and ", plain_indices_name, " dimensionalities must be equal but got ",
              compressed_indices.dim(), " and ", plain_indices.dim(), ", respectively");

  // 3.4
  TORCH_CHECK(
              dense_ndim >= 0,
              "values must have dimensionality > sum of batch and block dimensionalities (=",
              batch_ndim, " + ", block_ndim, ") but got ", values.dim());

  // 3.5
  TORCH_CHECK(plain_indices.stride(-1) == 1,
              "expected ", plain_indices_name, " to be a contiguous tensor per batch");

  // 3.6
  TORCH_CHECK(compressed_indices.stride(-1) == 1,
              "expected ", compressed_indices_name, " to be a contiguous tensor per batch");

  // 3.1
  TORCH_CHECK(
              static_cast<int>(size.size()) == batch_ndim + base_ndim + dense_ndim,
              "tensor dimensionality must be sum of batch, base, and dense dimensionalities (=",
              batch_ndim, " + ", base_ndim, " + ", dense_ndim, ") but got ", size.size());

  // For CSR/CSC formats, we define blocksize=(1, 1) so that checking
  // the sparse compressed tensor invariants can be unified with the
  // BSR/BSC invariants.
  // 3.10
  DimVector blocksize{
                      (block_ndim == 2 ? std::max<int64_t>(1, values.size(batch_ndim + 1)) : 1),
                      (block_ndim == 2 ? std::max<int64_t>(1, values.size(batch_ndim + 2)) : 1),
  };
  TORCH_INTERNAL_ASSERT(blocksize.size() == 2 && blocksize[0] > 0 && blocksize[1] > 0);

  // All batch sizes must be the same and consistent with tensor batchsize, 3.1, 3.8, 3.9, 3.10
  DimVector batchsize = DimVector(size.slice(0, batch_ndim));
  DimVector compressed_indices_batchsize = DimVector(compressed_indices.sizes().slice(0, batch_ndim));
  DimVector plain_indices_batchsize = DimVector(plain_indices.sizes().slice(0, batch_ndim));
  DimVector values_batchsize = DimVector(values.sizes().slice(0, batch_ndim));
  const int values_nnz = values.size(batch_ndim);
  DimVector values_blocksize = DimVector(values.sizes().slice(batch_ndim + 1, block_ndim));
  DimVector values_densesize = DimVector(values.sizes().slice(batch_ndim + 1 + block_ndim, dense_ndim));
  TORCH_CHECK(
      batchsize == compressed_indices_batchsize && batchsize == plain_indices_batchsize && batchsize == values_batchsize,
      "all batch dimensions of ", compressed_indices_name," (=", compressed_indices_batchsize, "), ", plain_indices_name," (=",
      plain_indices_batchsize, "), and values (=", values_batchsize, ") must be equal to tensor batch dimensions (=",
      batchsize, ")");

  // A tensor constitutes of full blocks, 3.1
  for (int i=0; i<block_ndim; i++) {
      TORCH_CHECK(size[batch_ndim + i] % blocksize[i] == 0,
                  "tensor shape[", batch_ndim + i, "] (=", size[batch_ndim + i],
                  ") must be divisible with blocksize[", i, "] (=", blocksize[i],
                  ") as defined by values shape");
  }
  const int nrows = size[batch_ndim] / blocksize[0];
  const int ncols = size[batch_ndim + 1] / blocksize[1];
  int compressed_dim_size, plain_dim_size;
  std::tie(compressed_dim_size, plain_dim_size) = AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(layout, "validate_sparse_compressed_tensor_args",
                                                                                            [&] { return std::make_tuple(nrows, ncols); },
                                                                                            [&] { return std::make_tuple(ncols, nrows); });
  // 3.8
  TORCH_CHECK(
              compressed_indices.size(-1) == compressed_dim_size + 1,
              compressed_indices_name, ".shape[-1] must be equal to the number of ",
              compressed_dim_name, "s + 1 (=", compressed_dim_size + 1, "), but got ", compressed_indices.size(-1));
  // 3.9, 3.10
  TORCH_CHECK(
              plain_indices.size(-1) == values_nnz,
              plain_indices_name, ".shape[-1] must be equal to nnz (=", values_nnz,
              ") as defined by values.shape[", batch_ndim, "], but got ", plain_indices.size(-1));
  // Type Invariants
  auto compressed_indices_type = compressed_indices.scalar_type();
  auto plain_indices_type = plain_indices.scalar_type();
  // 1.1, 1.2, 1.3
  TORCH_CHECK(
      compressed_indices_type == plain_indices_type,
      compressed_indices_name, " and ", plain_indices_name, " must have the same dtype, bot got ",
      compressed_indices_type, " and ", plain_indices_type, ", respectively");
  TORCH_CHECK(
      compressed_indices_type == kInt || compressed_indices_type == kLong,
      compressed_indices_name, " and ", plain_indices_name, " dtype must be Int or Long, but got ",
      compressed_indices_type);

  // Indices invariants
  if (plain_indices.numel() > 0) {
    at::_validate_compressed_sparse_indices(
        /*is_crow = */layout == kSparseCsr || layout == kSparseBsr,
        compressed_indices,
        plain_indices,
        compressed_dim_size,
        plain_dim_size,
        values_nnz
    );
  }

  // Device Invariants
  // 4.1
  TORCH_CHECK(
      values.device().type() == kCPU || values.device().type() == kCUDA,
      "device type of values (",
      values.device().type(),
      ") must be CPU or CUDA");
  // 4.2, 4.3, 4.4
  TORCH_CHECK(
      compressed_indices.get_device() == values.get_device(),
      "device of ", compressed_indices_name, " (=",
      compressed_indices.device(),
      ") must match device of values (=",
      values.device(),
      ")");
  TORCH_CHECK(
      compressed_indices.get_device() == plain_indices.get_device(),
      "device of ", compressed_indices_name, " (=",
      compressed_indices.device(),
      ") must match device of ", plain_indices_name," (=",
      plain_indices.device(),
      ")");
}

void _validate_sparse_compressed_tensor_args(const Tensor& compressed_indices, const Tensor& plain_indices, const Tensor& values, IntArrayRef size, Layout layout) {
  _validate_sparse_compressed_tensor_args_worker(compressed_indices, plain_indices, values, size, layout);
}

void _validate_sparse_csr_tensor_args(const Tensor& crow_indices, const Tensor& col_indices, const Tensor& values, IntArrayRef size) {
  _validate_sparse_compressed_tensor_args_worker(crow_indices, col_indices, values, size, kSparseCsr);
}

void _validate_sparse_csc_tensor_args(const Tensor& ccol_indices, const Tensor& row_indices, const Tensor& values, IntArrayRef size) {
  _validate_sparse_compressed_tensor_args_worker(ccol_indices, row_indices, values, size, kSparseCsc);
}

void _validate_sparse_bsr_tensor_args(const Tensor& crow_indices, const Tensor& col_indices, const Tensor& values, IntArrayRef size) {
  _validate_sparse_compressed_tensor_args_worker(crow_indices, col_indices, values, size, kSparseBsr);
}

void _validate_sparse_bsc_tensor_args(const Tensor& ccol_indices, const Tensor& row_indices, const Tensor& values, IntArrayRef size) {
  _validate_sparse_compressed_tensor_args_worker(ccol_indices, row_indices, values, size, kSparseBsc);
}

// Construction of CSR, CSC, BSR, and BSC tensors.

// Note: The usage of "Csr" in names like SparseCsrTensor,
// SparseCsrCPU, SparseCsrCUDA, and SparseCsrTensorImpl exists because
// of historical reasons (that ought to be removed in future) and does
// not mean that the corresponding functionality would be CSR layout
// only specific.
SparseCsrTensor new_compressed_tensor(const TensorOptions& options) {
  // TODO: remove this comment after enabling autograd support for CSR tensor
  // constructor.
  // TORCH_INTERNAL_ASSERT(impl::variable_excluded_from_dispatch());
  Layout layout = AT_DISPATCH_ALL_SPARSE_COMPRESSED_LAYOUTS(options.layout(), "new_compressed_tensor", [&] { return the_layout; });
  DispatchKey dispatch_key;

  TORCH_CHECK_NOT_IMPLEMENTED(
    options.device().type() == kCPU || options.device().type() == kCUDA,
     "Could not run 'new_compressed_tensor' from the '", options.device(), "' device.)");

  if (options.device().is_cuda()) {
    dispatch_key = DispatchKey::SparseCsrCUDA;
  } else {
    dispatch_key = DispatchKey::SparseCsrCPU;
  }

  return detail::make_tensor<SparseCsrTensorImpl>(DispatchKeySet(dispatch_key), options.device(), layout, options.dtype());
}


Tensor _sparse_compressed_tensor_unsafe(const Tensor& compressed_indices,
                                        const Tensor& plain_indices,
                                        const Tensor& values,
                                        IntArrayRef size,
                                        c10::optional<ScalarType> dtype,
                                        c10::optional<Layout> layout,
                                        c10::optional<Device> device,
                                        c10::optional<bool> pin_memory) {
  if (!layout) {
    AT_ERROR("sparse_compressed_tensor_unsafe expected sparse compressed tensor layout but got none");
  }
  Layout layout_ = layout.value();
  AT_DISPATCH_ALL_SPARSE_COMPRESSED_LAYOUTS(layout_, "sparse_compressed_tensor_unsafe", [&]{});
  if (at::globalContext().checkSparseTensorInvariants()) {
    _validate_sparse_compressed_tensor_args_worker(compressed_indices, plain_indices, values, size, layout_);
  }
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout_).device(device).pinned_memory(pin_memory);
  SparseCsrTensor self = new_compressed_tensor(options);
  get_sparse_csr_impl(self)->set_member_tensors(compressed_indices, plain_indices, values, size);
  return self;
}

template <Layout required_layout>
Tensor _sparse_compressed_tensor_unsafe_template(const Tensor& compressed_indices,
                                                 const Tensor& plain_indices,
                                                 const Tensor& values,
                                                 IntArrayRef size,
                                                 c10::optional<ScalarType> dtype,
                                                 c10::optional<Layout> layout,
                                                 c10::optional<Device> device,
                                                 c10::optional<bool> pin_memory) {
  Layout layout_ = layout.value_or(required_layout);
  TORCH_CHECK(layout_ == required_layout, "sparse compressed layout must be ",required_layout, " but got ", layout_);
  if (at::globalContext().checkSparseTensorInvariants()) {
    _validate_sparse_compressed_tensor_args_worker(compressed_indices, plain_indices, values, size, layout_);
  }
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout_).device(device).pinned_memory(pin_memory);
  SparseCsrTensor self = new_compressed_tensor(options);
  get_sparse_csr_impl(self)->set_member_tensors(compressed_indices, plain_indices, values, size);
  return self;
}

#define SPARSE_COMPRESSED_TENSOR_UNSAFE(KIND, REQUIRED_LAYOUT)          \
  Tensor _sparse_##KIND##_tensor_unsafe(const Tensor& compressed_indices, \
                                        const Tensor& plain_indices,    \
                                        const Tensor& values,           \
                                        IntArrayRef size,               \
                                        c10::optional<ScalarType> dtype, \
                                        c10::optional<Layout> layout,   \
                                        c10::optional<Device> device,   \
                                        c10::optional<bool> pin_memory) { \
    return _sparse_compressed_tensor_unsafe_template<REQUIRED_LAYOUT>(compressed_indices, plain_indices, values, size, dtype, layout, device, pin_memory); \
  }

SPARSE_COMPRESSED_TENSOR_UNSAFE(csr, kSparseCsr);
SPARSE_COMPRESSED_TENSOR_UNSAFE(csc, kSparseCsc);
SPARSE_COMPRESSED_TENSOR_UNSAFE(bsr, kSparseBsr);
SPARSE_COMPRESSED_TENSOR_UNSAFE(bsc, kSparseBsc);

DimVector _estimate_sparse_compressed_tensor_size(
    const Tensor& compressed_indices,
    const Tensor& plain_indices,
    const Tensor& values,
    Layout layout) {
  const int block_ndim = AT_DISPATCH_PLAIN_SPARSE_COMPRESSED_LAYOUTS(layout, "estimate_sparse_compressed_tensor_size", [&] { return 0; }, [&] { return 2; });
  const int base_ndim = 2;  // corresponds to compressed and plain indices
  const int batch_ndim = compressed_indices.dim() - 1;
  const std::string compressed_indices_name = compressedIndicesName(layout);
  const std::string plain_indices_name = plainIndicesName(layout);
  TORCH_CHECK(
              batch_ndim >= 0,
              compressed_indices_name, " must have dimensionality >= 1 but got ", compressed_indices.dim());
  TORCH_CHECK(
              compressed_indices.dim() == plain_indices.dim(),
              compressed_indices_name, " and ", plain_indices_name, " dimensionalities must be equal but got ",
              compressed_indices.dim(), " and ", plain_indices.dim(), ", respectively");
  const int dense_ndim = values.dim() - batch_ndim - block_ndim - 1;
  TORCH_CHECK(
              dense_ndim >= 0,
              "values must have dimensionality > sum of batch and block dimensionalities (=",
              batch_ndim, " + ", block_ndim, ") but got ", values.dim());
  DimVector blocksize{
                      (block_ndim == 2 ? std::max<int64_t>(1, values.size(batch_ndim + 1)) : 1),
                      (block_ndim == 2 ? std::max<int64_t>(1, values.size(batch_ndim + 2)) : 1)
  };
  DimVector size = DimVector(compressed_indices.sizes().slice(0, batch_ndim));
  int64_t compressed_dim_size = (compressed_indices.dim() > 0 && compressed_indices.size(-1) > 0 ? compressed_indices.size(-1) - 1 : 0);
  int64_t plain_dim_size = AT_DISPATCH_INTEGRAL_TYPES(plain_indices.scalar_type(), "estimate_sparse_compressed_tensor_size",
                                                      [&]() -> int64_t {
                                                        if (plain_indices.numel() > 0) {
                                                          return plain_indices.max().item<scalar_t>() + 1;
                                                        } else {
                                                          return 0;
                                                        }
                                                      });
  AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(layout, "estimate_sparse_compressed_tensor_size",
      [&]{
        size.push_back(compressed_dim_size * blocksize[0]);
        size.push_back(plain_dim_size * blocksize[1]);
      },
      [&]{
        size.push_back(plain_dim_size * blocksize[0]);
        size.push_back(compressed_dim_size * blocksize[1]);
      });
  for (int i=0; i<dense_ndim; i++) {
    int64_t j = batch_ndim + 1 + block_ndim + i;
    size.push_back((j < values.dim() ? values.size(j) : 1));
  }
  TORCH_CHECK(
              static_cast<int>(size.size()) == batch_ndim + base_ndim + dense_ndim,
              "tensor dimensionality must be sum of batch, base, and dense dimensionalities (=",
              batch_ndim, " + ", base_ndim, " + ", dense_ndim, ") but got ", size.size());
  return size;
}

// TODO: This constructor should probably use an ATen abstract method in order
// to make autograd dispatch available for the CSR constructor. See the relevant
// note in native_functions.yaml.
Tensor sparse_compressed_tensor(
    const Tensor& compressed_indices,
    const Tensor& plain_indices,
    const Tensor& values,
    IntArrayRef size,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory) {

  if (!layout) {
    AT_ERROR("sparse_compressed_tensor expected sparse compressed tensor layout but got none");
  }
  Layout layout_ = layout.value();
  AT_DISPATCH_ALL_SPARSE_COMPRESSED_LAYOUTS(layout_, "sparse_compressed_tensor", [&]{});

  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout_).device(device).pinned_memory(pin_memory);

  return at::native::_sparse_compressed_tensor_unsafe(
      compressed_indices,
      plain_indices,
      values,
      size,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt());
}

Tensor sparse_compressed_tensor(
    const Tensor& compressed_indices,
    const Tensor& plain_indices,
    const Tensor& values,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory) {

  if (!layout) {
    AT_ERROR("sparse_compressed_tensor expected sparse compressed tensor layout but got none");
  }
  Layout layout_ = layout.value();
  AT_DISPATCH_ALL_SPARSE_COMPRESSED_LAYOUTS(layout_, "sparse_compressed_tensor", [&]{});

  DimVector size = _estimate_sparse_compressed_tensor_size(compressed_indices, plain_indices, values, layout_);

  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout_).device(device).pinned_memory(pin_memory);

  return at::native::_sparse_compressed_tensor_unsafe(
      compressed_indices,
      plain_indices,
      values,
      size,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt());
}

#define SPARSE_COMPRESSED_TENSOR(KIND, REQUIRED_LAYOUT)                 \
  Tensor sparse_##KIND##_tensor(const Tensor& compressed_indices,       \
                                const Tensor& plain_indices,            \
                                const Tensor& values,                   \
                                c10::optional<ScalarType> dtype,        \
                                c10::optional<Layout> layout,           \
                                c10::optional<Device> device,           \
                                c10::optional<bool> pin_memory) {       \
    if (layout) {                                                       \
      TORCH_CHECK(layout.value() == REQUIRED_LAYOUT, "sparse " # KIND " layout must be ", REQUIRED_LAYOUT, " but got ", layout.value()); \
    }                                                                   \
    c10::optional<Layout> layout_(REQUIRED_LAYOUT);                     \
    return at::native::sparse_compressed_tensor(compressed_indices, plain_indices, values, dtype, layout_, device, pin_memory); \
  }                                                                     \
  Tensor sparse_##KIND##_tensor(const Tensor& compressed_indices,       \
                                const Tensor& plain_indices,            \
                                const Tensor& values,                   \
                                IntArrayRef size,                       \
                                c10::optional<ScalarType> dtype,        \
                                c10::optional<Layout> layout,           \
                                c10::optional<Device> device,           \
                                c10::optional<bool> pin_memory) {       \
    if (layout) {                                                       \
      TORCH_CHECK(layout.value() == REQUIRED_LAYOUT, "sparse " # KIND " layout must be ", REQUIRED_LAYOUT, " but got ", layout.value()); \
    }                                                                   \
    c10::optional<Layout> layout_(REQUIRED_LAYOUT);                     \
    return at::native::sparse_compressed_tensor(compressed_indices, plain_indices, values, size, dtype, layout_, device, pin_memory); \
  }

SPARSE_COMPRESSED_TENSOR(csr, kSparseCsr)
SPARSE_COMPRESSED_TENSOR(csc, kSparseCsc)
SPARSE_COMPRESSED_TENSOR(bsr, kSparseBsr)
SPARSE_COMPRESSED_TENSOR(bsc, kSparseBsc)

Tensor empty_sparse_compressed(
    IntArrayRef size,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<MemoryFormat> optional_memory_format) {
  check_size_nonnegative(size);
  TORCH_CHECK(size.size() >= 2, "torch.empty: Only batched sparse compressed (non-block) tensors are supported, but got size ", size);

  // Strided is the default layout for torch.empty.
  Layout layout_ = layout.value_or(Layout::Strided);

  // torch.empty cannot be used to create blocked tensors because its
  // API lacks a method to specify the block size.
  AT_DISPATCH_SPARSE_COMPRESSED_NONBLOCK_LAYOUTS(layout_, "empty_sparse_compressed", [&]{});

  int64_t nnz = 0;
  auto compressed_indices_size = DimVector(size.slice(0, size.size() - 2));
  auto plain_indices_and_values_size = DimVector(size.slice(0, size.size() - 2));
  compressed_indices_size.push_back(size[compressedDimension(layout_, size)] + 1);
  plain_indices_and_values_size.push_back(nnz);

  TensorOptions options = TensorOptions().dtype(ScalarType::Long).layout(Layout::Strided).device(device).pinned_memory(pin_memory);
  auto compressed_indices = at::empty(compressed_indices_size, options);
  auto plain_indices = at::empty(plain_indices_and_values_size, options);
  auto values = at::empty(plain_indices_and_values_size, options.dtype(dtype));

  return at::native::_sparse_compressed_tensor_unsafe(compressed_indices,
                                                      plain_indices,
                                                      values,
                                                      size,
                                                      dtype,
                                                      layout,
                                                      device,
                                                      pin_memory);
}

const Tensor& resize_sparse_csr_(
    const Tensor& self,
    IntArrayRef size,
    c10::optional<MemoryFormat> optional_memory_format) {
  check_size_nonnegative(size);
  TORCH_CHECK(size.size() >= 2, "torch.resize_: Only batched sparse CSR matrices are supported, but got size ", size);
  TORCH_CHECK(
      self.size(-1) <= size[size.size() - 1],
      "torch.resize_: Resizing columns of sparse CSR tensors to a smaller value is not supported. ",
      "The original number of columns is ",
      self.size(-1),
      " while the requested new number of columns is ", size[size.size() - 1], ".");
  get_sparse_csr_impl(self)->resize_(self._nnz(), size);
  return self;
}

Tensor& copy_sparse_compressed_(Tensor& self, const Tensor& src, bool non_blocking) {
  AT_DISPATCH_ALL_SPARSE_COMPRESSED_LAYOUTS(self.layout(), "copy_sparse_compressed_", [&]{});
  TORCH_CHECK(
      self.layout() == src.layout(),
      "torch.copy_: copy of sparse compressed tensors having different layouts is not supported.",
      " self layout is ", self.layout(), " and src layout is ", src.layout());
  TORCH_CHECK(
      self._nnz() == src._nnz(),  // actually, values copy allows different shapes as long as operands are broadcastable
      "torch.copy_: only sparse compressed tensors with the same number of specified elements are supported.");
  auto self_compressed_dim = compressedDimension(self.layout(), self.sizes());
  auto src_compressed_dim = compressedDimension(src.layout(), src.sizes());
  auto self_compressed_dims = self.size(self_compressed_dim);
  auto src_compressed_dims = src.size(compressedDimension(src.layout(), src.sizes()));
  if (self_compressed_dim == src_compressed_dim) {
    TORCH_CHECK(self_compressed_dims == src_compressed_dims,
                "torch.copy_: expected shapes of self and src to match along dimension ",
                self_compressed_dim, " for ",
                self.layout(), " layout but the corresponding dimensions of self and src are ",
                self_compressed_dims, " and ", src_compressed_dims, ", respectively.");
  } else {
    TORCH_CHECK(self_compressed_dims == src_compressed_dims,
                "torch.copy_: expected shapes of self and src to match along dimensions ",
                self_compressed_dim, " and ", src_compressed_dim, ", respectively, for ",
                self.layout(), " layout but the corresponding dimensions of self and src are ",
                self_compressed_dims, " and ", src_compressed_dims, ", respectively.");
  }
  AT_DISPATCH_PLAIN_SPARSE_COMPRESSED_LAYOUTS(self.layout(), "copy_sparse_compressed_",
                                              [&]{},
                                              [&]{
                                                auto self_values = self.values();
                                                auto src_values = src.values();
                                                auto self_blocksize = DimVector(self_values.sizes().slice(self_values.dim()-2, 2));
                                                auto src_blocksize = DimVector(src_values.sizes().slice(src_values.dim()-2, 2));
                                                TORCH_CHECK(self_blocksize == src_blocksize,
                                                            "torch.copy_: copy of sparse compressed tensors having different block sizes is not supported.",
                                                            " self and src block sizes are ", self_blocksize, " and ", src_blocksize, ", respectively.");
                                              });
  AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(self.layout(), "copy_sparse_compressed_",
                                            [&]{
                                              self.crow_indices().copy_(src.crow_indices(), non_blocking);
                                              self.col_indices().copy_(src.col_indices(), non_blocking);
                                            },
                                            [&]{
                                              self.ccol_indices().copy_(src.ccol_indices(), non_blocking);
                                              self.row_indices().copy_(src.row_indices(), non_blocking);
                                            });
  self.values().copy_(src.values(), non_blocking);
  return self;
}

// Access members of CSR tensors.
int64_t _nnz_sparse_csr(const SparseCsrTensor& self) {
  return get_sparse_csr_impl(self)->nnz();
}

Tensor values_sparse_csr(const Tensor& self) {
  return get_sparse_csr_impl(self)->values().alias();
}

Tensor crow_indices_sparse_csr(const Tensor& self) {
  return AT_DISPATCH_SPARSE_ROW_COMPRESSED_LAYOUTS(self.layout(),
                                                   "crow_indices",
                                                   [&]{ return get_sparse_csr_impl(self)->compressed_indices().alias(); });
}

Tensor col_indices_sparse_csr(const Tensor& self) {
  return AT_DISPATCH_SPARSE_ROW_COMPRESSED_LAYOUTS(self.layout(),
                                                   "col_indices",
                                                   [&]{ return get_sparse_csr_impl(self)->plain_indices().alias(); });
}

Tensor ccol_indices_sparse_csr(const Tensor& self) {
  return AT_DISPATCH_SPARSE_COL_COMPRESSED_LAYOUTS(self.layout(),
                                                   "ccol_indices",
                                                   [&]{ return get_sparse_csr_impl(self)->compressed_indices().alias(); });
}

Tensor row_indices_sparse_csr(const Tensor& self) {
  return AT_DISPATCH_SPARSE_COL_COMPRESSED_LAYOUTS(self.layout(),
                                                   "row_indices",
                                                   [&]{ return get_sparse_csr_impl(self)->plain_indices().alias(); });
}

Tensor crow_indices_default(const Tensor& self) {
  TORCH_CHECK(false, "crow_indices expected sparse row compressed tensor layout but got ", self.layout());
}

Tensor col_indices_default(const Tensor& self) {
  TORCH_CHECK(false, "col_indices expected sparse row compressed tensor layout but got ", self.layout());
}

Tensor ccol_indices_default(const Tensor& self) {
  TORCH_CHECK(false, "ccol_indices expected sparse column compressed tensor layout but got ", self.layout());
}

Tensor row_indices_default(const Tensor& self) {
  TORCH_CHECK(false, "row_indices expected sparse column compressed tensor layout but got ", self.layout());
}

int64_t sparse_dim_sparse_csr(const SparseCsrTensor& self) {
  return get_sparse_csr_impl(self)->sparse_dim();
}

int64_t dense_dim_sparse_csr(const SparseCsrTensor& self) {
  return get_sparse_csr_impl(self)->dense_dim();
}

bool _is_same_size_as_sparse_compressed(
    const SparseCsrTensor& self,
    const SparseCsrTensor& src) {
  return self.sizes().equals(src.sizes());
}

const SparseCsrTensor& resize_as_sparse_compressed_(
    const SparseCsrTensor& self,
    const SparseCsrTensor& src) {
  auto src_layout = src.layout();
  auto self_layout = self.layout();
  AT_DISPATCH_ALL_SPARSE_COMPRESSED_LAYOUTS(
      src_layout, "resize_as_sparse_compressed_: src ", []() {});
  AT_DISPATCH_ALL_SPARSE_COMPRESSED_LAYOUTS(
      self_layout, "resize_as_sparse_compressed_: self ", []() {});
  // Note: The impl method does all required checking to see if resize/data copy
  // on member tensors is required.
  get_sparse_csr_impl(self)->resize_as_sparse_compressed_tensor_(src);
  return self;
}

SparseCsrTensor clone_sparse_compressed(
                                        const SparseCsrTensor& self,
                                        c10::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(
      !optional_memory_format.has_value(),
      "unsupported memory format option ",
      optional_memory_format.value());
  TensorOptions options = self.options();
  auto compressed_indices = AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(self.layout(),
                                                                      "clone_sparse_compressed",
                                                                      [&]{ return self.crow_indices(); },
                                                                      [&]{ return self.ccol_indices(); });
  auto plain_indices = AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(self.layout(),
                                                                 "clone_sparse_compressed",
                                                                 [&]{ return self.col_indices(); },
                                                                 [&]{ return self.row_indices(); });
  return at::native::_sparse_compressed_tensor_unsafe(
                                                      compressed_indices.clone(),
                                                      plain_indices.clone(),
                                                      self.values().clone(),
                                                      self.sizes(),
                                                      optTypeMetaToScalarType(options.dtype_opt()),
                                                      options.layout_opt(),
                                                      options.device_opt(),
                                                      options.pinned_memory_opt());
}

Tensor empty_like_sparse_csr(
    const Tensor& self,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  TensorOptions options_ = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);
  TensorOptions options =
      self.options()
          .merge_in(options_)
          .merge_memory_format(optional_memory_format);

  TORCH_CHECK(options.layout() == self.layout(),
    "empty_like with different sparse layout is not supported (self is ",
    self.layout(), " but you requested ", options.layout(), ")");
  if (options.layout() == kSparseCsr) {
    auto result = at::native::_sparse_csr_tensor_unsafe(
        self.crow_indices().clone(),
        self.col_indices().clone(),
        at::empty(self.values().sizes(), options.layout(kStrided)),
        self.sizes(),
        optTypeMetaToScalarType(options.dtype()),
        self.layout(),
        options.device());
    return result;
  } else if (options.layout() == kSparseCsc) {
    auto result = at::native::_sparse_csc_tensor_unsafe(
        self.ccol_indices().clone(),
        self.row_indices().clone(),
        at::empty(self.values().sizes(), options.layout(kStrided)),
        self.sizes(),
        optTypeMetaToScalarType(options.dtype()),
        self.layout(),
        options.device());
    return result;
  } else if (options.layout() == kSparseBsr) {
    auto result = at::native::_sparse_bsr_tensor_unsafe(
        self.crow_indices().clone(),
        self.col_indices().clone(),
        at::empty(self.values().sizes(), options.layout(kStrided)),
        self.sizes(),
        optTypeMetaToScalarType(options.dtype()),
        self.layout(),
        options.device());

    return result;
  } else if (options.layout() == kSparseBsc) {
    auto result = at::native::_sparse_bsc_tensor_unsafe(
        self.ccol_indices().clone(),
        self.row_indices().clone(),
        at::empty(self.values().sizes(), options.layout(kStrided)),
        self.sizes(),
        optTypeMetaToScalarType(options.dtype()),
        self.layout(),
        options.device());
    return result;
  } else if (options.layout() == kStrided) {
    return at::native::empty_like(self, dtype, layout, device, pin_memory, optional_memory_format);
  } else {
    TORCH_CHECK(false, "Layout ", options.layout(), " is not supported");
  }
}

template <bool require_view, bool require_copy>
Tensor select_sparse_csr_worker(const Tensor& self, int64_t dim, int64_t index) {
  constexpr const char* select_name = (require_view ? "select()" : "select_copy()");
  AT_DISPATCH_ALL_SPARSE_COMPRESSED_LAYOUTS(
      self.layout(), "select", []() { return; });
  TORCH_CHECK_INDEX(
      self.dim() != 0, select_name, " cannot be applied to a 0-dim tensor.");
  dim = maybe_wrap_dim(dim, self.dim());
  auto size = self.size(dim);
  if (index < -size || index >= size) {
    TORCH_CHECK_INDEX(
        false,
        select_name, ": index ",
        index,
        " out of range for tensor of size ",
        self.sizes(),
        " at dimension ",
        dim);
  }
  if (index < 0) {
    index += size;
  }

  auto select_strided = [](const Tensor& self, int64_t dim, int64_t index) {
    if (require_copy) {
      return at::select_copy(self, dim, index);
    } else {
      return self.select(dim, index);
    }
  };

  TORCH_INTERNAL_ASSERT(dim >= 0 && dim < self.dim());

  auto new_sizes = DimVector(self.sizes());
  new_sizes.erase(new_sizes.begin() + dim);
  auto options = self.options();

  Tensor plain_indices;
  Tensor compressed_indices;
  std::tie(compressed_indices, plain_indices) =
      AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(
          self.layout(),
          "select",
          [&]() {
            return std::make_pair(self.crow_indices(), self.col_indices());
          },
          [&]() {
            return std::make_pair(self.ccol_indices(), self.row_indices());
          });
  auto n_batch = compressed_indices.dim() - 1;

  if (dim < n_batch) {
    // Selecting batch dimension
    return at::native::_sparse_compressed_tensor_unsafe(
        compressed_indices.select(dim, index),
        plain_indices.select(dim, index),
        select_strided(self.values(), dim, index),
        new_sizes,
        optTypeMetaToScalarType(options.dtype_opt()),
        options.layout_opt(),
        options.device_opt(),
        options.pinned_memory_opt());
  } else if (dim < n_batch + 2) {
    // Selecting sparse dimension
    TORCH_CHECK(
        n_batch == 0,
        select_name, ": selecting sparse dimensions is not implemented for batched sparse compressed tensors.")
    TORCH_INTERNAL_ASSERT(dim == 0 || dim == 1);

    DimVector blocksize{1, 1};
    AT_DISPATCH_PLAIN_SPARSE_COMPRESSED_LAYOUTS(self.layout(), "select", [&] {}, [&] {
      blocksize[0] = std::max<int64_t>(1, self.values().size(n_batch + 1));
      blocksize[1] = std::max<int64_t>(1, self.values().size(n_batch + 2));
    });

    auto indices_options = compressed_indices.options();
    int64_t fast_dim = AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(self.layout(), "select", [&]() { return 0; }, [&]() { return 1; });
    int64_t other_dim = (dim == 0 ? 1 : 0);
    Tensor indices;
    Tensor values;
    bool is_view = dim == fast_dim;
    if (is_view) {
      // select is always a view operation
      Tensor start_end = compressed_indices.narrow(0, index / blocksize[dim], 2).cpu();
      int64_t start = start_end[0].item<int64_t>();
      int64_t end = start_end[1].item<int64_t>();
      indices = plain_indices.slice(0, start, end);
      values = self.values().slice(0, start, end);
    } else {
      Tensor decompressed_indices = at::_convert_indices_from_csr_to_coo(compressed_indices, plain_indices)
        .select(0, 0);

      Tensor dim_indices = at::where(plain_indices.eq(index / blocksize[dim]))[0];
      // Notice that dim_indices is a sorted sequence of non-negative
      // distinct integers. Below we'll try to solve `dim_indices ==
      // arange(start, stop, step)`. If the solution exists then the
      // select will be a view operation also for the `dim !=
      // fast_dim` case.
      int64_t start{}, end{}, step{};
      if (solve_arange(dim_indices, start, end, step)) {
        indices = decompressed_indices.slice(0, start, end, step);
        values = self.values().slice(0, start, end, step);
        is_view = true;
      } else {
        // select will be a copy operation due to index_select!
        indices = decompressed_indices.index_select(0, dim_indices);
        values = self.values().index_select(0, dim_indices);
      }
    }

    AT_DISPATCH_PLAIN_SPARSE_COMPRESSED_LAYOUTS(self.layout(), "select", [&]() {},
        [&]() {
          /*
            The formula for select indices and values below are best
            explained by an example. Consider a BSR tensor with a
            block size (2, 3) having four blocks (the other two blocks
            contain all zeros and hence will not be specified):

              [ 1  2  3] | [ 7  8  9]
              [ 4  5  6] | [10 11 12]
              ---------------------
              [13 14 15] | [ 0  0  0]
              [16 17 18] | [ 0  0  0]
              -----------------------
              [ 0  0  0] | [19 20 21]
              [ 0  0  0] | [22 23 24]

            that represents a 6 x 6 tensor:

              [  1  2  3  7  8  9 ]
              [  4  5  6 10 11 12 ]
              [ 13 14 15  0  0  0 ]
              [ 16 17 18  0  0  0 ]
              [  0  0  0 19 20 21 ]
              [  0  0  0 22 23 24 ]

            The corresponding data for the BSR representation is:

              crow_indices = [0 2 3 4]
              col_indices =  [0 1 0 1]
              values = [ [[1 2 3], [4 5 6]], [[7 8 9], [10 11 12]], [[13 14 15], [16 17 18]], [[19 20 21], [22 23 24]] ]
              shape = (6, 6)

            From crow_indices, we can find that

              row_indices = [0 0 1 2]

            In the following, we'll illustrate the details of
            computing the result of torch.select_copy(input, dim,
            index) where dim is 0 or 1, and index is in
            range(shape[dim]).

            Select a row of a BSR tensor
            ----------------------------

            We will consider first the dim=0 case that corresponds to
            selecting a index-th row of the tensor. For instance, for
            dim=0 and index=1, the expected result would represent a
            1D tensor:

              [  4  5  6 10 11 12 ]

            that is a concatenated tensor of certain slices from the
            first and the second block that is computed as follows:

              values[dim_indices].select(1 + dim, index % blocksize[dim]).flatten(0, 1)
              -> values[[0, 1]][:, 1 % 2].flatten(0, 1)
              -> [ [[1 2 3], [4 5 6]], [[7 8 9], [10 11 12]] ][:, 1].flatten(0, 1)
              -> [ [4 5 6], [10 11 12]].flatten(0, 1)
              -> [ 4 5 6 10 11 12]

            where dim_indices is found as

              where(row_indices == index//blocksize[dim])
              -> where([0 0 1 2] == 1//2)
              -> [0 1]

            The corresponding column indices are computed as

              (col_indices[dim_indices].mul(blocksize[other_dim]).unsqueeze(1) + arange(blocksize[other_dim]).unsqueeze(0)).flatten(0, 1)

            where other_dim is 1 if dim is 0, and 0 if dim is 1. Let's
            expand the above expression with the data in the example:

              -> (col_indices[[0, 1]].mul(3).unsqueeze(1) + arange(3).unsqueeze(0)).flatten(0, 1)
              -> ([[0 1].mul(3).unsqueeze(1) + [[0 1 2]]).flatten(0, 1)
              -> ([[[0], [3]] + [[0 1 2]]).flatten(0, 1)     <- here addition will use broadcasting rules!
              -> ([[[0 1 2], [3 4 5]]).flatten(0, 1)
              -> [0 1 2 3 4 5]

            Finally, the select(dim=0, index=1) op on the given sparse
            compressed tensors will return a COO tensor:

              sparse_coo_tensor([0 1 2 3 4 5].unsqueeze(0), [4 5 6 10 11 12], (6,))

            that represents the expected result: [ 4 5 6 10 11 12 ]

            Select a column of a BSR tensor
            -------------------------------

            Next, we'll consider the dim=1 case that corresponds to
            selecting the index-th column of the tensor. For instance,
            for dim=1 and index=4, the expected result would represent
            a 1D tensor:

              [  8 11 0  0 20 23]

            that is a concatenated tensor of certain slices from the
            second and the last block:

              values[dim_indices].select(1 + dim, index % blocksize[dim]).flatten(0, 1)
              -> values[[1, 3]][:, :, 4 % 3 ].flatten(0, 1)
              -> [ [[7 8 9], [10 11 12]], [[19 20 21], [22 23 24]] ][:, 1, 1].flatten(0, 1)
              -> [ [8 11], [20 23]].flatten(0, 1)
              -> [ 8 11 20 23 ]

            The corresponding row indices are computed as

              (row_indices[dim_indices].mul(blocksize[other_dim]).unsqueeze(1) + arange(blocksize[other_dim]).unsqueeze(0)).flatten(0, 1)

            where dim_indices is

              where(col_indices == index//blocksize[dim])
              -> where([0 1 0 1] == 4//3)
              -> [1 3]

            and we have

              (row_indices[dim_indices].mul(blocksize[other_dim]).unsqueeze(1) + arange(blocksize[other_dim]).unsqueeze(0)).flatten(0, 1)
              -> (row_indices[[1 3]].mul(2).unsqueeze(1) + arange(2).unsqueeze(0)).flatten(0, 1)
              -> ([0 4].unsqueeze(1) + [0 1].unsqueeze(0)).flatten(0, 1)
              -> ([[0], [4]] + [[0 1]]).flatten(0, 1)     <- here addition will use broadcasting rules!
              -> ([[0 1], [4 5]]).flatten(0, 1)
              -> [ 0 1 4 5 ]

            Finally, the select(dim=1, index=4) op on the given sparse
            compressed tensors will return a COO tensor:

              sparse_coo_tensor([0 1 4 5].unsqueeze(0), [8 11 20 23], (6,))

            that represents the expected result: [ 8 11 0 0 20 23 ]

           */
          Tensor subblock_indices = at::arange(0, blocksize[other_dim], indices_options);
          indices = indices.mul(blocksize[other_dim]).unsqueeze(1).add(subblock_indices.unsqueeze(0)).flatten(0, 1);
          values = values.select(dim + 1, index % blocksize[dim]).flatten(0, 1);
          // flatten(0, 1) can be a view or a copy operation. If view
          // is required, it will be checked below via is_alias_of,
          // otherwise, we'll check if copy is made here to avoid
          // unnecessary clone below:
          if (require_copy) {
            is_view = values.is_alias_of(self.values());
          }
        });

    if (require_view) {
      TORCH_CHECK(values.is_alias_of(self.values()), select_name,
                  ": no view exists for the given input, consider using torch.select_copy.");
    }

    indices = indices.unsqueeze(0).to(kLong);
    if (require_copy && is_view) {
      values = values.clone();
    }
    return at::_sparse_coo_tensor_unsafe(indices, values, new_sizes)._coalesced_(true);
  } else {
    // Selecting dense dimension
    Tensor new_values = AT_DISPATCH_PLAIN_SPARSE_COMPRESSED_LAYOUTS(
        self.layout(),
        "select",
        // Non blocked layout (2 sparse dims become 1 nnz dim in values, so dim
        // is found one position to the left)
        [&]() { return select_strided(self.values(), dim - 1, index); },
        // Block layout (2 sparse dims become 1 nnz dim + 2 block-shape dims in
        // values, so dim is found 1 position to the right)
        [&]() { return select_strided(self.values(), dim + 1, index); });
    return at::native::_sparse_compressed_tensor_unsafe(
        compressed_indices,
        plain_indices,
        new_values,
        new_sizes,
        optTypeMetaToScalarType(options.dtype_opt()),
        options.layout_opt(),
        options.device_opt(),
        options.pinned_memory_opt());
  }
}

Tensor select_sparse_csr(const Tensor& self, int64_t dim, int64_t index) {
  return select_sparse_csr_worker<true, false>(self, dim, index);
}

Tensor select_copy_sparse_csr(const Tensor& self, int64_t dim, int64_t index) {
  return select_sparse_csr_worker<false, true>(self, dim, index);
}

} // namespace at::native
