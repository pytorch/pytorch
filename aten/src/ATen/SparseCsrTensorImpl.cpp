#include <ATen/ATen.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/SparseCsrTensorImpl.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/native/Resize.h>

namespace at {

namespace {
DeviceType SparseCsrTensorSetToDeviceType(DispatchKeySet key_set) {
  if (key_set.has(DispatchKey::SparseCsrCPU)) {
    return kCPU;
  } else if (key_set.has(DispatchKey::SparseCsrCUDA)) {
    return kCUDA;
  } else {
    TORCH_CHECK(false,
        "Cannot construct SparseCsrTensor with non-sparse tensor type ID ",
        key_set);
  }
}
} // namespace

SparseCsrTensorImpl::SparseCsrTensorImpl(
    at::DispatchKeySet key_set,
    at::Layout layout,
    const caffe2::TypeMeta data_type)
    : SparseCsrTensorImpl(
          key_set,
          data_type,
          at::empty(
              {0},
              at::initialTensorOptions()
                  .device(SparseCsrTensorSetToDeviceType(key_set))
                  .dtype(ScalarType::Int)) // crow_indices
          ,
          at::empty(
              {0},
              at::initialTensorOptions()
                  .device(SparseCsrTensorSetToDeviceType(key_set))
                  .dtype(ScalarType::Int)) // col_indices
          ,
          at::empty(
              {0},
              at::initialTensorOptions()
                  .device(SparseCsrTensorSetToDeviceType(key_set))
                  .dtype(data_type)) // values
          ,
          layout
      ) {}

SparseCsrTensorImpl::SparseCsrTensorImpl(
    at::DispatchKeySet key_set,
    const caffe2::TypeMeta data_type,
    at::Tensor crow_indices,
    at::Tensor col_indices,
    at::Tensor values,
    at::Layout layout)
    : TensorImpl(key_set, data_type, values.device()),
      crow_indices_(std::move(crow_indices)),
      col_indices_(std::move(col_indices)),
      values_(std::move(values)),
      layout_(layout) {
  // https://pytorch.org/blog/pytorch-feature-classification-changes/#beta
  TORCH_WARN_ONCE("Sparse ", at::sparse_csr::layoutToString(layout_, /*upper=*/true), " tensor support is in beta state."
                  "If you miss a functionality in the sparse tensor support, please submit a feature request "
                  "to https://github.com/pytorch/pytorch/issues.");
  set_storage_access_should_throw();
  is_non_overlapping_and_dense_ = false;
  set_has_contiguity_policy(HasContiguityPolicy::ContiguityNotSupported);
}

const char* SparseCsrTensorImpl::tensorimpl_type_name() const {
  return "SparseCsrTensorImpl";
}

void SparseCsrTensorImpl::resize_(int64_t nnz, IntArrayRef size) {
  // TODO: check how nnz is defined for block tensors? Is it the
  // number of blocks or the number of blocks times the numel per
  // block?
  TORCH_CHECK(layout_ == kSparseCsr, "resize_: layout ", layout_, " is not yet supported"); // TODO
  int row_dim = at::sparse_csr::rowDimension(layout_, size);
  int col_dim = at::sparse_csr::columnDimension(layout_, size);
  auto rows = size[row_dim];
  auto cols = size[col_dim];
  auto old_crow_indices_size = crow_indices_.size(-1);

  auto new_crow_indices_size = DimVector(size.slice(0, size.size() - 2));
  new_crow_indices_size.push_back(rows + 1);
  crow_indices_.resize_(new_crow_indices_size);
  if (rows + 1 >= old_crow_indices_size) {
    crow_indices_.narrow(-1, old_crow_indices_size, rows + 1 - old_crow_indices_size).fill_(nnz);
  } else {
    crow_indices_.narrow(-1, rows, 1).fill_(std::min<int64_t>(nnz, rows*cols));
  }
  auto col_indices_values_size = DimVector(size.slice(0, size.size() - 2));
  col_indices_values_size.push_back(std::min<int64_t>(nnz, rows*cols));
  col_indices_.resize_(col_indices_values_size);
  values_.resize_(col_indices_values_size);
  sizes_and_strides_.set_sizes(size);
}

void SparseCsrTensorImpl::resize_as_sparse_csr_tensor_(const Tensor& src) {
  set_layout(src.layout());
  crow_indices_ = at::empty_like(
      src.crow_indices(),
      src.crow_indices().options(),
      src.crow_indices().suggest_memory_format());
  col_indices_ = at::empty_like(
      src.col_indices(),
      src.col_indices().options(),
      src.col_indices().suggest_memory_format());
  values_ = at::empty_like(
      src.values(),
      src.values().options(),
      src.values().suggest_memory_format());
  sizes_and_strides_.set_sizes(src.sizes());
  refresh_numel();
}

void SparseCsrTensorImpl::set_member_tensors(
    const Tensor& crow_indices,
    const Tensor& col_indices,
    const Tensor& values,
    IntArrayRef size) {

  // CSR Type Invariants
  TORCH_CHECK(
      values.scalar_type() == typeMetaToScalarType(dtype()),
      "dtype of values (",
      values.scalar_type(),
      ") must match dtype of sparse tensor (",
      typeMetaToScalarType(dtype()),
      ")");

  crow_indices_ = crow_indices;
  col_indices_ = col_indices;
  values_ = values;

  sizes_and_strides_.set_sizes(size);
  refresh_numel();
}

IntArrayRef SparseCsrTensorImpl::strides() const {
  TORCH_CHECK(false, "Sparse ", at::sparse_csr::layoutToString(layout_, /*upper=*/true), " tensors do not have strides.");
}
int64_t SparseCsrTensorImpl::stride(int64_t d) const {
  TORCH_CHECK(false, "Sparse ", at::sparse_csr::layoutToString(layout_, /*upper=*/true), " tensors do not have strides.");
}
void SparseCsrTensorImpl::set_size(int64_t dim, int64_t new_size) {
  TORCH_CHECK(false, "Sparse ", at::sparse_csr::layoutToString(layout_, /*upper=*/true), " tensors do not have set_size.");
}
void SparseCsrTensorImpl::set_stride(int64_t dim, int64_t new_stride) {
  TORCH_CHECK(false, "Sparse ", at::sparse_csr::layoutToString(layout_, /*upper=*/true), " tensors do not have set_stride.");
}
void SparseCsrTensorImpl::set_storage_offset(int64_t storage_offset) {
  TORCH_CHECK(false, "Sparse ", at::sparse_csr::layoutToString(layout_, /*upper=*/true), " tensors do not have set_storage_offset.");
}

} // namespace at
