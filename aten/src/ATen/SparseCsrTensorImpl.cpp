#include <ATen/ATen.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/SparseCsrTensorImpl.h>
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

std::string SparseCsrTensorLayoutToSTRING(Layout layout) {
  switch (layout) {
  case kSparseCsr: return "CSR";
  case kSparseCsc: return "CSC";
  case kSparseBsr: return "BSR";
  case kSparseBsc: return "BSC";
  default:
    TORCH_CHECK(false, "Not a sparse compressed layout:", layout);
    return "";
  }
}
} // namespace

SparseCsrTensorImpl::SparseCsrTensorImpl(
    at::DispatchKeySet key_set,
    Layout layout,
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
    Layout layout)
    : TensorImpl(key_set, data_type, values.device()),
      crow_indices_(std::move(crow_indices)),
      col_indices_(std::move(col_indices)),
      values_(std::move(values)),
      layout_(layout) {
  // https://pytorch.org/blog/pytorch-feature-classification-changes/#beta
  TORCH_WARN_ONCE("Sparse ", SparseCsrTensorLayoutToSTRING(layout_), " tensor support is in beta state.");
  set_storage_access_should_throw();
  is_non_overlapping_and_dense_ = false;
  set_has_contiguity_policy(HasContiguityPolicy::ContiguityNotSupported);
}

const char* SparseCsrTensorImpl::tensorimpl_type_name() const {
  return "SparseCsrTensorImpl";
}

void SparseCsrTensorImpl::resize_(int64_t nnz, IntArrayRef size) {
  TORCH_CHECK((layout_ == kSparseCsr || layout_ == kSparseBsr), "resize_: layout ", layout_, " is not yet supported");  // TODO
  auto row_index = size.size() - ((layout_ == kSparseCsr || layout_ == kSparseBsr) ? 2 : 1);
  auto col_index = size.size() - ((layout_ == kSparseCsr || layout_ == kSparseBsr) ? 1 : 2);
  auto rows = size[row_index];
  auto cols = size[col_index];
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
  TORCH_CHECK(false, "Sparse ", SparseCsrTensorLayoutToSTRING(layout_)," tensors do not have strides.");
}
int64_t SparseCsrTensorImpl::stride(int64_t d) const {
  TORCH_CHECK(false, "Sparse ", SparseCsrTensorLayoutToSTRING(layout_), " tensors do not have strides.");
}
void SparseCsrTensorImpl::set_size(int64_t dim, int64_t new_size) {
  TORCH_CHECK(false, "Sparse ", SparseCsrTensorLayoutToSTRING(layout_), " tensors do not have set_size.");
}
void SparseCsrTensorImpl::set_stride(int64_t dim, int64_t new_stride) {
  TORCH_CHECK(false, "Sparse ", SparseCsrTensorLayoutToSTRING(layout_), " tensors do not have set_stride.");
}
void SparseCsrTensorImpl::set_storage_offset(int64_t storage_offset) {
  TORCH_CHECK(false, "Sparse ", SparseCsrTensorLayoutToSTRING(layout_), " tensors do not have set_storage_offset.");
}

} // namespace at
