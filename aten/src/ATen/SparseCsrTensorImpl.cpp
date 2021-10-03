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
} // namespace

SparseCsrTensorImpl::SparseCsrTensorImpl(
    at::DispatchKeySet key_set,
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
      ) {}

SparseCsrTensorImpl::SparseCsrTensorImpl(
    at::DispatchKeySet key_set,
    const caffe2::TypeMeta data_type,
    at::Tensor crow_indices,
    at::Tensor col_indices,
    at::Tensor values)
    : TensorImpl(key_set, data_type, values.device()),
      crow_indices_(std::move(crow_indices)),
      col_indices_(std::move(col_indices)),
      values_(std::move(values)) {}

void SparseCsrTensorImpl::resize_as_sparse_csr_tensor_(const Tensor& src) {
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
} // namespace at
