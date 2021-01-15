#include <ATen/ATen.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/CompressedSparseTensorImpl.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/core/LegacyTypeDispatch.h>

namespace at {
namespace {
  DeviceType sparseGCSTensorSetToDeviceType(DispatchKeySet key_set) {
    if (key_set.has(DispatchKey::CompressedRowSparseCPU)) {
      return kCPU;
    } else if (key_set.has(DispatchKey::CompressedRowSparseCUDA)) {
      return kCUDA;
    } else {
      AT_ERROR("Cannot construct SparseTensor with non-sparse tensor type ID ", key_set);
    }
  }
}

SparseGCSTensorImpl::SparseGCSTensorImpl(at::DispatchKeySet key_set,
                                         const caffe2::TypeMeta& data_type)
  :   SparseGCSTensorImpl(key_set, data_type
      , at::empty({0}, at::initialTensorOptions().device(sparseGCSTensorSetToDeviceType(key_set)).dtype(ScalarType::Int)) // crow_indices
      // indices in case of GCS tensor is always a 1D array so need to init size as {1,0}.
      , at::empty({0}, at::initialTensorOptions().device(sparseGCSTensorSetToDeviceType(key_set)).dtype(ScalarType::Int)) // indices
      , at::empty({0}, at::initialTensorOptions().device(sparseGCSTensorSetToDeviceType(key_set)).dtype(data_type)) // values
      , at::empty({0}, at::initialTensorOptions().device(sparseGCSTensorSetToDeviceType(key_set)).dtype(ScalarType::Int)) // reduction
) {}

SparseGCSTensorImpl::SparseGCSTensorImpl(at::DispatchKeySet key_set,
                                         const caffe2::TypeMeta& data_type,
                                         at::Tensor crow_indices, at::Tensor col_indices, 
                                         at::Tensor values, at::Tensor reduction)
  : TensorImpl(key_set, data_type, values.device()),
    crow_indices_(std::move(crow_indices)),
    col_indices_(std::move(col_indices)),
    values_(std::move(values)),
    reduction_(std::move(reduction)) {}

void SparseGCSTensorImpl::resize_and_clear_(int64_t nnz_size, int64_t ptr_size, int64_t redux_size, IntArrayRef size) {
  // TODO: perform error checking.
  TORCH_CHECK(size.size() + 1 == redux_size, 
              "size of the reduction array has to be len(sparse.shape)+1, but got: ", 
              redux_size);

  // call crow_indices().options() here since the struct contructor calls the tensor constructor
  // with args for device specific init.
  auto empty_crow_indices = at::empty(ptr_size, crow_indices().options());
  auto empty_col_indices = at::empty(nnz_size, col_indices().options());
  auto empty_values = at::empty(nnz_size, values().options());
  auto empty_reduction = at::empty(redux_size, reduction().options());

  TORCH_CHECK(empty_col_indices.scalar_type() == kInt, 
    "empty_col_indices must be an int32 type, but got: ", empty_col_indices.dtype());
  TORCH_CHECK(empty_crow_indices.scalar_type() == kInt,
    "empty_crow_indices must be int32 type, but got: ",   empty_crow_indices.dtype());
  TORCH_CHECK(empty_reduction.scalar_type() == kInt, 
    "empty_reduction must be int32 type, but got: ",  empty_reduction.dtype());

  // directly set to the member variables. there should be lots of error checking here.
  crow_indices_ = empty_crow_indices;
  col_indices_ = empty_col_indices;
  values_ = empty_values;
  reduction_ = empty_reduction;
  sizes_ = size.vec();
}

void SparseGCSTensorImpl::resize_as_(const Tensor& src) {
  crow_indices_ = at::empty_like(src.crow_indices(), src.crow_indices().options(), src.crow_indices().suggest_memory_format());
  col_indices_ = at::empty_like(src.col_indices(), src.col_indices().options(), 
    src.col_indices().suggest_memory_format());
  values_ = at::empty_like(src.values(), src.values().options(), src.values().suggest_memory_format());
  reduction_ = at::empty_like(src.reduction(), src.reduction().options(), src.reduction().suggest_memory_format());
  sizes_ = src.sizes();
}
  
void SparseGCSTensorImpl::set_member_tensors_unsafe(const Tensor& crow_indices, const Tensor& col_indices,
                                                      const Tensor& values, const Tensor& reduction) {
  TORCH_CHECK(!col_indices.is_sparse(), 
              "expected col_indices to be a dense tensor, but got indices of layout ", 
              col_indices.layout());
  TORCH_CHECK(!crow_indices.is_sparse(), 
              "expected crow_indices to be a dense tensor, but got crow_indices of layout ", 
              crow_indices.layout());
  TORCH_CHECK(!values.is_sparse(), 
              "expected values to be a dense tensor, but got values of layout ", 
              values.layout());
  TORCH_CHECK(!reduction.is_sparse(), 
              "expected reduction to be a dense tensor, but got reduction of layout ", 
              reduction.layout());

  TORCH_CHECK(values.device().type() == device().type(), "device type of values (", values.device().type(),
              ") must match device type of device().type()", device().type(), ")");
  TORCH_CHECK(!col_indices.is_cuda() || col_indices.get_device() == values.get_device(), "device of col_indices (", 
              col_indices.get_device(), ") must match device of values (", values.get_device(), ")");
  TORCH_CHECK(!crow_indices.is_cuda() || crow_indices.get_device() == values.get_device(), "device of crow_indices (", 
              crow_indices.get_device(), ") must match device of values (", values.get_device(), ")");

  TORCH_CHECK(col_indices.size(0) == values.size(0), 
              "col_indices and values must have same nnz, but got nnz from indices: ",
              col_indices.size(0), ", nnz from values: ", values.size(0));

  TORCH_CHECK(crow_indices.dim() == 1, "crow_indices must have dim=1 but got crow_indices.dim()=", 
              crow_indices.dim());
  TORCH_CHECK(col_indices.dim() == 1, "col_indices must have dim=1 but got col_indices.dim()=",
              col_indices.dim()); 
  TORCH_CHECK(values.dim() == 1, "values must have dim=1 but got values.dim()=", values.dim());
  TORCH_CHECK(reduction.dim() == 1, "reduction must have dim=1 but got reduction.dim()=", reduction.dim());

  TORCH_CHECK(col_indices.scalar_type() == kInt, "col_indices must be an int32 type, but got: ", 
              col_indices.dtype());
  TORCH_CHECK(crow_indices.scalar_type() == kInt, "crow_indices must be int32 type, but got: ", crow_indices.dtype());
  TORCH_CHECK(reduction.scalar_type() == kInt, "reduction must be int32 type, but got: ", reduction.dtype());
  TORCH_CHECK(values.scalar_type() == typeMetaToScalarType(dtype()), 
              "dtype of values (", values.scalar_type(), ") must match dtype of sparse tensor (", 
              typeMetaToScalarType(dtype()), ")");

  crow_indices_ = crow_indices;
  col_indices_ = col_indices;
  values_ = values;
  reduction_ = reduction;

  AT_ASSERT(device() == values_.device());    
  AT_ASSERT(col_indices_.device() == values_.device());
  AT_ASSERT(reduction_.device() == values_.device());

  auto reduction_accessor = reduction_.accessor<int32_t, 1>();

  rsplit_dim_ = reduction_accessor[reduction_.size(0)-1];
  TORCH_CHECK(rsplit_dim_ <= sizes_.size(), "Dimensions can only be split between 0 and ", 
              sizes_.size(), ", but got split dimension as: ", rsplit_dim_);
  
  int64_t dim0 = 1, dim1 = 1;
  for (int i = 0; i < rsplit_dim_; ++i) {
    dim0 *= sizes_[i];
  }
  TORCH_CHECK(dim0 <= INT_MAX, "row dimension of reduced tensor must be <= ", INT_MAX);

  for (int i = rsplit_dim_; i < sizes_.size(); ++i) {
    dim1 *= sizes_[i];
  }
  TORCH_CHECK(dim1 <= INT_MAX, "column dimension of reduced tensor must be <= ", INT_MAX);

  dims0_.resize(rsplit_dim_);
  strides0_.resize(1);
  
  dims1_.resize(sizes_.size() - rsplit_dim_);
  strides1_.resize(1);
  
  for (int i = 0; i < rsplit_dim_; ++i) { dims0_[i] = i; }
  for (int i = 0; i < sizes_.size() - rsplit_dim_; ++i) { dims1_[i] = i + rsplit_dim_; }

  make_strides(0, strides0_, dims0_);
  make_strides(rsplit_dim_, strides1_, dims1_);
}

void SparseGCSTensorImpl::make_strides(int shape_start, std::vector<int>& strides, std::vector<int>& dims) {
  int ndims = dims.size();
  strides[0] = 1;
  for (int i = 0; i < ndims-1; ++i) {
    strides.insert(strides.begin(), strides[0] * sizes_[shape_start + ndims - i - 1]);
  }
}
}
