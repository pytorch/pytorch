#include <ATen/ATen.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/SparseGCSTensorImpl.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/core/LegacyTypeDispatch.h>

namespace at {
namespace {
  DeviceType sparseGCSTensorSetToDeviceType(DispatchKeySet key_set) {
    if (key_set.has(DispatchKey::SparseGCS_CPU)) {
      return kCPU;
    } else if (key_set.has(DispatchKey::SparseGCS_CUDA)) {
      return kCUDA;
    } else {
      AT_ERROR("Cannot construct SparseTensor with non-sparse tensor type ID ", key_set);
    }
  }
}

SparseGCSTensorImpl::SparseGCSTensorImpl(at::DispatchKeySet key_set,
                                         const caffe2::TypeMeta& data_type)
  :   SparseGCSTensorImpl(key_set, data_type
      , at::empty({0}, at::initialTensorOptions().device(sparseGCSTensorSetToDeviceType(key_set)).dtype(ScalarType::Int)) // pointers
      // indices in case of GCS tensor is always a 1D array so need to init size as {1,0}.
      , at::empty({0}, at::initialTensorOptions().device(sparseGCSTensorSetToDeviceType(key_set)).dtype(ScalarType::Int)) // indices
      , at::empty({0}, at::initialTensorOptions().device(sparseGCSTensorSetToDeviceType(key_set)).dtype(data_type)) // values
      , at::empty({0}, at::initialTensorOptions().device(sparseGCSTensorSetToDeviceType(key_set)).dtype(ScalarType::Int)) // reduction
) {}

SparseGCSTensorImpl::SparseGCSTensorImpl(at::DispatchKeySet key_set,
                                         const caffe2::TypeMeta& data_type,
                                         at::Tensor pointers, at::Tensor indices, at::Tensor values,
                                         at::Tensor reduction)
  : TensorImpl(key_set, data_type, values.device()),
    pointers_(std::move(pointers)),
    indices_(std::move(indices)),
    values_(std::move(values)),
    reduction_(std::move(reduction)) {}

void SparseGCSTensorImpl::resize_and_clear_(int64_t nnz_size, int64_t ptr_size, int64_t redux_size, IntArrayRef size) {
  // TODO: perform error checking.
  TORCH_CHECK(size.size() + 1 == redux_size, "size of the reduction array has to be len(sparse.shape)+1, but got: ", redux_size);

  // call pointers().options() here since the struct contructor calls the tensor constructor
  // with args for device specific init.
  auto empty_pointers = at::empty(ptr_size, pointers().options());
  auto empty_indices = at::empty(nnz_size, indices().options());
  auto empty_values = at::empty(nnz_size, values().options());
  auto empty_reduction = at::empty(redux_size, reduction().options());

  TORCH_CHECK(empty_indices.scalar_type() == kInt,   "empty_indices must be an int32 type, but got: ", empty_indices.dtype());
  TORCH_CHECK(empty_pointers.scalar_type() == kInt,  "empty_pointers must be int32 type, but got: ",   empty_pointers.dtype());
  TORCH_CHECK(empty_reduction.scalar_type() == kInt, "empty_reduction must be int32 type, but got: ",  empty_reduction.dtype());

  // directly set to the member variables. there should be lots of error checking here.
  pointers_ = empty_pointers;
  indices_ = empty_indices;
  values_ = empty_values;
  reduction_ = empty_reduction;
  sizes_ = size.vec();
}

void SparseGCSTensorImpl::resize_as_(const Tensor& src) {
  pointers_ = at::empty_like(src.pointers(), src.pointers().options(), src.pointers().suggest_memory_format());
  indices_ = at::empty_like(src.indices(), src.indices().options(), src.indices().suggest_memory_format());
  values_ = at::empty_like(src.values(), src.values().options(), src.values().suggest_memory_format());
  reduction_ = at::empty_like(src.reduction(), src.reduction().options(), src.reduction().suggest_memory_format());
  sizes_ = src.sizes();
}
  
void SparseGCSTensorImpl::set_member_tensors_unsafe(const Tensor& pointers, const Tensor& indices,
                                                      const Tensor& values, const Tensor& reduction) {
  TORCH_CHECK(!indices.is_sparse(), "expected indices to be a dense tensor, but got indices of layout ", indices.layout());
  TORCH_CHECK(!pointers.is_sparse(), "expected pointers to be a dense tensor, but got pointers of layout ", pointers.layout());
  TORCH_CHECK(!values.is_sparse(), "expected values to be a dense tensor, but got values of layout ", values.layout());
  TORCH_CHECK(!reduction.is_sparse(), "expected reduction to be a dense tensor, but got reduction of layout ", reduction.layout());

  TORCH_CHECK(values.device().type() == device().type(), "device type of values (", values.device().type(),
              ") must match device type of device().type()", device().type(), ")");
  TORCH_CHECK(!indices.is_cuda() || indices.get_device() == values.get_device(), "device of indices (", indices.get_device(),
              ") must match device of values (", values.get_device(), ")");
  TORCH_CHECK(!pointers.is_cuda() || pointers.get_device() == values.get_device(), "device of pointers (", pointers.get_device(),
              ") must match device of values (", values.get_device(), ")");

  TORCH_CHECK(indices.size(0) == values.size(0), "indices and values must have same nnz, but got nnz from indices: ",
              indices.size(0), ", nnz from values: ", values.size(0));

  TORCH_CHECK(pointers.dim() == 1, "pointers must have dim=1 but got pointers.dim()=", pointers.dim());
  TORCH_CHECK(indices.dim() == 1, "indices must have dim=1 but got indices.dim()=", indices.dim()); 
  TORCH_CHECK(values.dim() == 1, "values must have dim=1 but got values.dim()=", values.dim());
  TORCH_CHECK(reduction.dim() == 1, "reduction must have dim=1 but got reduction.dim()=", reduction.dim());

  TORCH_CHECK(indices.scalar_type() == kInt, "indices must be an int32 type, but got: ", indices.dtype());
  TORCH_CHECK(pointers.scalar_type() == kInt, "pointers must be int32 type, but got: ", pointers.dtype());
  TORCH_CHECK(reduction.scalar_type() == kInt, "reduction must be int32 type, but got: ", reduction.dtype());
  TORCH_CHECK(values.scalar_type() == typeMetaToScalarType(dtype()), "dtype of values (", values.scalar_type(), ") must match dtype of sparse tensor (", typeMetaToScalarType(dtype()), ")");
  
  pointers_ = pointers;
  indices_ = indices;
  values_ = values;
  reduction_ = reduction;

  AT_ASSERT(device() == values_.device());    
  AT_ASSERT(indices_.device() == values_.device());
  AT_ASSERT(reduction_.device() == values_.device());

  auto reduction_accessor = reduction_.accessor<int32_t, 1>();

  rsplit_dim_ = reduction_accessor[reduction_.size(0)-1];
  TORCH_CHECK(rsplit_dim_ <= sizes_.size(), "Dimensions can only be split between 0 and ", sizes_.size(), ", but got split dimension as: ", rsplit_dim_);
    
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
