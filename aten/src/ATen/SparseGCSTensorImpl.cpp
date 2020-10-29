#include <ATen/ATen.h>
#include <ATen/SparseGCSTensorImpl.h>
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
      , at::empty({0}, at::initialTensorOptions().device(sparseGCSTensorSetToDeviceType(key_set)).dtype(ScalarType::Long))
      // indices in case of GCS tensor is always a 1D array so need to init size as {1,0}.
      , at::empty({0}, at::initialTensorOptions().device(sparseGCSTensorSetToDeviceType(key_set)).dtype(ScalarType::Long))
      , at::empty({0}, at::initialTensorOptions().device(sparseGCSTensorSetToDeviceType(key_set)).dtype(data_type))
      , at::empty({0}, at::initialTensorOptions().device(sparseGCSTensorSetToDeviceType(key_set)).dtype(ScalarType::Long))
      , Scalar()  ) {}

SparseGCSTensorImpl::SparseGCSTensorImpl(at::DispatchKeySet key_set,
                                         const caffe2::TypeMeta& data_type,
                                         at::Tensor pointers, at::Tensor indices, at::Tensor values,
                                         at::Tensor reduction, Scalar fill_value)
  : TensorImpl(key_set, data_type, values.device()),
    pointers_(std::move(pointers)),
    indices_(std::move(indices)),
    values_(std::move(values)),
    reduction_(std::move(reduction)),
    fill_value_(std::move(fill_value)) {}

void SparseGCSTensorImpl::resize_(IntArrayRef size) {
    
}

void SparseGCSTensorImpl::resize_and_clear_(int64_t nnz_size, int64_t ptr_size, int64_t redux_size, IntArrayRef size) {
  // TODO: perform error checking.

  // call pointers().options() here since the struct contructor calls the tensor constructor
  // with args for device specific init.
  auto empty_pointers = at::empty(ptr_size, pointers().options());
  auto empty_indices = at::empty(nnz_size, indices().options());
  auto empty_values = at::empty(nnz_size, values().options());
  auto empty_reduction = at::empty(redux_size, reduction().options());

  // directly set to the member variables. there should be lots of error checking here.
  pointers_ = empty_pointers;
  indices_ = empty_indices;
  values_ = empty_values;
  reduction_ = empty_reduction;
  sizes_ = size.vec();
}
  
void SparseGCSTensorImpl::set_member_tensors_unsafe(const Tensor& pointers, const Tensor& indices,
                                                      const Tensor& values, const Tensor& reduction,
                                                    const Scalar& fill_value) {
  // TODO: perform lots of error checking to check correct type and sizes of inputs. Check
  // SparseTensorImpl::set_indices_and_values_unsafe() for details
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

  TORCH_CHECK((((indices.dim() == pointers.dim()) == values.dim()) == reduction.dim()) == 1,
              "indices, pointers, values and reduction must have dim=1 but got indices.dim()=",
              indices.dim(), " pointers.dim()=", pointers.dim(), " values.dim()=", values.dim(), " reduction.dim()=", reduction.dim());
  
  pointers_ = pointers;
  indices_ = indices;
  values_ = values;
  reduction_ = reduction;
  fill_value_ = fill_value;

  AT_ASSERT(device() == values_.device());    
  AT_ASSERT(indices_.device() == values_.device());
  AT_ASSERT(reduction_.device() == values_.device());

  auto reduction_accessor = reduction_.accessor<int64_t, 1>();

  rsplit_dim_ = reduction_accessor[reduction_.size(0)-1];
    
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
