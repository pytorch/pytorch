// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/CUDAFloatType.h"
#include "ATen/CUDAFloatStorage.h"
#include "ATen/CUDAFloatTensor.h"
#include "ATen/CUDAGenerator.h"
#include "ATen/CUDAByteTensor.h"
#include "ATen/CUDAIntTensor.h"
#include "ATen/CUDALongTensor.h"
#include "ATen/SparseCUDAFloatTensor.h"
#include "ATen/CUDAFloatTensor.h"
#include "ATen/CUDALongTensor.h"
#include "ATen/Allocator.h"
#include "ATen/Utils.h"
#include "ATen/Half.h"
#include "ATen/WrapDimUtils.h"
#include "ATen/THLongStorageView.h"
#include "ATen/UndefinedTensor.h"
#include "ATen/NativeFunctions.h"
#include <iostream>
#include <sstream>

#include "ATen/Config.h"
#if AT_CUDA_ENABLED()
#include <ATen/cuda/CUDAHalf.cuh>
#endif

namespace at {

CUDAFloatType::CUDAFloatType(Context* context)
: Type(context, /*is_variable_or_undefined=*/false) {}
ScalarType CUDAFloatType::scalarType() const {
  return ScalarType::Float;
}
Backend CUDAFloatType::backend() const {
  return Backend::CUDA;
}
bool CUDAFloatType::is_cuda() const { return backend() == kCUDA || backend() == kSparseCUDA; }
bool CUDAFloatType::is_sparse() const { return backend() == kSparseCPU || backend() == kSparseCUDA; }
bool CUDAFloatType::is_distributed() const { return false; }

std::unique_ptr<Storage> CUDAFloatType::storage() const {
  return std::unique_ptr<Storage>(new CUDAFloatStorage(context));
}
std::unique_ptr<Storage> CUDAFloatType::storage(size_t size) const {
  return std::unique_ptr<Storage>(new CUDAFloatStorage(context,size));
}
std::unique_ptr<Storage> CUDAFloatType::storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const {
    return std::unique_ptr<Storage>(
      new CUDAFloatStorage(context,data,size,deleter));
}
std::unique_ptr<Storage> CUDAFloatType::storageWithAllocator(int64_t size, std::unique_ptr<Allocator> allocator) const {
    return std::unique_ptr<Storage>(
        new CUDAFloatStorage(context, size, std::move(allocator)));
}
Tensor CUDAFloatType::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THCudaTensor_retain(context->thc_state,  (THCudaTensor*) th_pointer);
  return Tensor(new CUDAFloatTensor(context,(THCudaTensor*)(th_pointer)), false);
}
std::unique_ptr<Storage> CUDAFloatType::unsafeStorageFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THCudaStorage_retain(context->thc_state,  (THCudaStorage*) th_pointer);
  return std::unique_ptr<Storage>(new CUDAFloatStorage(context, (THCudaStorage*) th_pointer));
}
std::unique_ptr<Generator> CUDAFloatType::generator() const {
  return std::unique_ptr<Generator>(new CUDAGenerator(context));
}

const char * CUDAFloatType::toString() const {
  return CUDAFloatType::typeString();
}
TypeID CUDAFloatType::ID() const {
  return TypeID::CUDAFloat;
}

std::size_t CUDAFloatType::elementSizeInBytes() const {
  return sizeof(float);
}

const char * CUDAFloatType::typeString() {
  return "CUDAFloatType";
}

/* example
Tensor * CUDAFloatType::add(Tensor & a, Tensor & b) {
  std::cout << "add CUDAFloatTensor\n";
  return &a;
}
*/

int64_t CUDAFloatType::storage_offset(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    return static_cast<int64_t>(THCudaTensor_storageOffset(context->thc_state, self_->tensor));
}
Tensor & CUDAFloatType::resize_(Tensor & self, IntList size) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    THCudaTensor_resize(context->thc_state, self_->tensor, size_, NULL);
    self_->maybeScalar(size.size() == 0);
    return self;
}
int64_t CUDAFloatType::numel(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    return static_cast<int64_t>(THCudaTensor_nElement(context->thc_state, self_->tensor));
}
Tensor & CUDAFloatType::set_(Tensor & self, Storage & storage) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto storage_ = checked_cast_storage<CUDAFloatStorage>(&storage,"storage",2);
    THCudaTensor_setStorage(context->thc_state, self_->tensor, storage_->storage, 0, THLongStorageView::makeFromLength(static_cast<int64_t>(storage.size())), NULL);
    self_->maybeScalar(false);
    return self;
}
Tensor & CUDAFloatType::set_(Tensor & self, Storage & sourceStorage, int64_t storage_offset, IntList size, IntList stride) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto sourceStorage_ = checked_cast_storage<CUDAFloatStorage>(&sourceStorage,"sourceStorage",2);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    THCudaTensor_setStorage(context->thc_state, self_->tensor, sourceStorage_->storage, storage_offset, size_, stride_);
    self_->maybeScalar(size.size() == 0);
    return self;
}
Tensor & CUDAFloatType::set_(Tensor & self, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto source_ = checked_cast_tensor<CUDAFloatTensor>(source.pImpl,"source",2, false);
    THCudaTensor_set(context->thc_state, self_->tensor, source_->tensor);
    self_->maybeScalar(source_->isScalar());
    return self;
}
Tensor & CUDAFloatType::set_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_setStorage(context->thc_state, self_->tensor, NULL, 0, NULL, NULL);
    self_->maybeScalar(false);
    return self;
}
Tensor & CUDAFloatType::fill_(Tensor & self, Scalar value) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toFloat();
    THCudaTensor_fill(context->thc_state, self_->tensor, value_);
    return self;
}
Tensor & CUDAFloatType::fill_(Tensor & self, const Tensor & value) const {
    if (value.dim() == 0) {
        return static_cast<const Type*>(this)->fill_(self, Scalar(value));
    }
    AT_ERROR("fill_ only supports a 0-dimensional value tensor, but got tensor "
        "with %" PRId64 " dimension(s)", value.dim());
}
bool CUDAFloatType::is_contiguous(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    return THCudaTensor_isContiguous(context->thc_state, self_->tensor);
}
bool CUDAFloatType::is_set_to(const Tensor & self, const Tensor & tensor) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto tensor_ = checked_cast_tensor<CUDAFloatTensor>(tensor.pImpl,"tensor",2, false);
    return THCudaTensor_isSetTo(context->thc_state, self_->tensor, tensor_->tensor);
}
Tensor & CUDAFloatType::s_masked_fill_(Tensor & self, const Tensor & mask, Scalar value) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CUDAByteTensor>(mask.pImpl,"mask",2, false);
    auto value_ = value.toFloat();
    THCudaTensor_maskedFill(context->thc_state, self_->tensor, mask_->tensor, value_);
    return self;
}
Tensor & CUDAFloatType::s_masked_fill_(Tensor & self, const Tensor & mask, const Tensor & value) const {
    if (value.dim() == 0) {
        return static_cast<const Type*>(this)->masked_fill_(self, mask, Scalar(value));
    }
    AT_ERROR("masked_fill_ only supports a 0-dimensional value tensor, but got tensor "
        "with %" PRId64 " dimension(s)", value.dim());
}
Tensor & CUDAFloatType::s_masked_scatter_(Tensor & self, const Tensor & mask, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CUDAByteTensor>(mask.pImpl,"mask",2, false);
    auto source_ = checked_cast_tensor<CUDAFloatTensor>(source.pImpl,"source",3, false);
    THCudaTensor_maskedCopy(context->thc_state, self_->tensor, mask_->tensor, source_->tensor);
    return self;
}
Tensor & CUDAFloatType::s_masked_select_out(Tensor & result, const Tensor & self, const Tensor & mask) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CUDAByteTensor>(mask.pImpl,"mask",2, false);
    THCudaTensor_maskedSelect(context->thc_state, result_->tensor, self_->tensor, mask_->tensor);
    result_->maybeScalar(self_->isScalar() && mask_->isScalar());
    return result;
}
Tensor CUDAFloatType::s_masked_select(const Tensor & self, const Tensor & mask) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CUDAByteTensor>(mask.pImpl,"mask",2, false);
    THCudaTensor_maskedSelect(context->thc_state, result_->tensor, self_->tensor, mask_->tensor);
    result_->maybeScalar(self_->isScalar() && mask_->isScalar());
    return result;
}
Tensor CUDAFloatType::transpose(const Tensor & self, int64_t dim0, int64_t dim1) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim0 = maybe_wrap_dim(dim0, self_);
    dim1 = maybe_wrap_dim(dim1, self_);
    return Tensor((new CUDAFloatTensor(context, THCudaTensor_newTranspose(context->thc_state, self_->tensor, dim0, dim1)))->maybeScalar(self_->isScalar()),false);
}
Tensor CUDAFloatType::t(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    return Tensor((new CUDAFloatTensor(context, THCudaTensor_newTranspose(context->thc_state, self_->tensor, 0, 1)))->maybeScalar(self_->isScalar()),false);
}
Tensor & CUDAFloatType::nonzero_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDALongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_nonzero(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::nonzero(const Tensor & self) const {
    auto result_ = new CUDALongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_nonzero(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::contiguous(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    return Tensor((new CUDAFloatTensor(context, THCudaTensor_newContiguous(context->thc_state, self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor CUDAFloatType::clone(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    return Tensor((new CUDAFloatTensor(context, THCudaTensor_newClone(context->thc_state, self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor CUDAFloatType::view(const Tensor & self, IntList size) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new CUDAFloatTensor(context, THCudaTensor_newView(context->thc_state, self_->tensor, size_)))->maybeScalar(size.size() == 0),false);
}
Tensor & CUDAFloatType::resize_as_(Tensor & self, const Tensor & the_template) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto the_template_ = checked_cast_tensor<CUDAFloatTensor>(the_template.pImpl,"the_template",2, false);
    THCudaTensor_resizeAs(context->thc_state, self_->tensor, the_template_->tensor);
    self_->maybeScalar(the_template_->isScalar());
    return self;
}
Tensor & CUDAFloatType::index_select_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    THCudaTensor_indexSelect(context->thc_state, result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
Tensor CUDAFloatType::index_select(const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    THCudaTensor_indexSelect(context->thc_state, result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
Tensor & CUDAFloatType::_indexCopy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    auto source_ = checked_cast_tensor<CUDAFloatTensor>(source.pImpl,"source",4, false);
    THCudaTensor_indexCopy(context->thc_state, self_->tensor, dim, index_->tensor, source_->tensor);
    return self;
}
Tensor & CUDAFloatType::take_out(Tensor & result, const Tensor & self, const Tensor & index) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",2, false);
    THCudaTensor_take(context->thc_state, result_->tensor, self_->tensor, index_->tensor);
    result_->maybeScalar(index_->isScalar());
    return result;
}
Tensor CUDAFloatType::take(const Tensor & self, const Tensor & index) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",2, false);
    THCudaTensor_take(context->thc_state, result_->tensor, self_->tensor, index_->tensor);
    result_->maybeScalar(index_->isScalar());
    return result;
}
Tensor & CUDAFloatType::put_(Tensor & self, const Tensor & index, const Tensor & source, bool accumulate) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",2, false);
    auto source_ = checked_cast_tensor<CUDAFloatTensor>(source.pImpl,"source",3, false);
    THCudaTensor_put(context->thc_state, self_->tensor, index_->tensor, source_->tensor, accumulate);
    return self;
}
Tensor & CUDAFloatType::index_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    auto source_ = checked_cast_tensor<CUDAFloatTensor>(source.pImpl,"source",4, false);
    THCudaTensor_indexAdd(context->thc_state, self_->tensor, dim, index_->tensor, source_->tensor);
    return self;
}
Tensor & CUDAFloatType::index_fill_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    auto value_ = value.toFloat();
    THCudaTensor_indexFill(context->thc_state, self_->tensor, dim, index_->tensor, value_);
    return self;
}
Tensor & CUDAFloatType::index_fill_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & value) const {
    if (value.dim() == 0) {
        return static_cast<const Type*>(this)->index_fill_(self, dim, index, Scalar(value));
    }
    AT_ERROR("index_fill_ only supports a 0-dimensional value tensor, but got tensor "
        "with %" PRId64 " dimension(s)", value.dim());
}
Tensor CUDAFloatType::unfold(const Tensor & self, int64_t dimension, int64_t size, int64_t step) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dimension = maybe_wrap_dim(dimension, self_);
    THCudaTensor_unfold(context->thc_state, result_->tensor, self_->tensor, dimension, size, step);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::_range_out(Tensor & result, Scalar start, Scalar end, Scalar step) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto start_ = start.toDouble();
    auto end_ = end.toDouble();
    auto step_ = step.toDouble();
    THCudaTensor_range(context->thc_state, result_->tensor, start_, end_, step_);
    return result;
}
Tensor CUDAFloatType::_range(Scalar start, Scalar end, Scalar step) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto start_ = start.toDouble();
    auto end_ = end.toDouble();
    auto step_ = step.toDouble();
    THCudaTensor_range(context->thc_state, result_->tensor, start_, end_, step_);
    return result;
}
Tensor & CUDAFloatType::_arange_out(Tensor & result, Scalar start, Scalar end, Scalar step) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto start_ = start.toDouble();
    auto end_ = end.toDouble();
    auto step_ = step.toDouble();
    THCudaTensor_arange(context->thc_state, result_->tensor, start_, end_, step_);
    return result;
}
Tensor CUDAFloatType::_arange(Scalar start, Scalar end, Scalar step) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto start_ = start.toDouble();
    auto end_ = end.toDouble();
    auto step_ = step.toDouble();
    THCudaTensor_arange(context->thc_state, result_->tensor, start_, end_, step_);
    return result;
}
Tensor & CUDAFloatType::_arange_out(Tensor & result, Scalar end) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto end_ = end.toDouble();
    THCudaTensor_arange(context->thc_state, result_->tensor, 0, end_, 1);
    return result;
}
Tensor CUDAFloatType::_arange(Scalar end) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto end_ = end.toDouble();
    THCudaTensor_arange(context->thc_state, result_->tensor, 0, end_, 1);
    return result;
}
Tensor & CUDAFloatType::scatter_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
    if (src.dim() == 0) {
        return static_cast<const Type*>(this)->scatter_(self, dim, index, Scalar(src));
    }
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    auto src_ = checked_cast_tensor<CUDAFloatTensor>(src.pImpl,"src",4, false);
    THCudaTensor_scatter(context->thc_state, self_->tensor, dim, index_->tensor, src_->tensor);
    return self;
}
Tensor & CUDAFloatType::scatter_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    auto value_ = value.toFloat();
    THCudaTensor_scatterFill(context->thc_state, self_->tensor, dim, index_->tensor, value_);
    return self;
}
Tensor & CUDAFloatType::scatter_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    auto src_ = checked_cast_tensor<CUDAFloatTensor>(src.pImpl,"src",4, false);
    THCudaTensor_scatterAdd(context->thc_state, self_->tensor, dim, index_->tensor, src_->tensor);
    return self;
}
Tensor & CUDAFloatType::gather_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    result.resize_(index.sizes());
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    THCudaTensor_gather(context->thc_state, result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
Tensor CUDAFloatType::gather(const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    result.resize_(index.sizes());
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    THCudaTensor_gather(context->thc_state, result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
void* CUDAFloatType::data_ptr(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    return THCudaTensor_data(context->thc_state, self_->tensor);
}
bool CUDAFloatType::equal(const Tensor & self, const Tensor & other) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",2, false);
    return THCudaTensor_equal(context->thc_state, self_->tensor, other_->tensor);
}
Tensor & CUDAFloatType::__and___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_bitand(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::__and__(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_bitand(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::s___and___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__and___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",2, false);
    THCudaTensor_cbitand(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAFloatType::s___and__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__and__(self, Scalar(other));
    }
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",2, false);
    THCudaTensor_cbitand(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAFloatType::__iand__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_bitand(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAFloatType::s___iand__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__iand__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",3, false);
    THCudaTensor_cbitand(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAFloatType::__or___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_bitor(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::__or__(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_bitor(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::s___or___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__or___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",2, false);
    THCudaTensor_cbitor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAFloatType::s___or__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__or__(self, Scalar(other));
    }
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",2, false);
    THCudaTensor_cbitor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAFloatType::__ior__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_bitor(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAFloatType::s___ior__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__ior__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",3, false);
    THCudaTensor_cbitor(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAFloatType::__xor___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_bitxor(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::__xor__(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_bitxor(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::s___xor___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__xor___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",2, false);
    THCudaTensor_cbitxor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAFloatType::s___xor__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__xor__(self, Scalar(other));
    }
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",2, false);
    THCudaTensor_cbitxor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAFloatType::__ixor__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_bitxor(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAFloatType::s___ixor__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__ixor__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",3, false);
    THCudaTensor_cbitxor(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAFloatType::__lshift___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_lshift(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::__lshift__(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_lshift(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::s___lshift___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__lshift___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",2, false);
    THCudaTensor_clshift(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAFloatType::s___lshift__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__lshift__(self, Scalar(other));
    }
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",2, false);
    THCudaTensor_clshift(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAFloatType::__ilshift__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_lshift(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAFloatType::s___ilshift__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__ilshift__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",3, false);
    THCudaTensor_clshift(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAFloatType::__rshift___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_rshift(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::__rshift__(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_rshift(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::s___rshift___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__rshift___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",2, false);
    THCudaTensor_crshift(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAFloatType::s___rshift__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__rshift__(self, Scalar(other));
    }
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",2, false);
    THCudaTensor_crshift(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAFloatType::__irshift__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_rshift(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAFloatType::s___irshift__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__irshift__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",3, false);
    THCudaTensor_crshift(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAFloatType::lt_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_ltValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::lt(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_ltValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::s_lt_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->lt_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",2, false);
    THCudaTensor_ltTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAFloatType::s_lt(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->lt(self, Scalar(other));
    }
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",2, false);
    THCudaTensor_ltTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAFloatType::lt_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_ltValueT(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAFloatType::s_lt_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->lt_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",3, false);
    THCudaTensor_ltTensorT(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAFloatType::gt_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_gtValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::gt(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_gtValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::s_gt_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->gt_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",2, false);
    THCudaTensor_gtTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAFloatType::s_gt(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->gt(self, Scalar(other));
    }
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",2, false);
    THCudaTensor_gtTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAFloatType::gt_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_gtValueT(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAFloatType::s_gt_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->gt_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",3, false);
    THCudaTensor_gtTensorT(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAFloatType::le_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_leValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::le(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_leValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::s_le_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->le_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",2, false);
    THCudaTensor_leTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAFloatType::s_le(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->le(self, Scalar(other));
    }
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",2, false);
    THCudaTensor_leTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAFloatType::le_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_leValueT(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAFloatType::s_le_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->le_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",3, false);
    THCudaTensor_leTensorT(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAFloatType::ge_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_geValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::ge(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_geValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::s_ge_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ge_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",2, false);
    THCudaTensor_geTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAFloatType::s_ge(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ge(self, Scalar(other));
    }
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",2, false);
    THCudaTensor_geTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAFloatType::ge_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_geValueT(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAFloatType::s_ge_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ge_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",3, false);
    THCudaTensor_geTensorT(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAFloatType::eq_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_eqValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::eq(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_eqValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::s_eq_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->eq_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",2, false);
    THCudaTensor_eqTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAFloatType::s_eq(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->eq(self, Scalar(other));
    }
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",2, false);
    THCudaTensor_eqTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAFloatType::eq_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_eqValueT(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAFloatType::s_eq_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->eq_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",3, false);
    THCudaTensor_eqTensorT(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAFloatType::ne_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_neValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::ne(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_neValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::s_ne_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ne_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",2, false);
    THCudaTensor_neTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAFloatType::s_ne(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ne(self, Scalar(other));
    }
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",2, false);
    THCudaTensor_neTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAFloatType::ne_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_neValueT(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAFloatType::s_ne_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ne_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",3, false);
    THCudaTensor_neTensorT(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
std::tuple<Tensor &,Tensor &> CUDAFloatType::min_out(Tensor & min, Tensor & min_indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto min_ = checked_cast_tensor<CUDAFloatTensor>(min.pImpl,"min",0, false);
    auto min_indices_ = checked_cast_tensor<CUDALongTensor>(min_indices.pImpl,"min_indices",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaTensor_min(context->thc_state, min_->tensor, min_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    min_->maybeScalar(maybe_scalar);
    min_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(min, min_indices);
}
std::tuple<Tensor,Tensor> CUDAFloatType::min(const Tensor & self, int64_t dim, bool keepdim) const {
    auto min_ = new CUDAFloatTensor(context);
    auto min = Tensor(min_, false);
    auto min_indices_ = new CUDALongTensor(context);
    auto min_indices = Tensor(min_indices_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaTensor_min(context->thc_state, min_->tensor, min_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    min_->maybeScalar(maybe_scalar);
    min_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(min, min_indices);
}
Tensor & CUDAFloatType::s_min_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",2, false);
    THCudaTensor_cmin(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAFloatType::s_min(const Tensor & self, const Tensor & other) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",2, false);
    THCudaTensor_cmin(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAFloatType::min(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<float>(THCudaTensor_minall(context->thc_state, self_->tensor)));
}
std::tuple<Tensor &,Tensor &> CUDAFloatType::max_out(Tensor & max, Tensor & max_indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto max_ = checked_cast_tensor<CUDAFloatTensor>(max.pImpl,"max",0, false);
    auto max_indices_ = checked_cast_tensor<CUDALongTensor>(max_indices.pImpl,"max_indices",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaTensor_max(context->thc_state, max_->tensor, max_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    max_->maybeScalar(maybe_scalar);
    max_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(max, max_indices);
}
std::tuple<Tensor,Tensor> CUDAFloatType::max(const Tensor & self, int64_t dim, bool keepdim) const {
    auto max_ = new CUDAFloatTensor(context);
    auto max = Tensor(max_, false);
    auto max_indices_ = new CUDALongTensor(context);
    auto max_indices = Tensor(max_indices_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaTensor_max(context->thc_state, max_->tensor, max_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    max_->maybeScalar(maybe_scalar);
    max_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(max, max_indices);
}
Tensor & CUDAFloatType::s_max_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",2, false);
    THCudaTensor_cmax(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAFloatType::s_max(const Tensor & self, const Tensor & other) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",2, false);
    THCudaTensor_cmax(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAFloatType::max(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<float>(THCudaTensor_maxall(context->thc_state, self_->tensor)));
}
std::tuple<Tensor &,Tensor &> CUDAFloatType::mode_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = checked_cast_tensor<CUDAFloatTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaTensor_mode(context->thc_state, values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CUDAFloatType::mode(const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = new CUDAFloatTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CUDALongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaTensor_mode(context->thc_state, values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
std::tuple<Tensor &,Tensor &> CUDAFloatType::median_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = checked_cast_tensor<CUDAFloatTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaTensor_median(context->thc_state, values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CUDAFloatType::median(const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = new CUDAFloatTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CUDALongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaTensor_median(context->thc_state, values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
Tensor CUDAFloatType::median(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<float>(THCudaTensor_medianall(context->thc_state, self_->tensor)));
}
std::tuple<Tensor &,Tensor &> CUDAFloatType::sort_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool descending) const {
    auto values_ = checked_cast_tensor<CUDAFloatTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaTensor_sort(context->thc_state, values_->tensor, indices_->tensor, self_->tensor, dim, descending);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CUDAFloatType::sort(const Tensor & self, int64_t dim, bool descending) const {
    auto values_ = new CUDAFloatTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CUDALongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaTensor_sort(context->thc_state, values_->tensor, indices_->tensor, self_->tensor, dim, descending);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
std::tuple<Tensor &,Tensor &> CUDAFloatType::topk_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) const {
    auto values_ = checked_cast_tensor<CUDAFloatTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaTensor_topk(context->thc_state, values_->tensor, indices_->tensor, self_->tensor, k, dim, largest, sorted);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CUDAFloatType::topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) const {
    auto values_ = new CUDAFloatTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CUDALongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaTensor_topk(context->thc_state, values_->tensor, indices_->tensor, self_->tensor, k, dim, largest, sorted);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
int64_t CUDAFloatType::get_device(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    return static_cast<int64_t>(THCudaTensor_getDevice(context->thc_state, self_->tensor));
}
Tensor & CUDAFloatType::_abs_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_abs(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::_abs(const Tensor & self) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_abs(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::sigmoid_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_sigmoid(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAFloatType::sigmoid_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_sigmoid(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::sigmoid(const Tensor & self) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_sigmoid(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::_log_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_log(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::_log(const Tensor & self) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_log(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::log10_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_log10(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAFloatType::log10_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_log10(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::log10(const Tensor & self) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_log10(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::log1p_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_log1p(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAFloatType::log1p_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_log1p(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::log1p(const Tensor & self) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_log1p(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::log2_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_log2(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAFloatType::log2_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_log2(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::log2(const Tensor & self) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_log2(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::lgamma_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_lgamma(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::lgamma(const Tensor & self) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_lgamma(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::lgamma_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_lgamma(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAFloatType::digamma_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_digamma(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::digamma(const Tensor & self) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_digamma(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::digamma_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_digamma(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAFloatType::polygamma_out(Tensor & result, int64_t n, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    THCudaTensor_polygamma(context->thc_state, result_->tensor, n, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::polygamma(int64_t n, const Tensor & self) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    THCudaTensor_polygamma(context->thc_state, result_->tensor, n, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::polygamma_(Tensor & self, int64_t n) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_polygamma(context->thc_state, self_->tensor, n, self_->tensor);
    return self;
}
Tensor & CUDAFloatType::_exp_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_exp(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::_exp(const Tensor & self) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_exp(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::expm1_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_expm1(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAFloatType::expm1_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_expm1(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::expm1(const Tensor & self) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_expm1(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::_cos_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_cos(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::_cos(const Tensor & self) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_cos(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::acos_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_acos(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAFloatType::acos_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_acos(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::acos(const Tensor & self) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_acos(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::cosh_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_cosh(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAFloatType::cosh_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_cosh(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::cosh(const Tensor & self) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_cosh(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::_sin_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_sin(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::_sin(const Tensor & self) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_sin(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::asin_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_asin(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAFloatType::asin_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_asin(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::asin(const Tensor & self) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_asin(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::sinh_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_sinh(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAFloatType::sinh_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_sinh(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::sinh(const Tensor & self) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_sinh(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::tan_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_tan(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAFloatType::tan_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_tan(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::tan(const Tensor & self) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_tan(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::atan_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_atan(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAFloatType::atan_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_atan(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::atan(const Tensor & self) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_atan(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::tanh_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_tanh(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAFloatType::tanh_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_tanh(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::tanh(const Tensor & self) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_tanh(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::erf_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_erf(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAFloatType::erf_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_erf(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::erf(const Tensor & self) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_erf(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::erfinv_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_erfinv(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAFloatType::erfinv_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_erfinv(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::erfinv(const Tensor & self) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_erfinv(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::_sqrt_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_sqrt(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::_sqrt(const Tensor & self) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_sqrt(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::rsqrt_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_rsqrt(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAFloatType::rsqrt_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_rsqrt(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::rsqrt(const Tensor & self) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_rsqrt(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::_ceil_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_ceil(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::_ceil(const Tensor & self) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_ceil(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::_floor_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_floor(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::_floor(const Tensor & self) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_floor(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::_round_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_round(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::_round(const Tensor & self) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_round(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::_trunc_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_trunc(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::_trunc(const Tensor & self) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_trunc(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::frac_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_frac(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAFloatType::frac_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_frac(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::frac(const Tensor & self) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_frac(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::mean_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaTensor_mean(context->thc_state, result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CUDAFloatType::mean(const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaTensor_mean(context->thc_state, result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CUDAFloatType::mean(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<float>(THCudaTensor_meanall(context->thc_state, self_->tensor)));
}
Tensor & CUDAFloatType::var_out(Tensor & result, const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaTensor_var(context->thc_state, result_->tensor, self_->tensor, dim, (unbiased) ? 0 : 1, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CUDAFloatType::var(const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaTensor_var(context->thc_state, result_->tensor, self_->tensor, dim, (unbiased) ? 0 : 1, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CUDAFloatType::var(const Tensor & self, bool unbiased) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<float>(THCudaTensor_varall(context->thc_state, self_->tensor, (unbiased) ? 0 : 1)));
}
Tensor & CUDAFloatType::std_out(Tensor & result, const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaTensor_std(context->thc_state, result_->tensor, self_->tensor, dim, (unbiased) ? 0 : 1, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CUDAFloatType::std(const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaTensor_std(context->thc_state, result_->tensor, self_->tensor, dim, (unbiased) ? 0 : 1, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CUDAFloatType::std(const Tensor & self, bool unbiased) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<float>(THCudaTensor_stdall(context->thc_state, self_->tensor, (unbiased) ? 0 : 1)));
}
Tensor & CUDAFloatType::norm_out(Tensor & result, const Tensor & self, Scalar p, int64_t dim, bool keepdim) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto p_ = p.toFloat();
    dim = maybe_wrap_dim(dim, self_);
    THCudaTensor_norm(context->thc_state, result_->tensor, self_->tensor, p_, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CUDAFloatType::norm(const Tensor & self, Scalar p, int64_t dim, bool keepdim) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto p_ = p.toFloat();
    dim = maybe_wrap_dim(dim, self_);
    THCudaTensor_norm(context->thc_state, result_->tensor, self_->tensor, p_, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CUDAFloatType::norm(const Tensor & self, Scalar p) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto p_ = p.toFloat();
    // norm(value) for a sparse tensor returns a DENSE 0-dim tensor
    if (self.is_sparse()) {
      auto result = THCudaTensor_normall(context->thc_state,  self_->tensor, convert<float>(p_));
      return toBackend(toDense(backend())).tensor({}).fill_(result);
    }
    // aten_custom_call is followed by the generated call to normall
    return scalarTensor(convert<float>(THCudaTensor_normall(context->thc_state, self_->tensor, p_)));
}
Tensor & CUDAFloatType::renorm_out(Tensor & result, const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto p_ = p.toFloat();
    dim = maybe_wrap_dim(dim, self_);
    auto maxnorm_ = maxnorm.toFloat();
    THCudaTensor_renorm(context->thc_state, result_->tensor, self_->tensor, p_, dim, maxnorm_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::renorm(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto p_ = p.toFloat();
    dim = maybe_wrap_dim(dim, self_);
    auto maxnorm_ = maxnorm.toFloat();
    THCudaTensor_renorm(context->thc_state, result_->tensor, self_->tensor, p_, dim, maxnorm_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::renorm_(Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto p_ = p.toFloat();
    dim = maybe_wrap_dim(dim, self_);
    auto maxnorm_ = maxnorm.toFloat();
    THCudaTensor_renorm(context->thc_state, self_->tensor, self_->tensor, p_, dim, maxnorm_);
    return self;
}
Tensor CUDAFloatType::s_dist(const Tensor & self, const Tensor & other, Scalar p) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",2, false);
    auto p_ = p.toFloat();
    return scalarTensor(convert<float>(THCudaTensor_dist(context->thc_state, self_->tensor, other_->tensor, p_)));
}
Tensor & CUDAFloatType::reciprocal_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_cinv(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::reciprocal(const Tensor & self) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_cinv(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::reciprocal_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_cinv(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAFloatType::neg_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_neg(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::neg(const Tensor & self) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_neg(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::neg_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_neg(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAFloatType::s_atan2_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",2, false);
    THCudaTensor_atan2(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAFloatType::s_atan2(const Tensor & self, const Tensor & other) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",2, false);
    THCudaTensor_atan2(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAFloatType::s_atan2_(Tensor & self, const Tensor & other) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",3, false);
    THCudaTensor_atan2(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAFloatType::pow_out(Tensor & result, const Tensor & self, Scalar exponent) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toFloat();
    THCudaTensor_pow(context->thc_state, result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::pow(const Tensor & self, Scalar exponent) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toFloat();
    THCudaTensor_pow(context->thc_state, result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::s_pow_out(Tensor & result, const Tensor & self, const Tensor & exponent) const {
    if (exponent.dim() == 0) {
        return static_cast<const Type*>(this)->pow_out(result, self, Scalar(exponent));
    }
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto exponent_ = checked_cast_tensor<CUDAFloatTensor>(exponent.pImpl,"exponent",2, false);
    THCudaTensor_cpow(context->thc_state, result_->tensor, self_->tensor, exponent_->tensor);
    result_->maybeScalar(self_->isScalar() && exponent_->isScalar());
    return result;
}
Tensor CUDAFloatType::s_pow(const Tensor & self, const Tensor & exponent) const {
    if (exponent.dim() == 0) {
        return static_cast<const Type*>(this)->pow(self, Scalar(exponent));
    }
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto exponent_ = checked_cast_tensor<CUDAFloatTensor>(exponent.pImpl,"exponent",2, false);
    THCudaTensor_cpow(context->thc_state, result_->tensor, self_->tensor, exponent_->tensor);
    result_->maybeScalar(self_->isScalar() && exponent_->isScalar());
    return result;
}
Tensor & CUDAFloatType::pow_out(Tensor & result, Scalar base, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto base_ = base.toFloat();
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    THCudaTensor_tpow(context->thc_state, result_->tensor, base_, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::pow(Scalar base, const Tensor & self) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto base_ = base.toFloat();
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    THCudaTensor_tpow(context->thc_state, result_->tensor, base_, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::pow_(Tensor & self, Scalar exponent) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toFloat();
    THCudaTensor_pow(context->thc_state, self_->tensor, self_->tensor, exponent_);
    return self;
}
Tensor & CUDAFloatType::s_pow_(Tensor & self, const Tensor & exponent) const {
    if (exponent.dim() == 0) {
        return static_cast<const Type*>(this)->pow_(self, Scalar(exponent));
    }
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto exponent_ = checked_cast_tensor<CUDAFloatTensor>(exponent.pImpl,"exponent",3, false);
    THCudaTensor_cpow(context->thc_state, self_->tensor, self_->tensor, exponent_->tensor);
    return self;
}
Tensor & CUDAFloatType::s_lerp_out(Tensor & result, const Tensor & self, const Tensor & end, Scalar weight) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto end_ = checked_cast_tensor<CUDAFloatTensor>(end.pImpl,"end",2, false);
    auto weight_ = weight.toFloat();
    THCudaTensor_lerp(context->thc_state, result_->tensor, self_->tensor, end_->tensor, weight_);
    result_->maybeScalar(self_->isScalar() && end_->isScalar());
    return result;
}
Tensor CUDAFloatType::s_lerp(const Tensor & self, const Tensor & end, Scalar weight) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto end_ = checked_cast_tensor<CUDAFloatTensor>(end.pImpl,"end",2, false);
    auto weight_ = weight.toFloat();
    THCudaTensor_lerp(context->thc_state, result_->tensor, self_->tensor, end_->tensor, weight_);
    result_->maybeScalar(self_->isScalar() && end_->isScalar());
    return result;
}
Tensor & CUDAFloatType::s_lerp_(Tensor & self, const Tensor & end, Scalar weight) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto end_ = checked_cast_tensor<CUDAFloatTensor>(end.pImpl,"end",3, false);
    auto weight_ = weight.toFloat();
    THCudaTensor_lerp(context->thc_state, self_->tensor, self_->tensor, end_->tensor, weight_);
    return self;
}
Tensor & CUDAFloatType::zero_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_zero(context->thc_state, self_->tensor);
    return self;
}
Tensor CUDAFloatType::_sumall(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<float>(THCudaTensor_sumall(context->thc_state, self_->tensor)));
}
Tensor & CUDAFloatType::_sum_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaTensor_sum(context->thc_state, result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CUDAFloatType::_sum(const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaTensor_sum(context->thc_state, result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CUDAFloatType::_prodall(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<float>(THCudaTensor_prodall(context->thc_state, self_->tensor)));
}
Tensor & CUDAFloatType::_prod_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaTensor_prod(context->thc_state, result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CUDAFloatType::_prod(const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaTensor_prod(context->thc_state, result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor & CUDAFloatType::cumsum_out(Tensor & result, const Tensor & self, int64_t dim) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaTensor_cumsum(context->thc_state, result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::cumsum(const Tensor & self, int64_t dim) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaTensor_cumsum(context->thc_state, result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::cumprod_out(Tensor & result, const Tensor & self, int64_t dim) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaTensor_cumprod(context->thc_state, result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::cumprod(const Tensor & self, int64_t dim) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaTensor_cumprod(context->thc_state, result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::sign_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_sign(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::sign(const Tensor & self) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_sign(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::sign_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_sign(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor CUDAFloatType::trace(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<float>(THCudaTensor_trace(context->thc_state, self_->tensor)));
}
Tensor & CUDAFloatType::add_out(Tensor & result, const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    auto alpha_ = alpha.toFloat();
    THCudaTensor_add_scaled(context->thc_state, result_->tensor, self_->tensor, other_, alpha_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::add(const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    auto alpha_ = alpha.toFloat();
    THCudaTensor_add_scaled(context->thc_state, result_->tensor, self_->tensor, other_, alpha_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::s_add_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if(other.type().is_sparse()) {
        return static_cast<const Type*>(this)->add_out(result, self, SparseTensor(other), alpha);
    }
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toFloat();
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",3, false);
    THCudaTensor_cadd(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAFloatType::s_add(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if(other.type().is_sparse()) {
        return static_cast<const Type*>(this)->add(self, SparseTensor(other), alpha);
    }
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add(self, Scalar(other), alpha);
    }
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toFloat();
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",3, false);
    THCudaTensor_cadd(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAFloatType::add_out(Tensor & result, const Tensor & self, SparseTensor other, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toFloat();
    auto other_ = checked_cast_tensor<SparseCUDAFloatTensor>(other.tref.pImpl,"other",3,false);
    THCSFloatTensor_spcadd(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::add(const Tensor & self, SparseTensor other, Scalar alpha) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toFloat();
    auto other_ = checked_cast_tensor<SparseCUDAFloatTensor>(other.tref.pImpl,"other",3,false);
    THCSFloatTensor_spcadd(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::add_(Tensor & self, Scalar other, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    auto alpha_ = alpha.toFloat();
    THCudaTensor_add_scaled(context->thc_state, self_->tensor, self_->tensor, other_, alpha_);
    return self;
}
Tensor & CUDAFloatType::s_add_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if(other.type().is_sparse()) {
        return static_cast<const Type*>(this)->add_(self, SparseTensor(other), alpha);
    }
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toFloat();
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",4, false);
    THCudaTensor_cadd(context->thc_state, self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & CUDAFloatType::add_(Tensor & self, SparseTensor other, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toFloat();
    auto other_ = checked_cast_tensor<SparseCUDAFloatTensor>(other.tref.pImpl,"other",4,false);
    THCSFloatTensor_spcadd(context->thc_state, self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & CUDAFloatType::sub_out(Tensor & result, const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    auto alpha_ = alpha.toFloat();
    THCudaTensor_sub_scaled(context->thc_state, result_->tensor, self_->tensor, other_, alpha_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::sub(const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    auto alpha_ = alpha.toFloat();
    THCudaTensor_sub_scaled(context->thc_state, result_->tensor, self_->tensor, other_, alpha_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::s_sub_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toFloat();
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",3, false);
    THCudaTensor_csub(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAFloatType::s_sub(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub(self, Scalar(other), alpha);
    }
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toFloat();
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",3, false);
    THCudaTensor_csub(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAFloatType::sub_(Tensor & self, Scalar other, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    auto alpha_ = alpha.toFloat();
    THCudaTensor_sub_scaled(context->thc_state, self_->tensor, self_->tensor, other_, alpha_);
    return self;
}
Tensor & CUDAFloatType::s_sub_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toFloat();
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",4, false);
    THCudaTensor_csub(context->thc_state, self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & CUDAFloatType::mul_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_mul(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::mul(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_mul(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::s_mul_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",2, false);
    THCudaTensor_cmul(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAFloatType::s_mul(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul(self, Scalar(other));
    }
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",2, false);
    THCudaTensor_cmul(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAFloatType::mul_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_mul(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAFloatType::s_mul_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",3, false);
    THCudaTensor_cmul(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAFloatType::div_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_div(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::div(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_div(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::s_div_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->div_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",2, false);
    THCudaTensor_cdiv(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAFloatType::s_div(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->div(self, Scalar(other));
    }
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",2, false);
    THCudaTensor_cdiv(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAFloatType::div_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_div(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAFloatType::s_div_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->div_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",3, false);
    THCudaTensor_cdiv(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAFloatType::fmod_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_fmod(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::fmod(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_fmod(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::s_fmod_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->fmod_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",2, false);
    THCudaTensor_cfmod(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAFloatType::s_fmod(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->fmod(self, Scalar(other));
    }
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",2, false);
    THCudaTensor_cfmod(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAFloatType::fmod_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_fmod(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAFloatType::s_fmod_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->fmod_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",3, false);
    THCudaTensor_cfmod(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAFloatType::remainder_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_remainder(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::remainder(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_remainder(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::s_remainder_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->remainder_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",2, false);
    THCudaTensor_cremainder(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAFloatType::s_remainder(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->remainder(self, Scalar(other));
    }
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",2, false);
    THCudaTensor_cremainder(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAFloatType::remainder_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THCudaTensor_remainder(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAFloatType::s_remainder_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->remainder_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",3, false);
    THCudaTensor_cremainder(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAFloatType::clamp_out(Tensor & result, const Tensor & self, Scalar min, Scalar max) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toFloat();
    auto max_ = max.toFloat();
    THCudaTensor_clamp(context->thc_state, result_->tensor, self_->tensor, min_, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::clamp(const Tensor & self, Scalar min, Scalar max) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toFloat();
    auto max_ = max.toFloat();
    THCudaTensor_clamp(context->thc_state, result_->tensor, self_->tensor, min_, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::clamp_(Tensor & self, Scalar min, Scalar max) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toFloat();
    auto max_ = max.toFloat();
    THCudaTensor_clamp(context->thc_state, self_->tensor, self_->tensor, min_, max_);
    return self;
}
Tensor & CUDAFloatType::clamp_min_out(Tensor & result, const Tensor & self, Scalar min) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toFloat();
    THCudaTensor_cmaxValue(context->thc_state, result_->tensor, self_->tensor, min_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::clamp_min(const Tensor & self, Scalar min) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toFloat();
    THCudaTensor_cmaxValue(context->thc_state, result_->tensor, self_->tensor, min_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::clamp_min_(Tensor & self, Scalar min) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toFloat();
    THCudaTensor_cmaxValue(context->thc_state, self_->tensor, self_->tensor, min_);
    return self;
}
Tensor & CUDAFloatType::clamp_max_out(Tensor & result, const Tensor & self, Scalar max) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto max_ = max.toFloat();
    THCudaTensor_cminValue(context->thc_state, result_->tensor, self_->tensor, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::clamp_max(const Tensor & self, Scalar max) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto max_ = max.toFloat();
    THCudaTensor_cminValue(context->thc_state, result_->tensor, self_->tensor, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::clamp_max_(Tensor & self, Scalar max) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto max_ = max.toFloat();
    THCudaTensor_cminValue(context->thc_state, self_->tensor, self_->tensor, max_);
    return self;
}
Tensor CUDAFloatType::_dot(const Tensor & self, const Tensor & tensor) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto tensor_ = checked_cast_tensor<CUDAFloatTensor>(tensor.pImpl,"tensor",2, false);
    return scalarTensor(convert<float>(THCudaTensor_dot(context->thc_state, self_->tensor, tensor_->tensor)));
}
Tensor & CUDAFloatType::tril_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_tril(context->thc_state, result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::tril(const Tensor & self, int64_t diagonal) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_tril(context->thc_state, result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::tril_(Tensor & self, int64_t diagonal) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_tril(context->thc_state, self_->tensor, self_->tensor, diagonal);
    return self;
}
Tensor & CUDAFloatType::triu_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_triu(context->thc_state, result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::triu(const Tensor & self, int64_t diagonal) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_triu(context->thc_state, result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::triu_(Tensor & self, int64_t diagonal) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_triu(context->thc_state, self_->tensor, self_->tensor, diagonal);
    return self;
}
Tensor & CUDAFloatType::cross_out(Tensor & result, const Tensor & self, const Tensor & other, int64_t dim) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",2, false);
    THCudaTensor_cross(context->thc_state, result_->tensor, self_->tensor, other_->tensor, dim);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAFloatType::cross(const Tensor & self, const Tensor & other, int64_t dim) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAFloatTensor>(other.pImpl,"other",2, false);
    THCudaTensor_cross(context->thc_state, result_->tensor, self_->tensor, other_->tensor, dim);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAFloatType::diag_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    if (self_->isScalar()) {
      throw std::runtime_error("Input must be 1-d or 2-d");
    }
    THCudaTensor_diag(context->thc_state, result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::diag(const Tensor & self, int64_t diagonal) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    if (self_->isScalar()) {
      throw std::runtime_error("Input must be 1-d or 2-d");
    }
    THCudaTensor_diag(context->thc_state, result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::s_addmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    if(mat1.type().is_sparse()) {
        return static_cast<const Type*>(this)->addmm_out(result, self, SparseTensor(mat1), mat2, beta, alpha);
    }
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toFloat();
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toFloat();
    auto mat1_ = checked_cast_tensor<CUDAFloatTensor>(mat1.pImpl,"mat1",4, false);
    auto mat2_ = checked_cast_tensor<CUDAFloatTensor>(mat2.pImpl,"mat2",5, false);
    THCudaTensor_addmm(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CUDAFloatType::s_addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    if(mat1.type().is_sparse()) {
        return static_cast<const Type*>(this)->addmm(self, SparseTensor(mat1), mat2, beta, alpha);
    }
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toFloat();
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toFloat();
    auto mat1_ = checked_cast_tensor<CUDAFloatTensor>(mat1.pImpl,"mat1",4, false);
    auto mat2_ = checked_cast_tensor<CUDAFloatTensor>(mat2.pImpl,"mat2",5, false);
    THCudaTensor_addmm(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CUDAFloatType::addmm_out(Tensor & result, const Tensor & self, SparseTensor mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toFloat();
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toFloat();
    auto mat1_ = checked_cast_tensor<SparseCUDAFloatTensor>(mat1.tref.pImpl,"mat1",4,false);
    auto mat2_ = checked_cast_tensor<CUDAFloatTensor>(mat2.pImpl,"mat2",5, false);
    THCSFloatTensor_spaddmm(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CUDAFloatType::addmm(const Tensor & self, SparseTensor mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toFloat();
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toFloat();
    auto mat1_ = checked_cast_tensor<SparseCUDAFloatTensor>(mat1.tref.pImpl,"mat1",4,false);
    auto mat2_ = checked_cast_tensor<CUDAFloatTensor>(mat2.pImpl,"mat2",5, false);
    THCSFloatTensor_spaddmm(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CUDAFloatType::addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    if(mat1.type().is_sparse()) {
        return static_cast<const Type*>(this)->addmm_(self, SparseTensor(mat1), mat2, beta, alpha);
    }
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toFloat();
    auto alpha_ = alpha.toFloat();
    auto mat1_ = checked_cast_tensor<CUDAFloatTensor>(mat1.pImpl,"mat1",5, false);
    auto mat2_ = checked_cast_tensor<CUDAFloatTensor>(mat2.pImpl,"mat2",6, false);
    THCudaTensor_addmm(context->thc_state, self_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    return self;
}
Tensor & CUDAFloatType::addmm_(Tensor & self, SparseTensor mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toFloat();
    auto alpha_ = alpha.toFloat();
    auto mat1_ = checked_cast_tensor<SparseCUDAFloatTensor>(mat1.tref.pImpl,"mat1",5,false);
    auto mat2_ = checked_cast_tensor<CUDAFloatTensor>(mat2.pImpl,"mat2",6, false);
    THCSFloatTensor_spaddmm(context->thc_state, self_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    return self;
}
Tensor & CUDAFloatType::s__addmv_out(Tensor & result, const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toFloat();
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toFloat();
    auto mat_ = checked_cast_tensor<CUDAFloatTensor>(mat.pImpl,"mat",4, false);
    auto vec_ = checked_cast_tensor<CUDAFloatTensor>(vec.pImpl,"vec",5, false);
    THCudaTensor_addmv(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, mat_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && mat_->isScalar() && vec_->isScalar());
    return result;
}
Tensor CUDAFloatType::s__addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toFloat();
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toFloat();
    auto mat_ = checked_cast_tensor<CUDAFloatTensor>(mat.pImpl,"mat",4, false);
    auto vec_ = checked_cast_tensor<CUDAFloatTensor>(vec.pImpl,"vec",5, false);
    THCudaTensor_addmv(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, mat_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && mat_->isScalar() && vec_->isScalar());
    return result;
}
Tensor & CUDAFloatType::_addmv_(Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toFloat();
    auto alpha_ = alpha.toFloat();
    auto mat_ = checked_cast_tensor<CUDAFloatTensor>(mat.pImpl,"mat",5, false);
    auto vec_ = checked_cast_tensor<CUDAFloatTensor>(vec.pImpl,"vec",6, false);
    THCudaTensor_addmv(context->thc_state, self_->tensor, beta_, self_->tensor, alpha_, mat_->tensor, vec_->tensor);
    return self;
}
Tensor & CUDAFloatType::s__addr_out(Tensor & result, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toFloat();
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toFloat();
    auto vec1_ = checked_cast_tensor<CUDAFloatTensor>(vec1.pImpl,"vec1",4, false);
    auto vec2_ = checked_cast_tensor<CUDAFloatTensor>(vec2.pImpl,"vec2",5, false);
    THCudaTensor_addr(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, vec1_->tensor, vec2_->tensor);
    result_->maybeScalar(self_->isScalar() && vec1_->isScalar() && vec2_->isScalar());
    return result;
}
Tensor CUDAFloatType::s__addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toFloat();
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toFloat();
    auto vec1_ = checked_cast_tensor<CUDAFloatTensor>(vec1.pImpl,"vec1",4, false);
    auto vec2_ = checked_cast_tensor<CUDAFloatTensor>(vec2.pImpl,"vec2",5, false);
    THCudaTensor_addr(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, vec1_->tensor, vec2_->tensor);
    result_->maybeScalar(self_->isScalar() && vec1_->isScalar() && vec2_->isScalar());
    return result;
}
Tensor & CUDAFloatType::_addr_(Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toFloat();
    auto alpha_ = alpha.toFloat();
    auto vec1_ = checked_cast_tensor<CUDAFloatTensor>(vec1.pImpl,"vec1",5, false);
    auto vec2_ = checked_cast_tensor<CUDAFloatTensor>(vec2.pImpl,"vec2",6, false);
    THCudaTensor_addr(context->thc_state, self_->tensor, beta_, self_->tensor, alpha_, vec1_->tensor, vec2_->tensor);
    return self;
}
Tensor & CUDAFloatType::_ger_out(Tensor & result, const Tensor & self, const Tensor & vec2) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.dim() == 0 ? 1 : self.size(0),vec2.dim() == 0 ? 1 : vec2.size(0) });
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto vec2_ = checked_cast_tensor<CUDAFloatTensor>(vec2.pImpl,"vec2",2, false);
    THCudaTensor_addr(context->thc_state, result_->tensor, float(0), result_->tensor, float(1), self_->tensor, vec2_->tensor);
    result_->maybeScalar(false);
    return result;
}
Tensor CUDAFloatType::_ger(const Tensor & self, const Tensor & vec2) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.dim() == 0 ? 1 : self.size(0),vec2.dim() == 0 ? 1 : vec2.size(0) });
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto vec2_ = checked_cast_tensor<CUDAFloatTensor>(vec2.pImpl,"vec2",2, false);
    THCudaTensor_addr(context->thc_state, result_->tensor, float(0), result_->tensor, float(1), self_->tensor, vec2_->tensor);
    result_->maybeScalar(false);
    return result;
}
Tensor & CUDAFloatType::_mv_out(Tensor & result, const Tensor & self, const Tensor & vec) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.size(0) });
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto vec_ = checked_cast_tensor<CUDAFloatTensor>(vec.pImpl,"vec",2, false);
    THCudaTensor_addmv(context->thc_state, result_->tensor, float(0), result_->tensor, float(1), self_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && vec_->isScalar());
    return result;
}
Tensor CUDAFloatType::_mv(const Tensor & self, const Tensor & vec) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.size(0) });
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto vec_ = checked_cast_tensor<CUDAFloatTensor>(vec.pImpl,"vec",2, false);
    THCudaTensor_addmv(context->thc_state, result_->tensor, float(0), result_->tensor, float(1), self_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && vec_->isScalar());
    return result;
}
Tensor & CUDAFloatType::_mm_out(Tensor & result, const Tensor & self, const Tensor & mat2) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.size(0),mat2.size(1) });
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CUDAFloatTensor>(mat2.pImpl,"mat2",2, false);
    THCudaTensor_addmm(context->thc_state, result_->tensor, float(0), result_->tensor, float(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CUDAFloatType::_mm(const Tensor & self, const Tensor & mat2) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.size(0),mat2.size(1) });
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CUDAFloatTensor>(mat2.pImpl,"mat2",2, false);
    THCudaTensor_addmm(context->thc_state, result_->tensor, float(0), result_->tensor, float(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CUDAFloatType::bmm_out(Tensor & result, const Tensor & self, const Tensor & mat2) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.size(0),self.size(1),mat2.size(2) });
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CUDAFloatTensor>(mat2.pImpl,"mat2",2, false);
    THCudaTensor_baddbmm(context->thc_state, result_->tensor, float(0), result_->tensor, float(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CUDAFloatType::bmm(const Tensor & self, const Tensor & mat2) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.size(0),self.size(1),mat2.size(2) });
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CUDAFloatTensor>(mat2.pImpl,"mat2",2, false);
    THCudaTensor_baddbmm(context->thc_state, result_->tensor, float(0), result_->tensor, float(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CUDAFloatType::s_addbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toFloat();
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toFloat();
    auto batch1_ = checked_cast_tensor<CUDAFloatTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CUDAFloatTensor>(batch2.pImpl,"batch2",5, false);
    THCudaTensor_addbmm(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor CUDAFloatType::s_addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toFloat();
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toFloat();
    auto batch1_ = checked_cast_tensor<CUDAFloatTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CUDAFloatTensor>(batch2.pImpl,"batch2",5, false);
    THCudaTensor_addbmm(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor & CUDAFloatType::addbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toFloat();
    auto alpha_ = alpha.toFloat();
    auto batch1_ = checked_cast_tensor<CUDAFloatTensor>(batch1.pImpl,"batch1",5, false);
    auto batch2_ = checked_cast_tensor<CUDAFloatTensor>(batch2.pImpl,"batch2",6, false);
    THCudaTensor_addbmm(context->thc_state, self_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    return self;
}
Tensor & CUDAFloatType::s_baddbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toFloat();
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toFloat();
    auto batch1_ = checked_cast_tensor<CUDAFloatTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CUDAFloatTensor>(batch2.pImpl,"batch2",5, false);
    THCudaTensor_baddbmm(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor CUDAFloatType::s_baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toFloat();
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toFloat();
    auto batch1_ = checked_cast_tensor<CUDAFloatTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CUDAFloatTensor>(batch2.pImpl,"batch2",5, false);
    THCudaTensor_baddbmm(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor & CUDAFloatType::baddbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toFloat();
    auto alpha_ = alpha.toFloat();
    auto batch1_ = checked_cast_tensor<CUDAFloatTensor>(batch1.pImpl,"batch1",5, false);
    auto batch2_ = checked_cast_tensor<CUDAFloatTensor>(batch2.pImpl,"batch2",6, false);
    THCudaTensor_baddbmm(context->thc_state, self_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    return self;
}
Tensor & CUDAFloatType::s_addcmul_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toFloat();
    auto tensor1_ = checked_cast_tensor<CUDAFloatTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CUDAFloatTensor>(tensor2.pImpl,"tensor2",4, false);
    THCudaTensor_addcmul(context->thc_state, result_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor CUDAFloatType::s_addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toFloat();
    auto tensor1_ = checked_cast_tensor<CUDAFloatTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CUDAFloatTensor>(tensor2.pImpl,"tensor2",4, false);
    THCudaTensor_addcmul(context->thc_state, result_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor & CUDAFloatType::s_addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toFloat();
    auto tensor1_ = checked_cast_tensor<CUDAFloatTensor>(tensor1.pImpl,"tensor1",4, false);
    auto tensor2_ = checked_cast_tensor<CUDAFloatTensor>(tensor2.pImpl,"tensor2",5, false);
    THCudaTensor_addcmul(context->thc_state, self_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    return self;
}
Tensor & CUDAFloatType::s_addcdiv_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toFloat();
    auto tensor1_ = checked_cast_tensor<CUDAFloatTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CUDAFloatTensor>(tensor2.pImpl,"tensor2",4, false);
    THCudaTensor_addcdiv(context->thc_state, result_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor CUDAFloatType::s_addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toFloat();
    auto tensor1_ = checked_cast_tensor<CUDAFloatTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CUDAFloatTensor>(tensor2.pImpl,"tensor2",4, false);
    THCudaTensor_addcdiv(context->thc_state, result_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor & CUDAFloatType::s_addcdiv_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toFloat();
    auto tensor1_ = checked_cast_tensor<CUDAFloatTensor>(tensor1.pImpl,"tensor1",4, false);
    auto tensor2_ = checked_cast_tensor<CUDAFloatTensor>(tensor2.pImpl,"tensor2",5, false);
    THCudaTensor_addcdiv(context->thc_state, self_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    return self;
}
std::tuple<Tensor &,Tensor &> CUDAFloatType::gesv_out(Tensor & solution, Tensor & lu, const Tensor & self, const Tensor & A) const {
    auto solution_ = checked_cast_tensor<CUDAFloatTensor>(solution.pImpl,"solution",0, false);
    auto lu_ = checked_cast_tensor<CUDAFloatTensor>(lu.pImpl,"lu",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto A_ = checked_cast_tensor<CUDAFloatTensor>(A.pImpl,"A",2, false);
    THCudaTensor_gesv(context->thc_state, solution_->tensor, lu_->tensor, self_->tensor, A_->tensor);
    bool maybe_scalar = self_->isScalar() && A_->isScalar();
    solution_->maybeScalar(maybe_scalar);
    lu_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(solution, lu);
}
std::tuple<Tensor,Tensor> CUDAFloatType::gesv(const Tensor & self, const Tensor & A) const {
    auto solution_ = new CUDAFloatTensor(context);
    auto solution = Tensor(solution_, false);
    auto lu_ = new CUDAFloatTensor(context);
    auto lu = Tensor(lu_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto A_ = checked_cast_tensor<CUDAFloatTensor>(A.pImpl,"A",2, false);
    THCudaTensor_gesv(context->thc_state, solution_->tensor, lu_->tensor, self_->tensor, A_->tensor);
    bool maybe_scalar = self_->isScalar() && A_->isScalar();
    solution_->maybeScalar(maybe_scalar);
    lu_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(solution, lu);
}
std::tuple<Tensor &,Tensor &> CUDAFloatType::gels_out(Tensor & res1, Tensor & res2, const Tensor & self, const Tensor & A) const {
    auto res1_ = checked_cast_tensor<CUDAFloatTensor>(res1.pImpl,"res1",0, false);
    auto res2_ = checked_cast_tensor<CUDAFloatTensor>(res2.pImpl,"res2",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto A_ = checked_cast_tensor<CUDAFloatTensor>(A.pImpl,"A",2, false);
    THCudaTensor_gels(context->thc_state, res1_->tensor, res2_->tensor, self_->tensor, A_->tensor);
    bool maybe_scalar = self_->isScalar() && A_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(res1, res2);
}
std::tuple<Tensor,Tensor> CUDAFloatType::gels(const Tensor & self, const Tensor & A) const {
    auto res1_ = new CUDAFloatTensor(context);
    auto res1 = Tensor(res1_, false);
    auto res2_ = new CUDAFloatTensor(context);
    auto res2 = Tensor(res2_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto A_ = checked_cast_tensor<CUDAFloatTensor>(A.pImpl,"A",2, false);
    THCudaTensor_gels(context->thc_state, res1_->tensor, res2_->tensor, self_->tensor, A_->tensor);
    bool maybe_scalar = self_->isScalar() && A_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(res1, res2);
}
std::tuple<Tensor &,Tensor &> CUDAFloatType::symeig_out(Tensor & res1, Tensor & res2, const Tensor & self, bool eigenvectors, bool upper) const {
    auto res1_ = checked_cast_tensor<CUDAFloatTensor>(res1.pImpl,"res1",0, false);
    auto res2_ = checked_cast_tensor<CUDAFloatTensor>(res2.pImpl,"res2",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_syev(context->thc_state, res1_->tensor, res2_->tensor, self_->tensor, (eigenvectors) ? "V" : "N", (upper) ? "U" : "L");
    bool maybe_scalar = self_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(res1, res2);
}
std::tuple<Tensor,Tensor> CUDAFloatType::symeig(const Tensor & self, bool eigenvectors, bool upper) const {
    auto res1_ = new CUDAFloatTensor(context);
    auto res1 = Tensor(res1_, false);
    auto res2_ = new CUDAFloatTensor(context);
    auto res2 = Tensor(res2_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_syev(context->thc_state, res1_->tensor, res2_->tensor, self_->tensor, (eigenvectors) ? "V" : "N", (upper) ? "U" : "L");
    bool maybe_scalar = self_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(res1, res2);
}
std::tuple<Tensor &,Tensor &> CUDAFloatType::eig_out(Tensor & res1, Tensor & res2, const Tensor & self, bool eigenvectors) const {
    auto res1_ = checked_cast_tensor<CUDAFloatTensor>(res1.pImpl,"res1",0, false);
    auto res2_ = checked_cast_tensor<CUDAFloatTensor>(res2.pImpl,"res2",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_geev(context->thc_state, res1_->tensor, res2_->tensor, self_->tensor, (eigenvectors) ? "V" : "N");
    bool maybe_scalar = self_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(res1, res2);
}
std::tuple<Tensor,Tensor> CUDAFloatType::eig(const Tensor & self, bool eigenvectors) const {
    auto res1_ = new CUDAFloatTensor(context);
    auto res1 = Tensor(res1_, false);
    auto res2_ = new CUDAFloatTensor(context);
    auto res2 = Tensor(res2_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_geev(context->thc_state, res1_->tensor, res2_->tensor, self_->tensor, (eigenvectors) ? "V" : "N");
    bool maybe_scalar = self_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(res1, res2);
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDAFloatType::svd_out(Tensor & res1, Tensor & res2, Tensor & res3, const Tensor & self, bool some) const {
    auto res1_ = checked_cast_tensor<CUDAFloatTensor>(res1.pImpl,"res1",0, false);
    auto res2_ = checked_cast_tensor<CUDAFloatTensor>(res2.pImpl,"res2",0, false);
    auto res3_ = checked_cast_tensor<CUDAFloatTensor>(res3.pImpl,"res3",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_gesvd(context->thc_state, res1_->tensor, res2_->tensor, res3_->tensor, self_->tensor, (some) ? "S" : "A");
    bool maybe_scalar = self_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    res3_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(res1, res2, res3);
}
std::tuple<Tensor,Tensor,Tensor> CUDAFloatType::svd(const Tensor & self, bool some) const {
    auto res1_ = new CUDAFloatTensor(context);
    auto res1 = Tensor(res1_, false);
    auto res2_ = new CUDAFloatTensor(context);
    auto res2 = Tensor(res2_, false);
    auto res3_ = new CUDAFloatTensor(context);
    auto res3 = Tensor(res3_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_gesvd(context->thc_state, res1_->tensor, res2_->tensor, res3_->tensor, self_->tensor, (some) ? "S" : "A");
    bool maybe_scalar = self_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    res3_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(res1, res2, res3);
}
Tensor & CUDAFloatType::inverse_out(Tensor & output, const Tensor & self) const {
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_getri(context->thc_state, output_->tensor, self_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAFloatType::inverse(const Tensor & self) const {
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_getri(context->thc_state, output_->tensor, self_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAFloatType::potrf_out(Tensor & output, const Tensor & self, bool upper) const {
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_potrf(context->thc_state, output_->tensor, self_->tensor, (upper) ? "U" : "L");
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAFloatType::potrf(const Tensor & self, bool upper) const {
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_potrf(context->thc_state, output_->tensor, self_->tensor, (upper) ? "U" : "L");
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAFloatType::potrs_out(Tensor & result, const Tensor & self, const Tensor & input2, bool upper) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto input2_ = checked_cast_tensor<CUDAFloatTensor>(input2.pImpl,"input2",2, false);
    THCudaTensor_potrs(context->thc_state, result_->tensor, self_->tensor, input2_->tensor, (upper) ? "U" : "L");
    result_->maybeScalar(self_->isScalar() && input2_->isScalar());
    return result;
}
Tensor CUDAFloatType::potrs(const Tensor & self, const Tensor & input2, bool upper) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto input2_ = checked_cast_tensor<CUDAFloatTensor>(input2.pImpl,"input2",2, false);
    THCudaTensor_potrs(context->thc_state, result_->tensor, self_->tensor, input2_->tensor, (upper) ? "U" : "L");
    result_->maybeScalar(self_->isScalar() && input2_->isScalar());
    return result;
}
Tensor & CUDAFloatType::potri_out(Tensor & output, const Tensor & self, bool upper) const {
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_potri(context->thc_state, output_->tensor, self_->tensor, (upper) ? "U" : "L");
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAFloatType::potri(const Tensor & self, bool upper) const {
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_potri(context->thc_state, output_->tensor, self_->tensor, (upper) ? "U" : "L");
    output_->maybeScalar(self_->isScalar());
    return output;
}
std::tuple<Tensor &,Tensor &> CUDAFloatType::qr_out(Tensor & res1, Tensor & res2, const Tensor & self) const {
    auto res1_ = checked_cast_tensor<CUDAFloatTensor>(res1.pImpl,"res1",0, false);
    auto res2_ = checked_cast_tensor<CUDAFloatTensor>(res2.pImpl,"res2",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_qr(context->thc_state, res1_->tensor, res2_->tensor, self_->tensor);
    bool maybe_scalar = self_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(res1, res2);
}
std::tuple<Tensor,Tensor> CUDAFloatType::qr(const Tensor & self) const {
    auto res1_ = new CUDAFloatTensor(context);
    auto res1 = Tensor(res1_, false);
    auto res2_ = new CUDAFloatTensor(context);
    auto res2 = Tensor(res2_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_qr(context->thc_state, res1_->tensor, res2_->tensor, self_->tensor);
    bool maybe_scalar = self_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(res1, res2);
}
std::tuple<Tensor &,Tensor &> CUDAFloatType::geqrf_out(Tensor & res1, Tensor & res2, const Tensor & self) const {
    auto res1_ = checked_cast_tensor<CUDAFloatTensor>(res1.pImpl,"res1",0, false);
    auto res2_ = checked_cast_tensor<CUDAFloatTensor>(res2.pImpl,"res2",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_geqrf(context->thc_state, res1_->tensor, res2_->tensor, self_->tensor);
    bool maybe_scalar = self_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(res1, res2);
}
std::tuple<Tensor,Tensor> CUDAFloatType::geqrf(const Tensor & self) const {
    auto res1_ = new CUDAFloatTensor(context);
    auto res1 = Tensor(res1_, false);
    auto res2_ = new CUDAFloatTensor(context);
    auto res2 = Tensor(res2_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    THCudaTensor_geqrf(context->thc_state, res1_->tensor, res2_->tensor, self_->tensor);
    bool maybe_scalar = self_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(res1, res2);
}
std::tuple<Tensor &,Tensor &> CUDAFloatType::btrifact_out(Tensor & result, Tensor & pivots, const Tensor & self, bool pivot) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto pivots_ = checked_cast_tensor<CUDAIntTensor>(pivots.pImpl,"pivots",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    THCudaTensor_btrifact(context->thc_state, result_->tensor, pivots_->tensor, NULL, pivot, self_->tensor);
    bool maybe_scalar = self_->isScalar();
    result_->maybeScalar(maybe_scalar);
    pivots_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(result, pivots);
}
std::tuple<Tensor,Tensor> CUDAFloatType::btrifact(const Tensor & self, bool pivot) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto pivots_ = new CUDAIntTensor(context);
    auto pivots = Tensor(pivots_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    THCudaTensor_btrifact(context->thc_state, result_->tensor, pivots_->tensor, NULL, pivot, self_->tensor);
    bool maybe_scalar = self_->isScalar();
    result_->maybeScalar(maybe_scalar);
    pivots_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(result, pivots);
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDAFloatType::btrifact_with_info_out(Tensor & result, Tensor & pivots, Tensor & info, const Tensor & self, bool pivot) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto pivots_ = checked_cast_tensor<CUDAIntTensor>(pivots.pImpl,"pivots",0, false);
    auto info_ = checked_cast_tensor<CUDAIntTensor>(info.pImpl,"info",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    THCudaTensor_btrifact(context->thc_state, result_->tensor, pivots_->tensor, info_->tensor, pivot, self_->tensor);
    bool maybe_scalar = self_->isScalar();
    result_->maybeScalar(maybe_scalar);
    pivots_->maybeScalar(maybe_scalar);
    info_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(result, pivots, info);
}
std::tuple<Tensor,Tensor,Tensor> CUDAFloatType::btrifact_with_info(const Tensor & self, bool pivot) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto pivots_ = new CUDAIntTensor(context);
    auto pivots = Tensor(pivots_, false);
    auto info_ = new CUDAIntTensor(context);
    auto info = Tensor(info_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    THCudaTensor_btrifact(context->thc_state, result_->tensor, pivots_->tensor, info_->tensor, pivot, self_->tensor);
    bool maybe_scalar = self_->isScalar();
    result_->maybeScalar(maybe_scalar);
    pivots_->maybeScalar(maybe_scalar);
    info_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(result, pivots, info);
}
Tensor & CUDAFloatType::btrisolve_out(Tensor & result, const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto LU_data_ = checked_cast_tensor<CUDAFloatTensor>(LU_data.pImpl,"LU_data",2, false);
    auto LU_pivots_ = checked_cast_tensor<CUDAIntTensor>(LU_pivots.pImpl,"LU_pivots",3, false);
    THCudaTensor_btrisolve(context->thc_state, result_->tensor, self_->tensor, LU_data_->tensor, LU_pivots_->tensor);
    result_->maybeScalar(self_->isScalar() && LU_data_->isScalar() && LU_pivots_->isScalar());
    return result;
}
Tensor CUDAFloatType::btrisolve(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto LU_data_ = checked_cast_tensor<CUDAFloatTensor>(LU_data.pImpl,"LU_data",2, false);
    auto LU_pivots_ = checked_cast_tensor<CUDAIntTensor>(LU_pivots.pImpl,"LU_pivots",3, false);
    THCudaTensor_btrisolve(context->thc_state, result_->tensor, self_->tensor, LU_data_->tensor, LU_pivots_->tensor);
    result_->maybeScalar(self_->isScalar() && LU_data_->isScalar() && LU_pivots_->isScalar());
    return result;
}
Tensor & CUDAFloatType::random_(Tensor & self, int64_t from, int64_t to, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    THCudaTensor_clampedRandom(context->thc_state, self_->tensor, from, to);
    return self;
}
Tensor & CUDAFloatType::random_(Tensor & self, int64_t to, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    THCudaTensor_cappedRandom(context->thc_state, self_->tensor, to);
    return self;
}
Tensor & CUDAFloatType::random_(Tensor & self, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    THCudaTensor_random(context->thc_state, self_->tensor);
    return self;
}
Tensor & CUDAFloatType::multinomial_out(Tensor & result, const Tensor & self, int64_t num_samples, bool replacement, Generator * generator) const {
    auto result_ = checked_cast_tensor<CUDALongTensor>(result.pImpl,"result",0, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    THCudaTensor_multinomial(context->thc_state, result_->tensor, self_->tensor, num_samples, replacement);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAFloatType::multinomial(const Tensor & self, int64_t num_samples, bool replacement, Generator * generator) const {
    auto result_ = new CUDALongTensor(context);
    auto result = Tensor(result_, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    THCudaTensor_multinomial(context->thc_state, result_->tensor, self_->tensor, num_samples, replacement);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::uniform_(Tensor & self, double from, double to, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    THCudaTensor_uniform(context->thc_state, self_->tensor, from, to);
    return self;
}
Tensor & CUDAFloatType::normal_out(Tensor & output, const Tensor & mean, double std, Generator * generator) const {
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",0, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    auto mean_ = checked_cast_tensor<CUDAFloatTensor>(mean.pImpl,"mean",2, false);
    THCudaTensor_normal_means(context->thc_state, output_->tensor, mean_->tensor, std);
    output_->maybeScalar(mean_->isScalar());
    return output;
}
Tensor CUDAFloatType::normal(const Tensor & mean, double std, Generator * generator) const {
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    auto mean_ = checked_cast_tensor<CUDAFloatTensor>(mean.pImpl,"mean",2, false);
    THCudaTensor_normal_means(context->thc_state, output_->tensor, mean_->tensor, std);
    output_->maybeScalar(mean_->isScalar());
    return output;
}
Tensor & CUDAFloatType::normal_out(Tensor & output, double mean, const Tensor & std, Generator * generator) const {
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",0, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    auto std_ = checked_cast_tensor<CUDAFloatTensor>(std.pImpl,"std",3, false);
    THCudaTensor_normal_stddevs(context->thc_state, output_->tensor, mean, std_->tensor);
    output_->maybeScalar(std_->isScalar());
    return output;
}
Tensor CUDAFloatType::normal(double mean, const Tensor & std, Generator * generator) const {
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    auto std_ = checked_cast_tensor<CUDAFloatTensor>(std.pImpl,"std",3, false);
    THCudaTensor_normal_stddevs(context->thc_state, output_->tensor, mean, std_->tensor);
    output_->maybeScalar(std_->isScalar());
    return output;
}
Tensor & CUDAFloatType::normal_out(Tensor & output, const Tensor & mean, const Tensor & std, Generator * generator) const {
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",0, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    auto mean_ = checked_cast_tensor<CUDAFloatTensor>(mean.pImpl,"mean",2, false);
    auto std_ = checked_cast_tensor<CUDAFloatTensor>(std.pImpl,"std",3, false);
    THCudaTensor_normal_means_stddevs(context->thc_state, output_->tensor, mean_->tensor, std_->tensor);
    output_->maybeScalar(mean_->isScalar() && std_->isScalar());
    return output;
}
Tensor CUDAFloatType::normal(const Tensor & mean, const Tensor & std, Generator * generator) const {
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    auto mean_ = checked_cast_tensor<CUDAFloatTensor>(mean.pImpl,"mean",2, false);
    auto std_ = checked_cast_tensor<CUDAFloatTensor>(std.pImpl,"std",3, false);
    THCudaTensor_normal_means_stddevs(context->thc_state, output_->tensor, mean_->tensor, std_->tensor);
    output_->maybeScalar(mean_->isScalar() && std_->isScalar());
    return output;
}
Tensor & CUDAFloatType::normal_(Tensor & self, double mean, double std, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    THCudaTensor_normal(context->thc_state, self_->tensor, mean, std);
    return self;
}
Tensor & CUDAFloatType::cauchy_(Tensor & self, double median, double sigma, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    THCudaTensor_cauchy(context->thc_state, self_->tensor, median, sigma);
    return self;
}
Tensor & CUDAFloatType::log_normal_(Tensor & self, double mean, double std, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    THCudaTensor_logNormal(context->thc_state, self_->tensor, mean, std);
    return self;
}
Tensor & CUDAFloatType::exponential_(Tensor & self, double lambd, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    THCudaTensor_exponential(context->thc_state, self_->tensor, lambd);
    return self;
}
Tensor & CUDAFloatType::geometric_(Tensor & self, double p, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    THCudaTensor_geometric(context->thc_state, self_->tensor, p);
    return self;
}
Tensor & CUDAFloatType::bernoulli_out(Tensor & output, const Tensor & self, Generator * generator) const {
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",0, false);
    output.resize_(self.sizes());
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    THCudaTensor_bernoulli_Tensor(context->thc_state, output_->tensor, self_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAFloatType::bernoulli(const Tensor & self, Generator * generator) const {
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    output.resize_(self.sizes());
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    THCudaTensor_bernoulli_Tensor(context->thc_state, output_->tensor, self_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAFloatType::tensor(Storage & storage, int64_t storageOffset, IntList size, IntList stride) const {
    auto storage_ = checked_cast_storage<CUDAFloatStorage>(&storage,"storage",1);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    return Tensor((new CUDAFloatTensor(context, THCudaTensor_newWithStorage(context->thc_state, storage_->storage, storageOffset, size_, stride_)))->maybeScalar(size.size() == 0),false);
}
Tensor CUDAFloatType::tensor(IntList size) const {
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new CUDAFloatTensor(context, THCudaTensor_newWithSize(context->thc_state, size_, NULL)))->maybeScalar(size.size() == 0),false);
}
Tensor CUDAFloatType::tensor(IntList size, IntList stride) const {
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    return Tensor((new CUDAFloatTensor(context, THCudaTensor_newWithSize(context->thc_state, size_, stride_)))->maybeScalar(size.size() == 0),false);
}
Tensor CUDAFloatType::tensor() const {
    return Tensor((new CUDAFloatTensor(context, THCudaTensor_new(context->thc_state))),false);
}
Tensor CUDAFloatType::alias(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    return Tensor((new CUDAFloatTensor(context, THCudaTensor_newWithTensor(context->thc_state, self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor & CUDAFloatType::_copy_ignoring_overlaps_(Tensor & self, const Tensor & src) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto src_ = checked_cast_tensor<CUDAFloatTensor>(src.pImpl,"src",2, false);
    THCudaTensor_copyIgnoringOverlaps(context->thc_state, self_->tensor, src_->tensor);
    return self;
}
Tensor & CUDAFloatType::as_strided_out(Tensor & result, const Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    auto result_ = checked_cast_tensor<CUDAFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    if (storage_offset == -1) {
      storage_offset = self_->tensor->storageOffset;
    }
    THCudaTensor_setStorage(context->thc_state, result_->tensor, self_->tensor->storage, storage_offset, size_, stride_);
    result_->maybeScalar(size.size() == 0);
    return result;
}
Tensor CUDAFloatType::as_strided(const Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    auto result_ = new CUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    if (storage_offset == -1) {
      storage_offset = self_->tensor->storageOffset;
    }
    THCudaTensor_setStorage(context->thc_state, result_->tensor, self_->tensor->storage, storage_offset, size_, stride_);
    result_->maybeScalar(size.size() == 0);
    return result;
}
Tensor & CUDAFloatType::as_strided_(Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    if (storage_offset == -1) {
      storage_offset = self_->tensor->storageOffset;
    }
    THCudaTensor_setStorage(context->thc_state, self_->tensor, self_->tensor->storage, storage_offset, size_, stride_);
    self_->maybeScalar(size.size() == 0);
    return self;
}
Tensor & CUDAFloatType::_cat_out(Tensor & self, TensorList tensors, int64_t dim) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",0, false);
    auto tensors_ = tensor_list_checked_cast<CUDAFloatTensor, Tensor, THCudaTensor>(tensors,"tensors",1);
    THCudaTensor_catArray(context->thc_state, self_->tensor, tensors_.data(), tensors_.size(), dim);
    return self;
}
Tensor CUDAFloatType::_cat(TensorList tensors, int64_t dim) const {
    auto self_ = new CUDAFloatTensor(context);
    auto self = Tensor(self_, false);
    auto tensors_ = tensor_list_checked_cast<CUDAFloatTensor, Tensor, THCudaTensor>(tensors,"tensors",1);
    THCudaTensor_catArray(context->thc_state, self_->tensor, tensors_.data(), tensors_.size(), dim);
    return self;
}
Tensor CUDAFloatType::_sparse_mask(const Tensor & self, SparseTensor mask) const {
    auto result_ = new SparseCUDAFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<SparseCUDAFloatTensor>(mask.tref.pImpl,"mask",2,false);
    THCudaTensor_sparseMask(context->thc_state, result_->tensor, self_->tensor, mask_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAFloatType::binary_cross_entropy_forward_out(Tensor & output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDAFloatTensor>(target.pImpl,"target",2, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",3, true);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",5, false);
    THNN_CudaBCECriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, weight_ ? weight_->tensor : NULL, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor CUDAFloatType::binary_cross_entropy_forward(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDAFloatTensor>(target.pImpl,"target",2, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",3, true);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaBCECriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, weight_ ? weight_->tensor : NULL, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor & CUDAFloatType::binary_cross_entropy_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDAFloatTensor>(target.pImpl,"target",3, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",4, true);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",6, false);
    THNN_CudaBCECriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, weight_ ? weight_->tensor : NULL, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::binary_cross_entropy_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDAFloatTensor>(target.pImpl,"target",3, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",4, true);
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaBCECriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, weight_ ? weight_->tensor : NULL, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAFloatType::kl_div_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDAFloatTensor>(target.pImpl,"target",2, false);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",4, false);
    THNN_CudaDistKLDivCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor CUDAFloatType::kl_div_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDAFloatTensor>(target.pImpl,"target",2, false);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaDistKLDivCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor & CUDAFloatType::kl_div_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDAFloatTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",5, false);
    THNN_CudaDistKLDivCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::kl_div_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDAFloatTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDistKLDivCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAFloatType::l1_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDAFloatTensor>(target.pImpl,"target",2, false);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",4, false);
    THNN_CudaAbsCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor CUDAFloatType::l1_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDAFloatTensor>(target.pImpl,"target",2, false);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaAbsCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor & CUDAFloatType::l1_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDAFloatTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",5, false);
    THNN_CudaAbsCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDAFloatTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaAbsCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAFloatType::mse_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDAFloatTensor>(target.pImpl,"target",2, false);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",4, false);
    THNN_CudaMSECriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor CUDAFloatType::mse_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDAFloatTensor>(target.pImpl,"target",2, false);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaMSECriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor & CUDAFloatType::mse_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDAFloatTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",5, false);
    THNN_CudaMSECriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::mse_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDAFloatTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaMSECriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAFloatType::multi_margin_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",2, false);
    auto p_ = p.toDouble();
    auto margin_ = margin.toDouble();
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",5, true);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",7, false);
    THNN_CudaMultiMarginCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, p_, weight_ ? weight_->tensor : NULL, margin_, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor CUDAFloatType::multi_margin_loss_forward(const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",2, false);
    auto p_ = p.toDouble();
    auto margin_ = margin.toDouble();
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",5, true);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaMultiMarginCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, p_, weight_ ? weight_->tensor : NULL, margin_, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor & CUDAFloatType::multi_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",3, false);
    auto p_ = p.toDouble();
    auto margin_ = margin.toDouble();
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",6, true);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",8, false);
    THNN_CudaMultiMarginCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, p_, weight_ ? weight_->tensor : NULL, margin_, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::multi_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",3, false);
    auto p_ = p.toDouble();
    auto margin_ = margin.toDouble();
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",6, true);
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaMultiMarginCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, p_, weight_ ? weight_->tensor : NULL, margin_, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &> CUDAFloatType::multilabel_margin_loss_forward_out(Tensor & output, Tensor & is_target, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",2, false);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",4, false);
    auto is_target_ = checked_cast_tensor<CUDAFloatTensor>(is_target.pImpl,"is_target",4, false);
    THNN_CudaMultiLabelMarginCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, is_target_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    is_target_->maybeScalar(target_->isScalar());
    return std::tuple<Tensor &, Tensor &>(output, is_target);
}
std::tuple<Tensor,Tensor> CUDAFloatType::multilabel_margin_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",2, false);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    auto is_target_ = new CUDAFloatTensor(context);
    auto is_target = Tensor(is_target_, false);
    THNN_CudaMultiLabelMarginCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, is_target_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    is_target_->maybeScalar(target_->isScalar());
    return std::tuple<Tensor, Tensor>(output, is_target);
}
Tensor & CUDAFloatType::multilabel_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce, const Tensor & is_target) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",3, false);
    auto is_target_ = checked_cast_tensor<CUDAFloatTensor>(is_target.pImpl,"is_target",6, false);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",6, false);
    THNN_CudaMultiLabelMarginCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, is_target_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::multilabel_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce, const Tensor & is_target) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",3, false);
    auto is_target_ = checked_cast_tensor<CUDAFloatTensor>(is_target.pImpl,"is_target",6, false);
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaMultiLabelMarginCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, is_target_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &> CUDAFloatType::nll_loss_forward_out(Tensor & output, Tensor & total_weight, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",2, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",3, true);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",6, false);
    auto total_weight_ = checked_cast_tensor<CUDAFloatTensor>(total_weight.pImpl,"total_weight",6, false);
    THNN_CudaClassNLLCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, weight_ ? weight_->tensor : NULL, total_weight_->tensor, ignore_index, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    total_weight_->maybeScalar(true);
    return std::tuple<Tensor &, Tensor &>(output, total_weight);
}
std::tuple<Tensor,Tensor> CUDAFloatType::nll_loss_forward(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",2, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",3, true);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    auto total_weight_ = new CUDAFloatTensor(context);
    auto total_weight = Tensor(total_weight_, false);
    THNN_CudaClassNLLCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, weight_ ? weight_->tensor : NULL, total_weight_->tensor, ignore_index, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    total_weight_->maybeScalar(true);
    return std::tuple<Tensor, Tensor>(output, total_weight);
}
Tensor & CUDAFloatType::nll_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",3, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",4, true);
    auto total_weight_ = checked_cast_tensor<CUDAFloatTensor>(total_weight.pImpl,"total_weight",8, false);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",8, false);
    THNN_CudaClassNLLCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, weight_ ? weight_->tensor : NULL, total_weight_->tensor, ignore_index, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::nll_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",3, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",4, true);
    auto total_weight_ = checked_cast_tensor<CUDAFloatTensor>(total_weight.pImpl,"total_weight",8, false);
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaClassNLLCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, weight_ ? weight_->tensor : NULL, total_weight_->tensor, ignore_index, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &> CUDAFloatType::nll_loss2d_forward_out(Tensor & output, Tensor & total_weight, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",2, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",3, true);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",6, false);
    auto total_weight_ = checked_cast_tensor<CUDAFloatTensor>(total_weight.pImpl,"total_weight",6, false);
    THNN_CudaSpatialClassNLLCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, weight_ ? weight_->tensor : NULL, total_weight_->tensor, ignore_index, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    total_weight_->maybeScalar(true);
    return std::tuple<Tensor &, Tensor &>(output, total_weight);
}
std::tuple<Tensor,Tensor> CUDAFloatType::nll_loss2d_forward(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",2, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",3, true);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    auto total_weight_ = new CUDAFloatTensor(context);
    auto total_weight = Tensor(total_weight_, false);
    THNN_CudaSpatialClassNLLCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, weight_ ? weight_->tensor : NULL, total_weight_->tensor, ignore_index, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    total_weight_->maybeScalar(true);
    return std::tuple<Tensor, Tensor>(output, total_weight);
}
Tensor & CUDAFloatType::nll_loss2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",3, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",4, true);
    auto total_weight_ = checked_cast_tensor<CUDAFloatTensor>(total_weight.pImpl,"total_weight",8, false);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",8, false);
    THNN_CudaSpatialClassNLLCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, weight_ ? weight_->tensor : NULL, total_weight_->tensor, ignore_index, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::nll_loss2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",3, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",4, true);
    auto total_weight_ = checked_cast_tensor<CUDAFloatTensor>(total_weight.pImpl,"total_weight",8, false);
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaSpatialClassNLLCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, weight_ ? weight_->tensor : NULL, total_weight_->tensor, ignore_index, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAFloatType::smooth_l1_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDAFloatTensor>(target.pImpl,"target",2, false);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",4, false);
    THNN_CudaSmoothL1Criterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor CUDAFloatType::smooth_l1_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDAFloatTensor>(target.pImpl,"target",2, false);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaSmoothL1Criterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor & CUDAFloatType::smooth_l1_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDAFloatTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",5, false);
    THNN_CudaSmoothL1Criterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::smooth_l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDAFloatTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaSmoothL1Criterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAFloatType::soft_margin_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDAFloatTensor>(target.pImpl,"target",2, false);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",4, false);
    THNN_CudaSoftMarginCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor CUDAFloatType::soft_margin_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDAFloatTensor>(target.pImpl,"target",2, false);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaSoftMarginCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor & CUDAFloatType::soft_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDAFloatTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",5, false);
    THNN_CudaSoftMarginCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::soft_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDAFloatTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaSoftMarginCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAFloatType::elu_forward_out(Tensor & output, const Tensor & self, Scalar alpha, Scalar scale) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toDouble();
    auto scale_ = scale.toDouble();
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",3, false);
    THNN_CudaELU_updateOutput(context->thc_state, self_->tensor, output_->tensor, alpha_, scale_, false);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAFloatType::elu_forward(const Tensor & self, Scalar alpha, Scalar scale) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toDouble();
    auto scale_ = scale.toDouble();
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaELU_updateOutput(context->thc_state, self_->tensor, output_->tensor, alpha_, scale_, false);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAFloatType::elu_backward_out(Tensor & grad_input, const Tensor & grad_output, Scalar alpha, Scalar scale, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto alpha_ = alpha.toDouble();
    auto scale_ = scale.toDouble();
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",4, false);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_CudaELU_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, output_->tensor, alpha_, scale_);
    grad_input_->maybeScalar(output_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::elu_backward(const Tensor & grad_output, Scalar alpha, Scalar scale, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto alpha_ = alpha.toDouble();
    auto scale_ = scale.toDouble();
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",4, false);
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaELU_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, output_->tensor, alpha_, scale_);
    grad_input_->maybeScalar(output_->isScalar());
    return grad_input;
}
Tensor & CUDAFloatType::elu_forward_(Tensor & self, Scalar alpha, Scalar scale) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toDouble();
    auto scale_ = scale.toDouble();
    THNN_CudaELU_updateOutput(context->thc_state, self_->tensor, self_->tensor, alpha_, scale_, true);
    return self;
}
Tensor & CUDAFloatType::glu_forward_out(Tensor & output, const Tensor & self, int64_t dim) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",2, false);
    THNN_CudaGatedLinear_updateOutput(context->thc_state, self_->tensor, output_->tensor, dim);
    output_->maybeScalar(false);
    return output;
}
Tensor CUDAFloatType::glu_forward(const Tensor & self, int64_t dim) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaGatedLinear_updateOutput(context->thc_state, self_->tensor, output_->tensor, dim);
    output_->maybeScalar(false);
    return output;
}
Tensor & CUDAFloatType::glu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    dim = maybe_wrap_dim(dim, self_);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaGatedLinear_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, dim);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::glu_backward(const Tensor & grad_output, const Tensor & self, int64_t dim) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    dim = maybe_wrap_dim(dim, self_);
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaGatedLinear_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, dim);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAFloatType::hardtanh_forward_out(Tensor & output, const Tensor & self, Scalar min_val, Scalar max_val) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto min_val_ = min_val.toDouble();
    auto max_val_ = max_val.toDouble();
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",3, false);
    THNN_CudaHardTanh_updateOutput(context->thc_state, self_->tensor, output_->tensor, min_val_, max_val_, false);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAFloatType::hardtanh_forward(const Tensor & self, Scalar min_val, Scalar max_val) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto min_val_ = min_val.toDouble();
    auto max_val_ = max_val.toDouble();
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaHardTanh_updateOutput(context->thc_state, self_->tensor, output_->tensor, min_val_, max_val_, false);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAFloatType::hardtanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto min_val_ = min_val.toDouble();
    auto max_val_ = max_val.toDouble();
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_CudaHardTanh_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, min_val_, max_val_, false);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::hardtanh_backward(const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto min_val_ = min_val.toDouble();
    auto max_val_ = max_val.toDouble();
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHardTanh_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, min_val_, max_val_, false);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAFloatType::hardtanh_forward_(Tensor & self, Scalar min_val, Scalar max_val) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto min_val_ = min_val.toDouble();
    auto max_val_ = max_val.toDouble();
    THNN_CudaHardTanh_updateOutput(context->thc_state, self_->tensor, self_->tensor, min_val_, max_val_, true);
    return self;
}
Tensor & CUDAFloatType::leaky_relu_forward_out(Tensor & output, const Tensor & self, Scalar negative_slope) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto negative_slope_ = negative_slope.toDouble();
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",2, false);
    THNN_CudaLeakyReLU_updateOutput(context->thc_state, self_->tensor, output_->tensor, negative_slope_, false);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAFloatType::leaky_relu_forward(const Tensor & self, Scalar negative_slope) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto negative_slope_ = negative_slope.toDouble();
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaLeakyReLU_updateOutput(context->thc_state, self_->tensor, output_->tensor, negative_slope_, false);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAFloatType::leaky_relu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar negative_slope) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto negative_slope_ = negative_slope.toDouble();
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaLeakyReLU_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, negative_slope_, false);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::leaky_relu_backward(const Tensor & grad_output, const Tensor & self, Scalar negative_slope) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto negative_slope_ = negative_slope.toDouble();
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaLeakyReLU_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, negative_slope_, false);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAFloatType::leaky_relu_forward_(Tensor & self, Scalar negative_slope) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto negative_slope_ = negative_slope.toDouble();
    THNN_CudaLeakyReLU_updateOutput(context->thc_state, self_->tensor, self_->tensor, negative_slope_, true);
    return self;
}
std::tuple<Tensor &,Tensor &> CUDAFloatType::log_sigmoid_forward_out(Tensor & output, Tensor & buffer, const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",1, false);
    auto buffer_ = checked_cast_tensor<CUDAFloatTensor>(buffer.pImpl,"buffer",1, false);
    THNN_CudaLogSigmoid_updateOutput(context->thc_state, self_->tensor, output_->tensor, buffer_->tensor);
    output_->maybeScalar(self_->isScalar());
    buffer_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &>(output, buffer);
}
std::tuple<Tensor,Tensor> CUDAFloatType::log_sigmoid_forward(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    auto buffer_ = new CUDAFloatTensor(context);
    auto buffer = Tensor(buffer_, false);
    THNN_CudaLogSigmoid_updateOutput(context->thc_state, self_->tensor, output_->tensor, buffer_->tensor);
    output_->maybeScalar(self_->isScalar());
    buffer_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor>(output, buffer);
}
Tensor & CUDAFloatType::log_sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & buffer) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto buffer_ = checked_cast_tensor<CUDAFloatTensor>(buffer.pImpl,"buffer",3, false);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaLogSigmoid_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, buffer_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::log_sigmoid_backward(const Tensor & grad_output, const Tensor & self, const Tensor & buffer) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto buffer_ = checked_cast_tensor<CUDAFloatTensor>(buffer.pImpl,"buffer",3, false);
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaLogSigmoid_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, buffer_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAFloatType::log_softmax_forward_out(Tensor & output, const Tensor & self, int64_t dim) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",2, false);
    THNN_CudaLogSoftMax_updateOutput(context->thc_state, self_->tensor, output_->tensor, dim);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAFloatType::log_softmax_forward(const Tensor & self, int64_t dim) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaLogSoftMax_updateOutput(context->thc_state, self_->tensor, output_->tensor, dim);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAFloatType::log_softmax_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",4, false);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_CudaLogSoftMax_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, output_->tensor, dim);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::log_softmax_backward(const Tensor & grad_output, const Tensor & self, int64_t dim, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",4, false);
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaLogSoftMax_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, output_->tensor, dim);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAFloatType::prelu_forward_out(Tensor & output, const Tensor & self, const Tensor & weight) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",2, false);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",2, false);
    THNN_CudaPReLU_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAFloatType::prelu_forward(const Tensor & self, const Tensor & weight) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",2, false);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaPReLU_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
std::tuple<Tensor &,Tensor &> CUDAFloatType::prelu_backward_out(Tensor & grad_input, Tensor & grad_weight, const Tensor & grad_output, const Tensor & self, const Tensor & weight) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",3, false);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",3, true);
    auto grad_weight_ = checked_cast_tensor<CUDAFloatTensor>(grad_weight.pImpl,"grad_weight",3, true);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    if (grad_input_) THNN_CudaPReLU_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor);
    if (grad_weight_) THNN_CudaPReLU_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &>(grad_input, grad_weight);
}
std::tuple<Tensor,Tensor> CUDAFloatType::prelu_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, std::array<bool,2> output_mask) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",3, false);
    auto grad_input_ = output_mask[0] ? new CUDAFloatTensor(context) : nullptr;
    auto grad_input = Tensor(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_input_, false);
    auto grad_weight_ = output_mask[1] ? new CUDAFloatTensor(context) : nullptr;
    auto grad_weight = Tensor(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_weight_, false);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    if (grad_input_) THNN_CudaPReLU_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor);
    if (grad_weight_) THNN_CudaPReLU_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor>(grad_input, grad_weight);
}
Tensor & CUDAFloatType::rrelu_with_noise_forward_out(Tensor & output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto noise_ = checked_cast_tensor<CUDAFloatTensor>(noise.pImpl,"noise",2, false);
    auto lower_ = lower.toDouble();
    auto upper_ = upper.toDouble();
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",6, false);
    THNN_CudaRReLU_updateOutput(context->thc_state, self_->tensor, output_->tensor, noise_->tensor, lower_, upper_, training, false, NULL);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAFloatType::rrelu_with_noise_forward(const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto noise_ = checked_cast_tensor<CUDAFloatTensor>(noise.pImpl,"noise",2, false);
    auto lower_ = lower.toDouble();
    auto upper_ = upper.toDouble();
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaRReLU_updateOutput(context->thc_state, self_->tensor, output_->tensor, noise_->tensor, lower_, upper_, training, false, NULL);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAFloatType::rrelu_with_noise_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto noise_ = checked_cast_tensor<CUDAFloatTensor>(noise.pImpl,"noise",3, false);
    auto lower_ = lower.toDouble();
    auto upper_ = upper.toDouble();
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",6, false);
    THNN_CudaRReLU_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, noise_->tensor, lower_, upper_, training, false);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::rrelu_with_noise_backward(const Tensor & grad_output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto noise_ = checked_cast_tensor<CUDAFloatTensor>(noise.pImpl,"noise",3, false);
    auto lower_ = lower.toDouble();
    auto upper_ = upper.toDouble();
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaRReLU_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, noise_->tensor, lower_, upper_, training, false);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAFloatType::rrelu_with_noise_forward_(Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto noise_ = checked_cast_tensor<CUDAFloatTensor>(noise.pImpl,"noise",2, false);
    auto lower_ = lower.toDouble();
    auto upper_ = upper.toDouble();
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    THNN_CudaRReLU_updateOutput(context->thc_state, self_->tensor, self_->tensor, noise_->tensor, lower_, upper_, training, true, NULL);
    return self;
}
Tensor & CUDAFloatType::softmax_forward_out(Tensor & output, const Tensor & self, int64_t dim) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",2, false);
    THNN_CudaSoftMax_updateOutput(context->thc_state, self_->tensor, output_->tensor, dim);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAFloatType::softmax_forward(const Tensor & self, int64_t dim) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaSoftMax_updateOutput(context->thc_state, self_->tensor, output_->tensor, dim);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAFloatType::softmax_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",4, false);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_CudaSoftMax_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, output_->tensor, dim);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::softmax_backward(const Tensor & grad_output, const Tensor & self, int64_t dim, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",4, false);
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaSoftMax_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, output_->tensor, dim);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAFloatType::softplus_forward_out(Tensor & output, const Tensor & self, Scalar beta, Scalar threshold) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toDouble();
    auto threshold_ = threshold.toDouble();
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",3, false);
    THNN_CudaSoftPlus_updateOutput(context->thc_state, self_->tensor, output_->tensor, beta_, threshold_);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAFloatType::softplus_forward(const Tensor & self, Scalar beta, Scalar threshold) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toDouble();
    auto threshold_ = threshold.toDouble();
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaSoftPlus_updateOutput(context->thc_state, self_->tensor, output_->tensor, beta_, threshold_);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAFloatType::softplus_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto beta_ = beta.toDouble();
    auto threshold_ = threshold.toDouble();
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",5, false);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",5, false);
    THNN_CudaSoftPlus_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, output_->tensor, beta_, threshold_);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::softplus_backward(const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto beta_ = beta.toDouble();
    auto threshold_ = threshold.toDouble();
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",5, false);
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaSoftPlus_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, output_->tensor, beta_, threshold_);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAFloatType::softshrink_forward_out(Tensor & output, const Tensor & self, Scalar lambd) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto lambd_ = lambd.toDouble();
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",2, false);
    THNN_CudaSoftShrink_updateOutput(context->thc_state, self_->tensor, output_->tensor, lambd_);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAFloatType::softshrink_forward(const Tensor & self, Scalar lambd) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto lambd_ = lambd.toDouble();
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaSoftShrink_updateOutput(context->thc_state, self_->tensor, output_->tensor, lambd_);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAFloatType::softshrink_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar lambd) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto lambd_ = lambd.toDouble();
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaSoftShrink_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, lambd_);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::softshrink_backward(const Tensor & grad_output, const Tensor & self, Scalar lambd) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto lambd_ = lambd.toDouble();
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaSoftShrink_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, lambd_);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAFloatType::threshold_forward_out(Tensor & output, const Tensor & self, Scalar threshold, Scalar value) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto threshold_ = threshold.toDouble();
    auto value_ = value.toDouble();
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",3, false);
    THNN_CudaThreshold_updateOutput(context->thc_state, self_->tensor, output_->tensor, threshold_, value_, false);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAFloatType::threshold_forward(const Tensor & self, Scalar threshold, Scalar value) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto threshold_ = threshold.toDouble();
    auto value_ = value.toDouble();
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaThreshold_updateOutput(context->thc_state, self_->tensor, output_->tensor, threshold_, value_, false);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAFloatType::threshold_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar threshold, Scalar value) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto threshold_ = threshold.toDouble();
    auto value_ = value.toDouble();
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_CudaThreshold_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, threshold_, value_, false);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::threshold_backward(const Tensor & grad_output, const Tensor & self, Scalar threshold, Scalar value) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto threshold_ = threshold.toDouble();
    auto value_ = value.toDouble();
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaThreshold_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, threshold_, value_, false);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAFloatType::threshold_forward_(Tensor & self, Scalar threshold, Scalar value) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto threshold_ = threshold.toDouble();
    auto value_ = value.toDouble();
    THNN_CudaThreshold_updateOutput(context->thc_state, self_->tensor, self_->tensor, threshold_, value_, true);
    return self;
}
Tensor & CUDAFloatType::adaptive_avg_pool2d_forward_out(Tensor & output, const Tensor & self, IntList output_size) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 2);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",2, false);
    THNN_CudaSpatialAdaptiveAveragePooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[1], output_size_[0]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAFloatType::adaptive_avg_pool2d_forward(const Tensor & self, IntList output_size) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 2);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaSpatialAdaptiveAveragePooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[1], output_size_[0]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAFloatType::adaptive_avg_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",2, false);
    THNN_CudaSpatialAdaptiveAveragePooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::adaptive_avg_pool2d_backward(const Tensor & grad_output, const Tensor & self) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaSpatialAdaptiveAveragePooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAFloatType::adaptive_avg_pool3d_forward_out(Tensor & output, const Tensor & self, IntList output_size) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 2);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",2, false);
    THNN_CudaVolumetricAdaptiveAveragePooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[0], output_size_[2], output_size_[1]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAFloatType::adaptive_avg_pool3d_forward(const Tensor & self, IntList output_size) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 2);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaVolumetricAdaptiveAveragePooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[0], output_size_[2], output_size_[1]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAFloatType::adaptive_avg_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",2, false);
    THNN_CudaVolumetricAdaptiveAveragePooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::adaptive_avg_pool3d_backward(const Tensor & grad_output, const Tensor & self) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaVolumetricAdaptiveAveragePooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &> CUDAFloatType::adaptive_max_pool2d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList output_size) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 2);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",2, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",2, false);
    THNN_CudaSpatialAdaptiveMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, output_size_[1], output_size_[0]);
    bool maybe_scalar = self_->isScalar();
    output_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(output, indices);
}
std::tuple<Tensor,Tensor> CUDAFloatType::adaptive_max_pool2d_forward(const Tensor & self, IntList output_size) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 2);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    auto indices_ = new CUDALongTensor(context);
    auto indices = Tensor(indices_, false);
    THNN_CudaSpatialAdaptiveMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, output_size_[1], output_size_[0]);
    bool maybe_scalar = self_->isScalar();
    output_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(output, indices);
}
Tensor & CUDAFloatType::adaptive_max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",3, false);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaSpatialAdaptiveMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::adaptive_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",3, false);
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaSpatialAdaptiveMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &> CUDAFloatType::adaptive_max_pool3d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList output_size) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 2);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",2, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",2, false);
    THNN_CudaVolumetricAdaptiveMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, output_size_[0], output_size_[2], output_size_[1]);
    bool maybe_scalar = self_->isScalar();
    output_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(output, indices);
}
std::tuple<Tensor,Tensor> CUDAFloatType::adaptive_max_pool3d_forward(const Tensor & self, IntList output_size) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 2);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    auto indices_ = new CUDALongTensor(context);
    auto indices = Tensor(indices_, false);
    THNN_CudaVolumetricAdaptiveMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, output_size_[0], output_size_[2], output_size_[1]);
    bool maybe_scalar = self_->isScalar();
    output_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(output, indices);
}
Tensor & CUDAFloatType::adaptive_max_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",3, false);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaVolumetricAdaptiveMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::adaptive_max_pool3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",3, false);
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaVolumetricAdaptiveMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAFloatType::avg_pool2d_forward_out(Tensor & output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 2);
    auto stride_ = check_intlist<2>(stride, "stride", 3, kernel_size);
    auto padding_ = check_intlist<2>(padding, "padding", 4);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",6, false);
    THNN_CudaSpatialAveragePooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], ceil_mode, count_include_pad);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAFloatType::avg_pool2d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 2);
    auto stride_ = check_intlist<2>(stride, "stride", 3, kernel_size);
    auto padding_ = check_intlist<2>(padding, "padding", 4);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaSpatialAveragePooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], ceil_mode, count_include_pad);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAFloatType::avg_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto stride_ = check_intlist<2>(stride, "stride", 4, kernel_size);
    auto padding_ = check_intlist<2>(padding, "padding", 5);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",7, false);
    THNN_CudaSpatialAveragePooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], ceil_mode, count_include_pad);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::avg_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto stride_ = check_intlist<2>(stride, "stride", 4, kernel_size);
    auto padding_ = check_intlist<2>(padding, "padding", 5);
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaSpatialAveragePooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], ceil_mode, count_include_pad);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAFloatType::avg_pool3d_forward_out(Tensor & output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 2);
    auto stride_ = check_intlist<3>(stride, "stride", 3, kernel_size);
    auto padding_ = check_intlist<3>(padding, "padding", 4);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",6, false);
    THNN_CudaVolumetricAveragePooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], ceil_mode, count_include_pad);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAFloatType::avg_pool3d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 2);
    auto stride_ = check_intlist<3>(stride, "stride", 3, kernel_size);
    auto padding_ = check_intlist<3>(padding, "padding", 4);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaVolumetricAveragePooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], ceil_mode, count_include_pad);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAFloatType::avg_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto stride_ = check_intlist<3>(stride, "stride", 4, kernel_size);
    auto padding_ = check_intlist<3>(padding, "padding", 5);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",7, false);
    THNN_CudaVolumetricAveragePooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], ceil_mode, count_include_pad);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::avg_pool3d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto stride_ = check_intlist<3>(stride, "stride", 4, kernel_size);
    auto padding_ = check_intlist<3>(padding, "padding", 5);
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaVolumetricAveragePooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], ceil_mode, count_include_pad);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &> CUDAFloatType::fractional_max_pool2d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & random_samples) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 2);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 3);
    auto random_samples_ = checked_cast_tensor<CUDAFloatTensor>(random_samples.pImpl,"random_samples",4, false);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",4, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",4, false);
    THNN_CudaSpatialFractionalMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[1], output_size_[0], kernel_size_[1], kernel_size_[0], indices_->tensor, random_samples_->tensor);
    output_->maybeScalar(false);
    return std::tuple<Tensor &, Tensor &>(output, indices);
}
std::tuple<Tensor,Tensor> CUDAFloatType::fractional_max_pool2d_forward(const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & random_samples) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 2);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 3);
    auto random_samples_ = checked_cast_tensor<CUDAFloatTensor>(random_samples.pImpl,"random_samples",4, false);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    auto indices_ = new CUDALongTensor(context);
    auto indices = Tensor(indices_, false);
    THNN_CudaSpatialFractionalMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[1], output_size_[0], kernel_size_[1], kernel_size_[0], indices_->tensor, random_samples_->tensor);
    output_->maybeScalar(false);
    return std::tuple<Tensor, Tensor>(output, indices);
}
Tensor & CUDAFloatType::fractional_max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 4);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",5, false);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",5, false);
    THNN_CudaSpatialFractionalMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, output_size_[1], output_size_[0], kernel_size_[1], kernel_size_[0], indices_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::fractional_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 4);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",5, false);
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaSpatialFractionalMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, output_size_[1], output_size_[0], kernel_size_[1], kernel_size_[0], indices_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &> CUDAFloatType::max_pool2d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 2);
    auto stride_ = check_intlist<2>(stride, "stride", 3, kernel_size);
    auto padding_ = check_intlist<2>(padding, "padding", 4);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 5);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",6, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",6, false);
    THNN_CudaSpatialDilatedMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], ceil_mode);
    bool maybe_scalar = self_->isScalar();
    output_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(output, indices);
}
std::tuple<Tensor,Tensor> CUDAFloatType::max_pool2d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 2);
    auto stride_ = check_intlist<2>(stride, "stride", 3, kernel_size);
    auto padding_ = check_intlist<2>(padding, "padding", 4);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 5);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    auto indices_ = new CUDALongTensor(context);
    auto indices = Tensor(indices_, false);
    THNN_CudaSpatialDilatedMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], ceil_mode);
    bool maybe_scalar = self_->isScalar();
    output_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(output, indices);
}
Tensor & CUDAFloatType::max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto stride_ = check_intlist<2>(stride, "stride", 4, kernel_size);
    auto padding_ = check_intlist<2>(padding, "padding", 5);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 6);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",8, false);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",8, false);
    THNN_CudaSpatialDilatedMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], ceil_mode);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::max_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto stride_ = check_intlist<2>(stride, "stride", 4, kernel_size);
    auto padding_ = check_intlist<2>(padding, "padding", 5);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 6);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",8, false);
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaSpatialDilatedMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], ceil_mode);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &> CUDAFloatType::max_pool3d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 2);
    auto stride_ = check_intlist<3>(stride, "stride", 3, kernel_size);
    auto padding_ = check_intlist<3>(padding, "padding", 4);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 5);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",6, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",6, false);
    THNN_CudaVolumetricDilatedMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], ceil_mode);
    bool maybe_scalar = self_->isScalar();
    output_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(output, indices);
}
std::tuple<Tensor,Tensor> CUDAFloatType::max_pool3d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 2);
    auto stride_ = check_intlist<3>(stride, "stride", 3, kernel_size);
    auto padding_ = check_intlist<3>(padding, "padding", 4);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 5);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    auto indices_ = new CUDALongTensor(context);
    auto indices = Tensor(indices_, false);
    THNN_CudaVolumetricDilatedMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], ceil_mode);
    bool maybe_scalar = self_->isScalar();
    output_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(output, indices);
}
Tensor & CUDAFloatType::max_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto stride_ = check_intlist<3>(stride, "stride", 4, kernel_size);
    auto padding_ = check_intlist<3>(padding, "padding", 5);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 6);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",8, false);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",8, false);
    THNN_CudaVolumetricDilatedMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], ceil_mode);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::max_pool3d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto stride_ = check_intlist<3>(stride, "stride", 4, kernel_size);
    auto padding_ = check_intlist<3>(padding, "padding", 5);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 6);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",8, false);
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaVolumetricDilatedMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], ceil_mode);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAFloatType::max_unpool2d_forward_out(Tensor & output, const Tensor & self, const Tensor & indices, IntList output_size) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",2, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 3);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",3, false);
    THNN_CudaSpatialMaxUnpooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, output_size_[1], output_size_[0]);
    output_->maybeScalar(self_->isScalar() && indices_->isScalar());
    return output;
}
Tensor CUDAFloatType::max_unpool2d_forward(const Tensor & self, const Tensor & indices, IntList output_size) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",2, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 3);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaSpatialMaxUnpooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, output_size_[1], output_size_[0]);
    output_->maybeScalar(self_->isScalar() && indices_->isScalar());
    return output;
}
Tensor & CUDAFloatType::max_unpool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",3, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 4);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_CudaSpatialMaxUnpooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor, output_size_[1], output_size_[0]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::max_unpool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",3, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 4);
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaSpatialMaxUnpooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor, output_size_[1], output_size_[0]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAFloatType::max_unpool3d_forward_out(Tensor & output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",2, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 3);
    auto stride_ = check_intlist<3>(stride, "stride", 4);
    auto padding_ = check_intlist<3>(padding, "padding", 5);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",5, false);
    THNN_CudaVolumetricMaxUnpooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, output_size_[0], output_size_[2], output_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1]);
    output_->maybeScalar(self_->isScalar() && indices_->isScalar());
    return output;
}
Tensor CUDAFloatType::max_unpool3d_forward(const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",2, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 3);
    auto stride_ = check_intlist<3>(stride, "stride", 4);
    auto padding_ = check_intlist<3>(padding, "padding", 5);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaVolumetricMaxUnpooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, output_size_[0], output_size_[2], output_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1]);
    output_->maybeScalar(self_->isScalar() && indices_->isScalar());
    return output;
}
Tensor & CUDAFloatType::max_unpool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",3, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 4);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",6, false);
    THNN_CudaVolumetricMaxUnpooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor, output_size_[0], output_size_[2], output_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::max_unpool3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",3, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 4);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaVolumetricMaxUnpooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor, output_size_[0], output_size_[2], output_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAFloatType::reflection_pad1d_forward_out(Tensor & output, const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<2>(padding, "padding", 2);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",2, false);
    THNN_CudaTemporalReflectionPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAFloatType::reflection_pad1d_forward(const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<2>(padding, "padding", 2);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaTemporalReflectionPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAFloatType::reflection_pad1d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<2>(padding, "padding", 3);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaTemporalReflectionPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::reflection_pad1d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<2>(padding, "padding", 3);
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaTemporalReflectionPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAFloatType::reflection_pad2d_forward_out(Tensor & output, const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<4>(padding, "padding", 2);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",2, false);
    THNN_CudaSpatialReflectionPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1], padding_[2], padding_[3]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAFloatType::reflection_pad2d_forward(const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<4>(padding, "padding", 2);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaSpatialReflectionPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1], padding_[2], padding_[3]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAFloatType::reflection_pad2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<4>(padding, "padding", 3);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaSpatialReflectionPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1], padding_[2], padding_[3]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::reflection_pad2d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<4>(padding, "padding", 3);
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaSpatialReflectionPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1], padding_[2], padding_[3]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAFloatType::replication_pad1d_forward_out(Tensor & output, const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<2>(padding, "padding", 2);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",2, false);
    THNN_CudaTemporalReplicationPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAFloatType::replication_pad1d_forward(const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<2>(padding, "padding", 2);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaTemporalReplicationPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAFloatType::replication_pad1d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<2>(padding, "padding", 3);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaTemporalReplicationPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::replication_pad1d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<2>(padding, "padding", 3);
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaTemporalReplicationPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAFloatType::replication_pad2d_forward_out(Tensor & output, const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<4>(padding, "padding", 2);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",2, false);
    THNN_CudaSpatialReplicationPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1], padding_[2], padding_[3]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAFloatType::replication_pad2d_forward(const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<4>(padding, "padding", 2);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaSpatialReplicationPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1], padding_[2], padding_[3]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAFloatType::replication_pad2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<4>(padding, "padding", 3);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaSpatialReplicationPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1], padding_[2], padding_[3]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::replication_pad2d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<4>(padding, "padding", 3);
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaSpatialReplicationPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1], padding_[2], padding_[3]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAFloatType::replication_pad3d_forward_out(Tensor & output, const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<6>(padding, "padding", 2);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",2, false);
    THNN_CudaVolumetricReplicationPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1], padding_[2], padding_[3], padding_[4], padding_[5]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAFloatType::replication_pad3d_forward(const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<6>(padding, "padding", 2);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaVolumetricReplicationPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1], padding_[2], padding_[3], padding_[4], padding_[5]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAFloatType::replication_pad3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<6>(padding, "padding", 3);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaVolumetricReplicationPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1], padding_[2], padding_[3], padding_[4], padding_[5]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::replication_pad3d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<6>(padding, "padding", 3);
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaVolumetricReplicationPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1], padding_[2], padding_[3], padding_[4], padding_[5]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAFloatType::upsample_linear1d_forward_out(Tensor & output, const Tensor & self, IntList output_size, bool align_corners) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<1>(output_size, "output_size", 2);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",3, false);
    THNN_CudaTemporalUpSamplingLinear_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[0], align_corners);
    return output;
}
Tensor CUDAFloatType::upsample_linear1d_forward(const Tensor & self, IntList output_size, bool align_corners) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<1>(output_size, "output_size", 2);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaTemporalUpSamplingLinear_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[0], align_corners);
    return output;
}
Tensor & CUDAFloatType::upsample_linear1d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_size_ = check_intlist<1>(output_size, "output_size", 2);
    auto input_size_ = check_intlist<3>(input_size, "input_size", 3);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_CudaTemporalUpSamplingLinear_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, input_size_[0], input_size_[1], input_size_[2], output_size_[0], align_corners);
    grad_input_->maybeScalar(false);
    return grad_input;
}
Tensor CUDAFloatType::upsample_linear1d_backward(const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_size_ = check_intlist<1>(output_size, "output_size", 2);
    auto input_size_ = check_intlist<3>(input_size, "input_size", 3);
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaTemporalUpSamplingLinear_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, input_size_[0], input_size_[1], input_size_[2], output_size_[0], align_corners);
    grad_input_->maybeScalar(false);
    return grad_input;
}
Tensor & CUDAFloatType::upsample_bilinear2d_forward_out(Tensor & output, const Tensor & self, IntList output_size, bool align_corners) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 2);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",3, false);
    THNN_CudaSpatialUpSamplingBilinear_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[0], output_size_[1], align_corners);
    return output;
}
Tensor CUDAFloatType::upsample_bilinear2d_forward(const Tensor & self, IntList output_size, bool align_corners) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 2);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaSpatialUpSamplingBilinear_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[0], output_size_[1], align_corners);
    return output;
}
Tensor & CUDAFloatType::upsample_bilinear2d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 2);
    auto input_size_ = check_intlist<4>(input_size, "input_size", 3);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_CudaSpatialUpSamplingBilinear_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, input_size_[0], input_size_[1], input_size_[2], input_size_[3], output_size_[0], output_size_[1], align_corners);
    grad_input_->maybeScalar(false);
    return grad_input;
}
Tensor CUDAFloatType::upsample_bilinear2d_backward(const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 2);
    auto input_size_ = check_intlist<4>(input_size, "input_size", 3);
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaSpatialUpSamplingBilinear_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, input_size_[0], input_size_[1], input_size_[2], input_size_[3], output_size_[0], output_size_[1], align_corners);
    grad_input_->maybeScalar(false);
    return grad_input;
}
Tensor & CUDAFloatType::upsample_trilinear3d_forward_out(Tensor & output, const Tensor & self, IntList output_size, bool align_corners) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 2);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",3, false);
    THNN_CudaVolumetricUpSamplingTrilinear_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[0], output_size_[1], output_size_[2], align_corners);
    return output;
}
Tensor CUDAFloatType::upsample_trilinear3d_forward(const Tensor & self, IntList output_size, bool align_corners) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 2);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaVolumetricUpSamplingTrilinear_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[0], output_size_[1], output_size_[2], align_corners);
    return output;
}
Tensor & CUDAFloatType::upsample_trilinear3d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 2);
    auto input_size_ = check_intlist<5>(input_size, "input_size", 3);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_CudaVolumetricUpSamplingTrilinear_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, input_size_[0], input_size_[1], input_size_[2], input_size_[3], input_size_[4], output_size_[0], output_size_[1], output_size_[2], align_corners);
    grad_input_->maybeScalar(false);
    return grad_input;
}
Tensor CUDAFloatType::upsample_trilinear3d_backward(const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 2);
    auto input_size_ = check_intlist<5>(input_size, "input_size", 3);
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaVolumetricUpSamplingTrilinear_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, input_size_[0], input_size_[1], input_size_[2], input_size_[3], input_size_[4], output_size_[0], output_size_[1], output_size_[2], align_corners);
    grad_input_->maybeScalar(false);
    return grad_input;
}
Tensor & CUDAFloatType::upsample_nearest1d_forward_out(Tensor & output, const Tensor & self, int64_t scale_factor) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",2, false);
    THNN_CudaTemporalUpSamplingNearest_updateOutput(context->thc_state, self_->tensor, output_->tensor, scale_factor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAFloatType::upsample_nearest1d_forward(const Tensor & self, int64_t scale_factor) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaTemporalUpSamplingNearest_updateOutput(context->thc_state, self_->tensor, output_->tensor, scale_factor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAFloatType::upsample_nearest1d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaTemporalUpSamplingNearest_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, scale_factor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::upsample_nearest1d_backward(const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaTemporalUpSamplingNearest_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, scale_factor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAFloatType::upsample_nearest2d_forward_out(Tensor & output, const Tensor & self, int64_t scale_factor) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",2, false);
    THNN_CudaSpatialUpSamplingNearest_updateOutput(context->thc_state, self_->tensor, output_->tensor, scale_factor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAFloatType::upsample_nearest2d_forward(const Tensor & self, int64_t scale_factor) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaSpatialUpSamplingNearest_updateOutput(context->thc_state, self_->tensor, output_->tensor, scale_factor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAFloatType::upsample_nearest2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaSpatialUpSamplingNearest_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, scale_factor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::upsample_nearest2d_backward(const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaSpatialUpSamplingNearest_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, scale_factor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAFloatType::upsample_nearest3d_forward_out(Tensor & output, const Tensor & self, int64_t scale_factor) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",2, false);
    THNN_CudaVolumetricUpSamplingNearest_updateOutput(context->thc_state, self_->tensor, output_->tensor, scale_factor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAFloatType::upsample_nearest3d_forward(const Tensor & self, int64_t scale_factor) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaVolumetricUpSamplingNearest_updateOutput(context->thc_state, self_->tensor, output_->tensor, scale_factor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAFloatType::upsample_nearest3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaVolumetricUpSamplingNearest_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, scale_factor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::upsample_nearest3d_backward(const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaVolumetricUpSamplingNearest_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, scale_factor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAFloatType::_sigmoid_forward_out(Tensor & output, const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",1, false);
    THNN_CudaSigmoid_updateOutput(context->thc_state, self_->tensor, output_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAFloatType::_sigmoid_forward(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaSigmoid_updateOutput(context->thc_state, self_->tensor, output_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAFloatType::_sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",2, false);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",2, false);
    THNN_CudaSigmoid_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, output_->tensor);
    grad_input_->maybeScalar(output_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::_sigmoid_backward(const Tensor & grad_output, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",2, false);
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaSigmoid_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, output_->tensor);
    grad_input_->maybeScalar(output_->isScalar());
    return grad_input;
}
Tensor & CUDAFloatType::_tanh_forward_out(Tensor & output, const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",1, false);
    THNN_CudaTanh_updateOutput(context->thc_state, self_->tensor, output_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAFloatType::_tanh_forward(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaTanh_updateOutput(context->thc_state, self_->tensor, output_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAFloatType::_tanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",2, false);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",2, false);
    THNN_CudaTanh_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, output_->tensor);
    grad_input_->maybeScalar(output_->isScalar());
    return grad_input;
}
Tensor CUDAFloatType::_tanh_backward(const Tensor & grad_output, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",2, false);
    auto grad_input_ = new CUDAFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaTanh_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, output_->tensor);
    grad_input_->maybeScalar(output_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDAFloatType::thnn_batch_norm_forward_out(Tensor & output, Tensor & save_mean, Tensor & save_std, const Tensor & self, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",2, true);
    auto bias_ = checked_cast_tensor<CUDAFloatTensor>(bias.pImpl,"bias",3, true);
    auto running_mean_ = checked_cast_tensor<CUDAFloatTensor>(running_mean.pImpl,"running_mean",4, true);
    auto running_var_ = checked_cast_tensor<CUDAFloatTensor>(running_var.pImpl,"running_var",5, true);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",8, false);
    auto save_mean_ = checked_cast_tensor<CUDAFloatTensor>(save_mean.pImpl,"save_mean",8, false);
    auto save_std_ = checked_cast_tensor<CUDAFloatTensor>(save_std.pImpl,"save_std",8, false);
    THNN_CudaBatchNormalization_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_ ? weight_->tensor : NULL, bias_ ? bias_->tensor : NULL, running_mean_ ? running_mean_->tensor : NULL, running_var_ ? running_var_->tensor : NULL, save_mean_->tensor, save_std_->tensor, training, momentum, eps);
    bool maybe_scalar = self_->isScalar() && (!weight_ || weight_->isScalar()) && (!bias_ || bias_->isScalar()) && (!running_mean_ || running_mean_->isScalar()) && (!running_var_ || running_var_->isScalar());
    output_->maybeScalar(maybe_scalar);
    save_mean_->maybeScalar(maybe_scalar);
    save_std_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(output, save_mean, save_std);
}
std::tuple<Tensor,Tensor,Tensor> CUDAFloatType::thnn_batch_norm_forward(const Tensor & self, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",2, true);
    auto bias_ = checked_cast_tensor<CUDAFloatTensor>(bias.pImpl,"bias",3, true);
    auto running_mean_ = checked_cast_tensor<CUDAFloatTensor>(running_mean.pImpl,"running_mean",4, true);
    auto running_var_ = checked_cast_tensor<CUDAFloatTensor>(running_var.pImpl,"running_var",5, true);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    auto save_mean_ = new CUDAFloatTensor(context);
    auto save_mean = Tensor(save_mean_, false);
    auto save_std_ = new CUDAFloatTensor(context);
    auto save_std = Tensor(save_std_, false);
    THNN_CudaBatchNormalization_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_ ? weight_->tensor : NULL, bias_ ? bias_->tensor : NULL, running_mean_ ? running_mean_->tensor : NULL, running_var_ ? running_var_->tensor : NULL, save_mean_->tensor, save_std_->tensor, training, momentum, eps);
    bool maybe_scalar = self_->isScalar() && (!weight_ || weight_->isScalar()) && (!bias_ || bias_->isScalar()) && (!running_mean_ || running_mean_->isScalar()) && (!running_var_ || running_var_->isScalar());
    output_->maybeScalar(maybe_scalar);
    save_mean_->maybeScalar(maybe_scalar);
    save_std_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(output, save_mean, save_std);
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDAFloatType::thnn_batch_norm_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, bool training, double eps, const Tensor & save_mean, const Tensor & save_std) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",3, true);
    auto running_mean_ = checked_cast_tensor<CUDAFloatTensor>(running_mean.pImpl,"running_mean",4, true);
    auto running_var_ = checked_cast_tensor<CUDAFloatTensor>(running_var.pImpl,"running_var",5, true);
    auto save_mean_ = checked_cast_tensor<CUDAFloatTensor>(save_mean.pImpl,"save_mean",8, true);
    auto save_std_ = checked_cast_tensor<CUDAFloatTensor>(save_std.pImpl,"save_std",9, true);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",9, true);
    auto grad_weight_ = checked_cast_tensor<CUDAFloatTensor>(grad_weight.pImpl,"grad_weight",9, true);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = checked_cast_tensor<CUDAFloatTensor>(grad_bias.pImpl,"grad_bias",9, true);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    THNN_CudaBatchNormalization_backward(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, weight_ ? weight_->tensor : NULL, running_mean_ ? running_mean_->tensor : NULL, running_var_ ? running_var_->tensor : NULL, save_mean_ ? save_mean_->tensor : NULL, save_std_ ? save_std_->tensor : NULL, training, 1, eps);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &, Tensor &>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> CUDAFloatType::thnn_batch_norm_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, bool training, double eps, const Tensor & save_mean, const Tensor & save_std, std::array<bool,3> output_mask) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",3, true);
    auto running_mean_ = checked_cast_tensor<CUDAFloatTensor>(running_mean.pImpl,"running_mean",4, true);
    auto running_var_ = checked_cast_tensor<CUDAFloatTensor>(running_var.pImpl,"running_var",5, true);
    auto save_mean_ = checked_cast_tensor<CUDAFloatTensor>(save_mean.pImpl,"save_mean",8, true);
    auto save_std_ = checked_cast_tensor<CUDAFloatTensor>(save_std.pImpl,"save_std",9, true);
    auto grad_input_ = output_mask[0] ? new CUDAFloatTensor(context) : nullptr;
    auto grad_input = Tensor(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_input_, false);
    auto grad_weight_ = output_mask[1] ? new CUDAFloatTensor(context) : nullptr;
    auto grad_weight = Tensor(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_weight_, false);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = output_mask[2] ? new CUDAFloatTensor(context) : nullptr;
    auto grad_bias = Tensor(grad_bias_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_bias_, false);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    THNN_CudaBatchNormalization_backward(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, weight_ ? weight_->tensor : NULL, running_mean_ ? running_mean_->tensor : NULL, running_var_ ? running_var_->tensor : NULL, save_mean_ ? save_mean_->tensor : NULL, save_std_ ? save_std_->tensor : NULL, training, 1, eps);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDAFloatType::thnn_conv_transpose2d_forward_out(Tensor & output, Tensor & columns, Tensor & ones, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CUDAFloatTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto output_padding_ = check_intlist<2>(output_padding, "output_padding", 7);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 8);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",8, false);
    auto columns_ = checked_cast_tensor<CUDAFloatTensor>(columns.pImpl,"columns",8, false);
    auto ones_ = checked_cast_tensor<CUDAFloatTensor>(ones.pImpl,"ones",8, false);
    THNN_CudaSpatialFullDilatedConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], output_padding_[1], output_padding_[0]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    columns_->maybeScalar(maybe_scalar);
    ones_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(output, columns, ones);
}
std::tuple<Tensor,Tensor,Tensor> CUDAFloatType::thnn_conv_transpose2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CUDAFloatTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto output_padding_ = check_intlist<2>(output_padding, "output_padding", 7);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 8);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    auto columns_ = new CUDAFloatTensor(context);
    auto columns = Tensor(columns_, false);
    auto ones_ = new CUDAFloatTensor(context);
    auto ones = Tensor(ones_, false);
    THNN_CudaSpatialFullDilatedConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], output_padding_[1], output_padding_[0]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    columns_->maybeScalar(maybe_scalar);
    ones_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(output, columns, ones);
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDAFloatType::thnn_conv_transpose2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & columns, const Tensor & ones) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto output_padding_ = check_intlist<2>(output_padding, "output_padding", 7);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 8);
    auto columns_ = checked_cast_tensor<CUDAFloatTensor>(columns.pImpl,"columns",9, false);
    auto ones_ = checked_cast_tensor<CUDAFloatTensor>(ones.pImpl,"ones",10, false);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",10, true);
    auto grad_weight_ = checked_cast_tensor<CUDAFloatTensor>(grad_weight.pImpl,"grad_weight",10, true);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = checked_cast_tensor<CUDAFloatTensor>(grad_bias.pImpl,"grad_bias",10, true);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(1) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_CudaSpatialFullDilatedConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, columns_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], output_padding_[1], output_padding_[0]);
    if (grad_weight_ || grad_bias_) THNN_CudaSpatialFullDilatedConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], output_padding_[1], output_padding_[0], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &, Tensor &>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> CUDAFloatType::thnn_conv_transpose2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto output_padding_ = check_intlist<2>(output_padding, "output_padding", 7);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 8);
    auto columns_ = checked_cast_tensor<CUDAFloatTensor>(columns.pImpl,"columns",9, false);
    auto ones_ = checked_cast_tensor<CUDAFloatTensor>(ones.pImpl,"ones",10, false);
    auto grad_input_ = output_mask[0] ? new CUDAFloatTensor(context) : nullptr;
    auto grad_input = Tensor(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_input_, false);
    auto grad_weight_ = output_mask[1] ? new CUDAFloatTensor(context) : nullptr;
    auto grad_weight = Tensor(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_weight_, false);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = output_mask[2] ? new CUDAFloatTensor(context) : nullptr;
    auto grad_bias = Tensor(grad_bias_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_bias_, false);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(1) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_CudaSpatialFullDilatedConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, columns_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], output_padding_[1], output_padding_[0]);
    if (grad_weight_ || grad_bias_) THNN_CudaSpatialFullDilatedConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], output_padding_[1], output_padding_[0], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDAFloatType::thnn_conv_transpose3d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CUDAFloatTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto output_padding_ = check_intlist<3>(output_padding, "output_padding", 7);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 8);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",8, false);
    auto finput_ = checked_cast_tensor<CUDAFloatTensor>(finput.pImpl,"finput",8, false);
    auto fgrad_input_ = checked_cast_tensor<CUDAFloatTensor>(fgrad_input.pImpl,"fgrad_input",8, false);
    THNN_CudaVolumetricFullDilatedConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], output_padding_[0], output_padding_[2], output_padding_[1]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    finput_->maybeScalar(maybe_scalar);
    fgrad_input_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(output, finput, fgrad_input);
}
std::tuple<Tensor,Tensor,Tensor> CUDAFloatType::thnn_conv_transpose3d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CUDAFloatTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto output_padding_ = check_intlist<3>(output_padding, "output_padding", 7);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 8);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    auto finput_ = new CUDAFloatTensor(context);
    auto finput = Tensor(finput_, false);
    auto fgrad_input_ = new CUDAFloatTensor(context);
    auto fgrad_input = Tensor(fgrad_input_, false);
    THNN_CudaVolumetricFullDilatedConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], output_padding_[0], output_padding_[2], output_padding_[1]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    finput_->maybeScalar(maybe_scalar);
    fgrad_input_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(output, finput, fgrad_input);
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDAFloatType::thnn_conv_transpose3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & finput, const Tensor & fgrad_input) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto output_padding_ = check_intlist<3>(output_padding, "output_padding", 7);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 8);
    auto finput_ = checked_cast_tensor<CUDAFloatTensor>(finput.pImpl,"finput",9, false);
    auto fgrad_input_ = checked_cast_tensor<CUDAFloatTensor>(fgrad_input.pImpl,"fgrad_input",10, false);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",10, true);
    auto grad_weight_ = checked_cast_tensor<CUDAFloatTensor>(grad_weight.pImpl,"grad_weight",10, true);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = checked_cast_tensor<CUDAFloatTensor>(grad_bias.pImpl,"grad_bias",10, true);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(1) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_CudaVolumetricFullDilatedConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], output_padding_[0], output_padding_[2], output_padding_[1]);
    if (grad_weight_ || grad_bias_) THNN_CudaVolumetricFullDilatedConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], output_padding_[0], output_padding_[2], output_padding_[1], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &, Tensor &>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> CUDAFloatType::thnn_conv_transpose3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto output_padding_ = check_intlist<3>(output_padding, "output_padding", 7);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 8);
    auto finput_ = checked_cast_tensor<CUDAFloatTensor>(finput.pImpl,"finput",9, false);
    auto fgrad_input_ = checked_cast_tensor<CUDAFloatTensor>(fgrad_input.pImpl,"fgrad_input",10, false);
    auto grad_input_ = output_mask[0] ? new CUDAFloatTensor(context) : nullptr;
    auto grad_input = Tensor(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_input_, false);
    auto grad_weight_ = output_mask[1] ? new CUDAFloatTensor(context) : nullptr;
    auto grad_weight = Tensor(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_weight_, false);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = output_mask[2] ? new CUDAFloatTensor(context) : nullptr;
    auto grad_bias = Tensor(grad_bias_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_bias_, false);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(1) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_CudaVolumetricFullDilatedConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], output_padding_[0], output_padding_[2], output_padding_[1]);
    if (grad_weight_ || grad_bias_) THNN_CudaVolumetricFullDilatedConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], output_padding_[0], output_padding_[2], output_padding_[1], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDAFloatType::thnn_conv2d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CUDAFloatTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",6, false);
    auto finput_ = checked_cast_tensor<CUDAFloatTensor>(finput.pImpl,"finput",6, false);
    auto fgrad_input_ = checked_cast_tensor<CUDAFloatTensor>(fgrad_input.pImpl,"fgrad_input",6, false);
    THNN_CudaSpatialConvolutionMM_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    finput_->maybeScalar(maybe_scalar);
    fgrad_input_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(output, finput, fgrad_input);
}
std::tuple<Tensor,Tensor,Tensor> CUDAFloatType::thnn_conv2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CUDAFloatTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    auto finput_ = new CUDAFloatTensor(context);
    auto finput = Tensor(finput_, false);
    auto fgrad_input_ = new CUDAFloatTensor(context);
    auto fgrad_input = Tensor(fgrad_input_, false);
    THNN_CudaSpatialConvolutionMM_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    finput_->maybeScalar(maybe_scalar);
    fgrad_input_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(output, finput, fgrad_input);
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDAFloatType::thnn_conv2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto finput_ = checked_cast_tensor<CUDAFloatTensor>(finput.pImpl,"finput",7, false);
    auto fgrad_input_ = checked_cast_tensor<CUDAFloatTensor>(fgrad_input.pImpl,"fgrad_input",8, false);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",8, true);
    auto grad_weight_ = checked_cast_tensor<CUDAFloatTensor>(grad_weight.pImpl,"grad_weight",8, true);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = checked_cast_tensor<CUDAFloatTensor>(grad_bias.pImpl,"grad_bias",8, true);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_CudaSpatialConvolutionMM_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, finput_->tensor, fgrad_input_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
    if (grad_weight_ || grad_bias_) THNN_CudaSpatialConvolutionMM_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &, Tensor &>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> CUDAFloatType::thnn_conv2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto finput_ = checked_cast_tensor<CUDAFloatTensor>(finput.pImpl,"finput",7, false);
    auto fgrad_input_ = checked_cast_tensor<CUDAFloatTensor>(fgrad_input.pImpl,"fgrad_input",8, false);
    auto grad_input_ = output_mask[0] ? new CUDAFloatTensor(context) : nullptr;
    auto grad_input = Tensor(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_input_, false);
    auto grad_weight_ = output_mask[1] ? new CUDAFloatTensor(context) : nullptr;
    auto grad_weight = Tensor(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_weight_, false);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = output_mask[2] ? new CUDAFloatTensor(context) : nullptr;
    auto grad_bias = Tensor(grad_bias_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_bias_, false);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_CudaSpatialConvolutionMM_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, finput_->tensor, fgrad_input_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
    if (grad_weight_ || grad_bias_) THNN_CudaSpatialConvolutionMM_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}
Tensor & CUDAFloatType::thnn_conv_depthwise2d_forward_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CUDAFloatTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",7, false);
    THNN_CudaSpatialDepthwiseConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
    output_->maybeScalar(self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar()));
    return output;
}
Tensor CUDAFloatType::thnn_conv_depthwise2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CUDAFloatTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaSpatialDepthwiseConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
    output_->maybeScalar(self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar()));
    return output;
}
std::tuple<Tensor &,Tensor &> CUDAFloatType::thnn_conv_depthwise2d_backward_out(Tensor & grad_input, Tensor & grad_weight, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",7, true);
    auto grad_weight_ = checked_cast_tensor<CUDAFloatTensor>(grad_weight.pImpl,"grad_weight",7, true);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    if (grad_input_) THNN_CudaSpatialDepthwiseConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
    if (grad_weight_) THNN_CudaSpatialDepthwiseConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &>(grad_input, grad_weight);
}
std::tuple<Tensor,Tensor> CUDAFloatType::thnn_conv_depthwise2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, std::array<bool,2> output_mask) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
    auto grad_input_ = output_mask[0] ? new CUDAFloatTensor(context) : nullptr;
    auto grad_input = Tensor(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_input_, false);
    auto grad_weight_ = output_mask[1] ? new CUDAFloatTensor(context) : nullptr;
    auto grad_weight = Tensor(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_weight_, false);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    if (grad_input_) THNN_CudaSpatialDepthwiseConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
    if (grad_weight_) THNN_CudaSpatialDepthwiseConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor>(grad_input, grad_weight);
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDAFloatType::thnn_conv_dilated2d_forward_out(Tensor & output, Tensor & columns, Tensor & ones, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CUDAFloatTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",7, false);
    auto columns_ = checked_cast_tensor<CUDAFloatTensor>(columns.pImpl,"columns",7, false);
    auto ones_ = checked_cast_tensor<CUDAFloatTensor>(ones.pImpl,"ones",7, false);
    THNN_CudaSpatialDilatedConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    columns_->maybeScalar(maybe_scalar);
    ones_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(output, columns, ones);
}
std::tuple<Tensor,Tensor,Tensor> CUDAFloatType::thnn_conv_dilated2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CUDAFloatTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    auto columns_ = new CUDAFloatTensor(context);
    auto columns = Tensor(columns_, false);
    auto ones_ = new CUDAFloatTensor(context);
    auto ones = Tensor(ones_, false);
    THNN_CudaSpatialDilatedConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    columns_->maybeScalar(maybe_scalar);
    ones_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(output, columns, ones);
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDAFloatType::thnn_conv_dilated2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
    auto columns_ = checked_cast_tensor<CUDAFloatTensor>(columns.pImpl,"columns",8, false);
    auto ones_ = checked_cast_tensor<CUDAFloatTensor>(ones.pImpl,"ones",9, false);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",9, true);
    auto grad_weight_ = checked_cast_tensor<CUDAFloatTensor>(grad_weight.pImpl,"grad_weight",9, true);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = checked_cast_tensor<CUDAFloatTensor>(grad_bias.pImpl,"grad_bias",9, true);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_CudaSpatialDilatedConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, columns_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
    if (grad_weight_ || grad_bias_) THNN_CudaSpatialDilatedConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &, Tensor &>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> CUDAFloatType::thnn_conv_dilated2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
    auto columns_ = checked_cast_tensor<CUDAFloatTensor>(columns.pImpl,"columns",8, false);
    auto ones_ = checked_cast_tensor<CUDAFloatTensor>(ones.pImpl,"ones",9, false);
    auto grad_input_ = output_mask[0] ? new CUDAFloatTensor(context) : nullptr;
    auto grad_input = Tensor(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_input_, false);
    auto grad_weight_ = output_mask[1] ? new CUDAFloatTensor(context) : nullptr;
    auto grad_weight = Tensor(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_weight_, false);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = output_mask[2] ? new CUDAFloatTensor(context) : nullptr;
    auto grad_bias = Tensor(grad_bias_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_bias_, false);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_CudaSpatialDilatedConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, columns_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
    if (grad_weight_ || grad_bias_) THNN_CudaSpatialDilatedConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDAFloatType::thnn_conv_dilated3d_forward_out(Tensor & output, Tensor & columns, Tensor & ones, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CUDAFloatTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 7);
    auto output_ = checked_cast_tensor<CUDAFloatTensor>(output.pImpl,"output",7, false);
    auto columns_ = checked_cast_tensor<CUDAFloatTensor>(columns.pImpl,"columns",7, false);
    auto ones_ = checked_cast_tensor<CUDAFloatTensor>(ones.pImpl,"ones",7, false);
    THNN_CudaVolumetricDilatedConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    columns_->maybeScalar(maybe_scalar);
    ones_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(output, columns, ones);
}
std::tuple<Tensor,Tensor,Tensor> CUDAFloatType::thnn_conv_dilated3d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CUDAFloatTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 7);
    auto output_ = new CUDAFloatTensor(context);
    auto output = Tensor(output_, false);
    auto columns_ = new CUDAFloatTensor(context);
    auto columns = Tensor(columns_, false);
    auto ones_ = new CUDAFloatTensor(context);
    auto ones = Tensor(ones_, false);
    THNN_CudaVolumetricDilatedConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    columns_->maybeScalar(maybe_scalar);
    ones_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(output, columns, ones);
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDAFloatType::thnn_conv_dilated3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 7);
    auto columns_ = checked_cast_tensor<CUDAFloatTensor>(columns.pImpl,"columns",8, false);
    auto ones_ = checked_cast_tensor<CUDAFloatTensor>(ones.pImpl,"ones",9, false);
    auto grad_input_ = checked_cast_tensor<CUDAFloatTensor>(grad_input.pImpl,"grad_input",9, true);
    auto grad_weight_ = checked_cast_tensor<CUDAFloatTensor>(grad_weight.pImpl,"grad_weight",9, true);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = checked_cast_tensor<CUDAFloatTensor>(grad_bias.pImpl,"grad_bias",9, true);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_CudaVolumetricDilatedConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, columns_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1]);
    if (grad_weight_ || grad_bias_) THNN_CudaVolumetricDilatedConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &, Tensor &>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> CUDAFloatType::thnn_conv_dilated3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) const {
    auto grad_output_ = checked_cast_tensor<CUDAFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAFloatTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDAFloatTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 7);
    auto columns_ = checked_cast_tensor<CUDAFloatTensor>(columns.pImpl,"columns",8, false);
    auto ones_ = checked_cast_tensor<CUDAFloatTensor>(ones.pImpl,"ones",9, false);
    auto grad_input_ = output_mask[0] ? new CUDAFloatTensor(context) : nullptr;
    auto grad_input = Tensor(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_input_, false);
    auto grad_weight_ = output_mask[1] ? new CUDAFloatTensor(context) : nullptr;
    auto grad_weight = Tensor(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_weight_, false);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = output_mask[2] ? new CUDAFloatTensor(context) : nullptr;
    auto grad_bias = Tensor(grad_bias_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_bias_, false);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_CudaVolumetricDilatedConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, columns_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1]);
    if (grad_weight_ || grad_bias_) THNN_CudaVolumetricDilatedConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}
Tensor & CUDAFloatType::abs_out(Tensor & result, const Tensor & self) const {
    return  at::native::_abs_out_cuda(result, self);
}
Tensor & CUDAFloatType::ceil_out(Tensor & result, const Tensor & self) const {
    return  at::native::_ceil_out_cuda(result, self);
}
Tensor & CUDAFloatType::cos_out(Tensor & result, const Tensor & self) const {
    return  at::native::_cos_out_cuda(result, self);
}
Tensor CUDAFloatType::embedding_dense_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) const {
    return  at::native::embedding_backward_cuda(grad, indices, num_weights, padding_idx, scale_grad_by_freq);
}
Tensor & CUDAFloatType::embedding_renorm_(Tensor & self, const Tensor & indices, double max_norm, double norm_type) const {
    return  at::native::embedding_renorm_cuda_(self, indices, max_norm, norm_type);
}
std::tuple<Tensor,Tensor,Tensor> CUDAFloatType::embedding_bag(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse) const {
    return  at::native::embedding_bag_cuda(weight, indices, offsets, scale_grad_by_freq, mode, sparse);
}
Tensor CUDAFloatType::embedding_bag_dense_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode) const {
    return  at::native::embedding_bag_backward_cuda(grad, indices, offsets, offset2bag, bag_size, num_weights, scale_grad_by_freq, mode);
}
Tensor & CUDAFloatType::exp_out(Tensor & result, const Tensor & self) const {
    return  at::native::_exp_out_cuda(result, self);
}
Tensor & CUDAFloatType::eye_out(Tensor & result, int64_t n, int64_t m) const {
    return  at::native::eye_out_cuda(result, n, m);
}
Tensor & CUDAFloatType::floor_out(Tensor & result, const Tensor & self) const {
    return  at::native::_floor_out_cuda(result, self);
}
Tensor CUDAFloatType::_fft_with_size(const Tensor & self, int64_t signal_ndim, bool complex_input, bool complex_output, bool inverse, IntList checked_signal_sizes, bool normalized, bool onesided, IntList output_sizes) const {
    return  at::native::_fft_cufft(self, signal_ndim, complex_input, complex_output, inverse, checked_signal_sizes, normalized, onesided, output_sizes);
}
Tensor & CUDAFloatType::log_out(Tensor & result, const Tensor & self) const {
    return  at::native::_log_out_cuda(result, self);
}
std::tuple<Tensor,Tensor> CUDAFloatType::RoiPooling2d_forward(const Tensor & input, const Tensor & rois, int64_t pooledHeight, int64_t pooledWidth, double spatialScale) const {
    return  at::native::RoiPooling2d_forward_cuda(input, rois, pooledHeight, pooledWidth, spatialScale);
}
Tensor CUDAFloatType::RoiPooling2d_backward(const Tensor & input, const Tensor & rois, int64_t pooledHeight, int64_t pooledWidth, double spatialScale, const Tensor & gradOutput, const Tensor & argmaxes) const {
    return  at::native::RoiPooling2d_backward_cuda(input, rois, pooledHeight, pooledWidth, spatialScale, gradOutput, argmaxes);
}
Tensor & CUDAFloatType::round_out(Tensor & result, const Tensor & self) const {
    return  at::native::_round_out_cuda(result, self);
}
Tensor & CUDAFloatType::sin_out(Tensor & result, const Tensor & self) const {
    return  at::native::_sin_out_cuda(result, self);
}
Tensor & CUDAFloatType::sspaddmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return  at::native::_sspaddmm_out_only_sparse(result, self, mat1, mat2, beta, alpha);
}
Tensor CUDAFloatType::sum(const Tensor & self) const {
    return  at::native::_sum_cuda(self);
}
Tensor & CUDAFloatType::sum_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    return  at::native::_sum_out_cuda(result, self, dim, keepdim);
}
Tensor & CUDAFloatType::sqrt_out(Tensor & result, const Tensor & self) const {
    return  at::native::_sqrt_out_cuda(result, self);
}
Tensor CUDAFloatType::prod(const Tensor & self) const {
    return  at::native::_prod_cuda(self);
}
Tensor & CUDAFloatType::prod_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    return  at::native::_prod_out_cuda(result, self, dim, keepdim);
}
Tensor & CUDAFloatType::trunc_out(Tensor & result, const Tensor & self) const {
    return  at::native::_trunc_out_cuda(result, self);
}
std::tuple<Tensor,Tensor> CUDAFloatType::_unique(const Tensor & self, bool sorted, bool return_inverse) const {
    return  at::native::_unique_cuda(self, sorted, return_inverse);
}
Tensor CUDAFloatType::_s_where(const Tensor & condition, const Tensor & self, const Tensor & other) const {
    return  at::native::_s_where_cuda(condition, self, other);
}
Tensor CUDAFloatType::_standard_gamma_grad(const Tensor & self, const Tensor & output) const {
    return  at::native::_standard_gamma_grad_cuda(self, output);
}
Tensor CUDAFloatType::poisson(const Tensor & self, Generator * generator) const {
    return  at::native::_s_poisson_cuda(self, generator);
}

}
