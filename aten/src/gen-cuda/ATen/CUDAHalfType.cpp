// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/CUDAHalfType.h"
#include "ATen/CUDAHalfStorage.h"
#include "ATen/CUDAHalfTensor.h"
#include "ATen/CUDAGenerator.h"
#include "ATen/CUDAByteTensor.h"
#include "ATen/CUDAIntTensor.h"
#include "ATen/CUDALongTensor.h"
#include "ATen/Tensor.h"
#include "ATen/CUDAHalfTensor.h"
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

CUDAHalfType::CUDAHalfType(Context* context)
: Type(context, /*is_variable_or_undefined=*/false) {}
ScalarType CUDAHalfType::scalarType() const {
  return ScalarType::Half;
}
Backend CUDAHalfType::backend() const {
  return Backend::CUDA;
}
bool CUDAHalfType::is_cuda() const { return backend() == kCUDA || backend() == kSparseCUDA; }
bool CUDAHalfType::is_sparse() const { return backend() == kSparseCPU || backend() == kSparseCUDA; }
bool CUDAHalfType::is_distributed() const { return false; }

std::unique_ptr<Storage> CUDAHalfType::storage() const {
  return std::unique_ptr<Storage>(new CUDAHalfStorage(context));
}
std::unique_ptr<Storage> CUDAHalfType::storage(size_t size) const {
  return std::unique_ptr<Storage>(new CUDAHalfStorage(context,size));
}
std::unique_ptr<Storage> CUDAHalfType::storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const {
    return std::unique_ptr<Storage>(
      new CUDAHalfStorage(context,data,size,deleter));
}
std::unique_ptr<Storage> CUDAHalfType::storageWithAllocator(int64_t size, std::unique_ptr<Allocator> allocator) const {
    return std::unique_ptr<Storage>(
        new CUDAHalfStorage(context, size, std::move(allocator)));
}
Tensor CUDAHalfType::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THCudaHalfTensor_retain(context->thc_state,  (THCudaHalfTensor*) th_pointer);
  return Tensor(new CUDAHalfTensor(context,(THCudaHalfTensor*)(th_pointer)), false);
}
std::unique_ptr<Storage> CUDAHalfType::unsafeStorageFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THCudaHalfStorage_retain(context->thc_state,  (THCudaHalfStorage*) th_pointer);
  return std::unique_ptr<Storage>(new CUDAHalfStorage(context, (THCudaHalfStorage*) th_pointer));
}
std::unique_ptr<Generator> CUDAHalfType::generator() const {
  return std::unique_ptr<Generator>(new CUDAGenerator(context));
}

const char * CUDAHalfType::toString() const {
  return CUDAHalfType::typeString();
}
TypeID CUDAHalfType::ID() const {
  return TypeID::CUDAHalf;
}

std::size_t CUDAHalfType::elementSizeInBytes() const {
  return sizeof(Half);
}

const char * CUDAHalfType::typeString() {
  return "CUDAHalfType";
}

/* example
Tensor * CUDAHalfType::add(Tensor & a, Tensor & b) {
  std::cout << "add CUDAHalfTensor\n";
  return &a;
}
*/

int64_t CUDAHalfType::storage_offset(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    return static_cast<int64_t>(THCudaHalfTensor_storageOffset(context->thc_state, self_->tensor));
}
Tensor & CUDAHalfType::resize_(Tensor & self, IntList size) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    THCudaHalfTensor_resize(context->thc_state, self_->tensor, size_, NULL);
    self_->maybeScalar(size.size() == 0);
    return self;
}
int64_t CUDAHalfType::numel(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    return static_cast<int64_t>(THCudaHalfTensor_nElement(context->thc_state, self_->tensor));
}
Tensor & CUDAHalfType::set_(Tensor & self, Storage & storage) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto storage_ = checked_cast_storage<CUDAHalfStorage>(&storage,"storage",2);
    THCudaHalfTensor_setStorage(context->thc_state, self_->tensor, storage_->storage, 0, THLongStorageView::makeFromLength(static_cast<int64_t>(storage.size())), NULL);
    self_->maybeScalar(false);
    return self;
}
Tensor & CUDAHalfType::set_(Tensor & self, Storage & sourceStorage, int64_t storage_offset, IntList size, IntList stride) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto sourceStorage_ = checked_cast_storage<CUDAHalfStorage>(&sourceStorage,"sourceStorage",2);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    THCudaHalfTensor_setStorage(context->thc_state, self_->tensor, sourceStorage_->storage, storage_offset, size_, stride_);
    self_->maybeScalar(size.size() == 0);
    return self;
}
Tensor & CUDAHalfType::set_(Tensor & self, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto source_ = checked_cast_tensor<CUDAHalfTensor>(source.pImpl,"source",2, false);
    THCudaHalfTensor_set(context->thc_state, self_->tensor, source_->tensor);
    self_->maybeScalar(source_->isScalar());
    return self;
}
Tensor & CUDAHalfType::set_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_setStorage(context->thc_state, self_->tensor, NULL, 0, NULL, NULL);
    self_->maybeScalar(false);
    return self;
}
Tensor & CUDAHalfType::fill_(Tensor & self, Scalar value) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toHalf();
    THCudaHalfTensor_fill(context->thc_state, self_->tensor, convert<half>(value_));
    return self;
}
Tensor & CUDAHalfType::fill_(Tensor & self, const Tensor & value) const {
    if (value.dim() == 0) {
        return static_cast<const Type*>(this)->fill_(self, Scalar(value));
    }
    AT_ERROR("fill_ only supports a 0-dimensional value tensor, but got tensor "
        "with %" PRId64 " dimension(s)", value.dim());
}
bool CUDAHalfType::is_contiguous(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    return THCudaHalfTensor_isContiguous(context->thc_state, self_->tensor);
}
bool CUDAHalfType::is_set_to(const Tensor & self, const Tensor & tensor) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto tensor_ = checked_cast_tensor<CUDAHalfTensor>(tensor.pImpl,"tensor",2, false);
    return THCudaHalfTensor_isSetTo(context->thc_state, self_->tensor, tensor_->tensor);
}
Tensor & CUDAHalfType::s_masked_fill_(Tensor & self, const Tensor & mask, Scalar value) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CUDAByteTensor>(mask.pImpl,"mask",2, false);
    auto value_ = value.toHalf();
    THCudaHalfTensor_maskedFill(context->thc_state, self_->tensor, mask_->tensor, convert<half>(value_));
    return self;
}
Tensor & CUDAHalfType::s_masked_fill_(Tensor & self, const Tensor & mask, const Tensor & value) const {
    if (value.dim() == 0) {
        return static_cast<const Type*>(this)->masked_fill_(self, mask, Scalar(value));
    }
    AT_ERROR("masked_fill_ only supports a 0-dimensional value tensor, but got tensor "
        "with %" PRId64 " dimension(s)", value.dim());
}
Tensor & CUDAHalfType::s_masked_scatter_(Tensor & self, const Tensor & mask, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CUDAByteTensor>(mask.pImpl,"mask",2, false);
    auto source_ = checked_cast_tensor<CUDAHalfTensor>(source.pImpl,"source",3, false);
    THCudaHalfTensor_maskedCopy(context->thc_state, self_->tensor, mask_->tensor, source_->tensor);
    return self;
}
Tensor & CUDAHalfType::s_masked_select_out(Tensor & result, const Tensor & self, const Tensor & mask) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CUDAByteTensor>(mask.pImpl,"mask",2, false);
    THCudaHalfTensor_maskedSelect(context->thc_state, result_->tensor, self_->tensor, mask_->tensor);
    result_->maybeScalar(self_->isScalar() && mask_->isScalar());
    return result;
}
Tensor CUDAHalfType::s_masked_select(const Tensor & self, const Tensor & mask) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CUDAByteTensor>(mask.pImpl,"mask",2, false);
    THCudaHalfTensor_maskedSelect(context->thc_state, result_->tensor, self_->tensor, mask_->tensor);
    result_->maybeScalar(self_->isScalar() && mask_->isScalar());
    return result;
}
Tensor CUDAHalfType::transpose(const Tensor & self, int64_t dim0, int64_t dim1) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim0 = maybe_wrap_dim(dim0, self_);
    dim1 = maybe_wrap_dim(dim1, self_);
    return Tensor((new CUDAHalfTensor(context, THCudaHalfTensor_newTranspose(context->thc_state, self_->tensor, dim0, dim1)))->maybeScalar(self_->isScalar()),false);
}
Tensor CUDAHalfType::t(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    return Tensor((new CUDAHalfTensor(context, THCudaHalfTensor_newTranspose(context->thc_state, self_->tensor, 0, 1)))->maybeScalar(self_->isScalar()),false);
}
Tensor & CUDAHalfType::nonzero_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDALongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_nonzero(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::nonzero(const Tensor & self) const {
    auto result_ = new CUDALongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_nonzero(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::contiguous(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    return Tensor((new CUDAHalfTensor(context, THCudaHalfTensor_newContiguous(context->thc_state, self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor CUDAHalfType::clone(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    return Tensor((new CUDAHalfTensor(context, THCudaHalfTensor_newClone(context->thc_state, self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor CUDAHalfType::view(const Tensor & self, IntList size) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new CUDAHalfTensor(context, THCudaHalfTensor_newView(context->thc_state, self_->tensor, size_)))->maybeScalar(size.size() == 0),false);
}
Tensor & CUDAHalfType::resize_as_(Tensor & self, const Tensor & the_template) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto the_template_ = checked_cast_tensor<CUDAHalfTensor>(the_template.pImpl,"the_template",2, false);
    THCudaHalfTensor_resizeAs(context->thc_state, self_->tensor, the_template_->tensor);
    self_->maybeScalar(the_template_->isScalar());
    return self;
}
Tensor & CUDAHalfType::index_select_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    THCudaHalfTensor_indexSelect(context->thc_state, result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
Tensor CUDAHalfType::index_select(const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    THCudaHalfTensor_indexSelect(context->thc_state, result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
Tensor & CUDAHalfType::_indexCopy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    auto source_ = checked_cast_tensor<CUDAHalfTensor>(source.pImpl,"source",4, false);
    THCudaHalfTensor_indexCopy(context->thc_state, self_->tensor, dim, index_->tensor, source_->tensor);
    return self;
}
Tensor & CUDAHalfType::take_out(Tensor & result, const Tensor & self, const Tensor & index) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",2, false);
    THCudaHalfTensor_take(context->thc_state, result_->tensor, self_->tensor, index_->tensor);
    result_->maybeScalar(index_->isScalar());
    return result;
}
Tensor CUDAHalfType::take(const Tensor & self, const Tensor & index) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",2, false);
    THCudaHalfTensor_take(context->thc_state, result_->tensor, self_->tensor, index_->tensor);
    result_->maybeScalar(index_->isScalar());
    return result;
}
Tensor & CUDAHalfType::put_(Tensor & self, const Tensor & index, const Tensor & source, bool accumulate) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",2, false);
    auto source_ = checked_cast_tensor<CUDAHalfTensor>(source.pImpl,"source",3, false);
    THCudaHalfTensor_put(context->thc_state, self_->tensor, index_->tensor, source_->tensor, accumulate);
    return self;
}
Tensor & CUDAHalfType::index_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    auto source_ = checked_cast_tensor<CUDAHalfTensor>(source.pImpl,"source",4, false);
    THCudaHalfTensor_indexAdd(context->thc_state, self_->tensor, dim, index_->tensor, source_->tensor);
    return self;
}
Tensor & CUDAHalfType::index_fill_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    auto value_ = value.toHalf();
    THCudaHalfTensor_indexFill(context->thc_state, self_->tensor, dim, index_->tensor, convert<half>(value_));
    return self;
}
Tensor & CUDAHalfType::index_fill_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & value) const {
    if (value.dim() == 0) {
        return static_cast<const Type*>(this)->index_fill_(self, dim, index, Scalar(value));
    }
    AT_ERROR("index_fill_ only supports a 0-dimensional value tensor, but got tensor "
        "with %" PRId64 " dimension(s)", value.dim());
}
Tensor CUDAHalfType::unfold(const Tensor & self, int64_t dimension, int64_t size, int64_t step) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dimension = maybe_wrap_dim(dimension, self_);
    THCudaHalfTensor_unfold(context->thc_state, result_->tensor, self_->tensor, dimension, size, step);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::_range_out(Tensor & result, Scalar start, Scalar end, Scalar step) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto start_ = start.toDouble();
    auto end_ = end.toDouble();
    auto step_ = step.toDouble();
    THCudaHalfTensor_range(context->thc_state, result_->tensor, start_, end_, step_);
    return result;
}
Tensor CUDAHalfType::_range(Scalar start, Scalar end, Scalar step) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto start_ = start.toDouble();
    auto end_ = end.toDouble();
    auto step_ = step.toDouble();
    THCudaHalfTensor_range(context->thc_state, result_->tensor, start_, end_, step_);
    return result;
}
Tensor & CUDAHalfType::_arange_out(Tensor & result, Scalar start, Scalar end, Scalar step) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto start_ = start.toDouble();
    auto end_ = end.toDouble();
    auto step_ = step.toDouble();
    THCudaHalfTensor_arange(context->thc_state, result_->tensor, start_, end_, step_);
    return result;
}
Tensor CUDAHalfType::_arange(Scalar start, Scalar end, Scalar step) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto start_ = start.toDouble();
    auto end_ = end.toDouble();
    auto step_ = step.toDouble();
    THCudaHalfTensor_arange(context->thc_state, result_->tensor, start_, end_, step_);
    return result;
}
Tensor & CUDAHalfType::_arange_out(Tensor & result, Scalar end) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto end_ = end.toDouble();
    THCudaHalfTensor_arange(context->thc_state, result_->tensor, 0, end_, 1);
    return result;
}
Tensor CUDAHalfType::_arange(Scalar end) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto end_ = end.toDouble();
    THCudaHalfTensor_arange(context->thc_state, result_->tensor, 0, end_, 1);
    return result;
}
Tensor & CUDAHalfType::scatter_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
    if (src.dim() == 0) {
        return static_cast<const Type*>(this)->scatter_(self, dim, index, Scalar(src));
    }
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    auto src_ = checked_cast_tensor<CUDAHalfTensor>(src.pImpl,"src",4, false);
    THCudaHalfTensor_scatter(context->thc_state, self_->tensor, dim, index_->tensor, src_->tensor);
    return self;
}
Tensor & CUDAHalfType::scatter_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    auto value_ = value.toHalf();
    THCudaHalfTensor_scatterFill(context->thc_state, self_->tensor, dim, index_->tensor, convert<half>(value_));
    return self;
}
Tensor & CUDAHalfType::scatter_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    auto src_ = checked_cast_tensor<CUDAHalfTensor>(src.pImpl,"src",4, false);
    THCudaHalfTensor_scatterAdd(context->thc_state, self_->tensor, dim, index_->tensor, src_->tensor);
    return self;
}
Tensor & CUDAHalfType::gather_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    result.resize_(index.sizes());
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    THCudaHalfTensor_gather(context->thc_state, result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
Tensor CUDAHalfType::gather(const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    result.resize_(index.sizes());
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    THCudaHalfTensor_gather(context->thc_state, result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
void* CUDAHalfType::data_ptr(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    return THCudaHalfTensor_data(context->thc_state, self_->tensor);
}
bool CUDAHalfType::equal(const Tensor & self, const Tensor & other) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",2, false);
    return THCudaHalfTensor_equal(context->thc_state, self_->tensor, other_->tensor);
}
Tensor & CUDAHalfType::__and___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_bitand(context->thc_state, result_->tensor, self_->tensor, convert<half>(other_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::__and__(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_bitand(context->thc_state, result_->tensor, self_->tensor, convert<half>(other_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::s___and___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__and___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",2, false);
    THCudaHalfTensor_cbitand(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAHalfType::s___and__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__and__(self, Scalar(other));
    }
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",2, false);
    THCudaHalfTensor_cbitand(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAHalfType::__iand__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_bitand(context->thc_state, self_->tensor, self_->tensor, convert<half>(other_));
    return self;
}
Tensor & CUDAHalfType::s___iand__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__iand__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",3, false);
    THCudaHalfTensor_cbitand(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAHalfType::__or___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_bitor(context->thc_state, result_->tensor, self_->tensor, convert<half>(other_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::__or__(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_bitor(context->thc_state, result_->tensor, self_->tensor, convert<half>(other_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::s___or___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__or___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",2, false);
    THCudaHalfTensor_cbitor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAHalfType::s___or__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__or__(self, Scalar(other));
    }
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",2, false);
    THCudaHalfTensor_cbitor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAHalfType::__ior__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_bitor(context->thc_state, self_->tensor, self_->tensor, convert<half>(other_));
    return self;
}
Tensor & CUDAHalfType::s___ior__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__ior__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",3, false);
    THCudaHalfTensor_cbitor(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAHalfType::__xor___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_bitxor(context->thc_state, result_->tensor, self_->tensor, convert<half>(other_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::__xor__(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_bitxor(context->thc_state, result_->tensor, self_->tensor, convert<half>(other_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::s___xor___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__xor___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",2, false);
    THCudaHalfTensor_cbitxor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAHalfType::s___xor__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__xor__(self, Scalar(other));
    }
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",2, false);
    THCudaHalfTensor_cbitxor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAHalfType::__ixor__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_bitxor(context->thc_state, self_->tensor, self_->tensor, convert<half>(other_));
    return self;
}
Tensor & CUDAHalfType::s___ixor__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__ixor__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",3, false);
    THCudaHalfTensor_cbitxor(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAHalfType::__lshift___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_lshift(context->thc_state, result_->tensor, self_->tensor, convert<half>(other_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::__lshift__(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_lshift(context->thc_state, result_->tensor, self_->tensor, convert<half>(other_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::s___lshift___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__lshift___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",2, false);
    THCudaHalfTensor_clshift(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAHalfType::s___lshift__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__lshift__(self, Scalar(other));
    }
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",2, false);
    THCudaHalfTensor_clshift(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAHalfType::__ilshift__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_lshift(context->thc_state, self_->tensor, self_->tensor, convert<half>(other_));
    return self;
}
Tensor & CUDAHalfType::s___ilshift__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__ilshift__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",3, false);
    THCudaHalfTensor_clshift(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAHalfType::__rshift___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_rshift(context->thc_state, result_->tensor, self_->tensor, convert<half>(other_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::__rshift__(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_rshift(context->thc_state, result_->tensor, self_->tensor, convert<half>(other_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::s___rshift___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__rshift___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",2, false);
    THCudaHalfTensor_crshift(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAHalfType::s___rshift__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__rshift__(self, Scalar(other));
    }
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",2, false);
    THCudaHalfTensor_crshift(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAHalfType::__irshift__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_rshift(context->thc_state, self_->tensor, self_->tensor, convert<half>(other_));
    return self;
}
Tensor & CUDAHalfType::s___irshift__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__irshift__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",3, false);
    THCudaHalfTensor_crshift(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAHalfType::lt_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_ltValue(context->thc_state, result_->tensor, self_->tensor, convert<half>(other_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::lt(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_ltValue(context->thc_state, result_->tensor, self_->tensor, convert<half>(other_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::s_lt_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->lt_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",2, false);
    THCudaHalfTensor_ltTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAHalfType::s_lt(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->lt(self, Scalar(other));
    }
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",2, false);
    THCudaHalfTensor_ltTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAHalfType::lt_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_ltValueT(context->thc_state, self_->tensor, self_->tensor, convert<half>(other_));
    return self;
}
Tensor & CUDAHalfType::s_lt_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->lt_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",3, false);
    THCudaHalfTensor_ltTensorT(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAHalfType::gt_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_gtValue(context->thc_state, result_->tensor, self_->tensor, convert<half>(other_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::gt(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_gtValue(context->thc_state, result_->tensor, self_->tensor, convert<half>(other_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::s_gt_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->gt_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",2, false);
    THCudaHalfTensor_gtTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAHalfType::s_gt(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->gt(self, Scalar(other));
    }
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",2, false);
    THCudaHalfTensor_gtTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAHalfType::gt_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_gtValueT(context->thc_state, self_->tensor, self_->tensor, convert<half>(other_));
    return self;
}
Tensor & CUDAHalfType::s_gt_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->gt_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",3, false);
    THCudaHalfTensor_gtTensorT(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAHalfType::le_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_leValue(context->thc_state, result_->tensor, self_->tensor, convert<half>(other_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::le(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_leValue(context->thc_state, result_->tensor, self_->tensor, convert<half>(other_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::s_le_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->le_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",2, false);
    THCudaHalfTensor_leTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAHalfType::s_le(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->le(self, Scalar(other));
    }
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",2, false);
    THCudaHalfTensor_leTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAHalfType::le_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_leValueT(context->thc_state, self_->tensor, self_->tensor, convert<half>(other_));
    return self;
}
Tensor & CUDAHalfType::s_le_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->le_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",3, false);
    THCudaHalfTensor_leTensorT(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAHalfType::ge_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_geValue(context->thc_state, result_->tensor, self_->tensor, convert<half>(other_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::ge(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_geValue(context->thc_state, result_->tensor, self_->tensor, convert<half>(other_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::s_ge_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ge_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",2, false);
    THCudaHalfTensor_geTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAHalfType::s_ge(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ge(self, Scalar(other));
    }
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",2, false);
    THCudaHalfTensor_geTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAHalfType::ge_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_geValueT(context->thc_state, self_->tensor, self_->tensor, convert<half>(other_));
    return self;
}
Tensor & CUDAHalfType::s_ge_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ge_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",3, false);
    THCudaHalfTensor_geTensorT(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAHalfType::eq_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_eqValue(context->thc_state, result_->tensor, self_->tensor, convert<half>(other_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::eq(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_eqValue(context->thc_state, result_->tensor, self_->tensor, convert<half>(other_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::s_eq_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->eq_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",2, false);
    THCudaHalfTensor_eqTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAHalfType::s_eq(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->eq(self, Scalar(other));
    }
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",2, false);
    THCudaHalfTensor_eqTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAHalfType::eq_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_eqValueT(context->thc_state, self_->tensor, self_->tensor, convert<half>(other_));
    return self;
}
Tensor & CUDAHalfType::s_eq_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->eq_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",3, false);
    THCudaHalfTensor_eqTensorT(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAHalfType::ne_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_neValue(context->thc_state, result_->tensor, self_->tensor, convert<half>(other_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::ne(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_neValue(context->thc_state, result_->tensor, self_->tensor, convert<half>(other_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::s_ne_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ne_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",2, false);
    THCudaHalfTensor_neTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAHalfType::s_ne(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ne(self, Scalar(other));
    }
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",2, false);
    THCudaHalfTensor_neTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAHalfType::ne_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_neValueT(context->thc_state, self_->tensor, self_->tensor, convert<half>(other_));
    return self;
}
Tensor & CUDAHalfType::s_ne_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ne_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",3, false);
    THCudaHalfTensor_neTensorT(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
std::tuple<Tensor &,Tensor &> CUDAHalfType::min_out(Tensor & min, Tensor & min_indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto min_ = checked_cast_tensor<CUDAHalfTensor>(min.pImpl,"min",0, false);
    auto min_indices_ = checked_cast_tensor<CUDALongTensor>(min_indices.pImpl,"min_indices",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaHalfTensor_min(context->thc_state, min_->tensor, min_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    min_->maybeScalar(maybe_scalar);
    min_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(min, min_indices);
}
std::tuple<Tensor,Tensor> CUDAHalfType::min(const Tensor & self, int64_t dim, bool keepdim) const {
    auto min_ = new CUDAHalfTensor(context);
    auto min = Tensor(min_, false);
    auto min_indices_ = new CUDALongTensor(context);
    auto min_indices = Tensor(min_indices_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaHalfTensor_min(context->thc_state, min_->tensor, min_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    min_->maybeScalar(maybe_scalar);
    min_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(min, min_indices);
}
Tensor & CUDAHalfType::s_min_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",2, false);
    THCudaHalfTensor_cmin(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAHalfType::s_min(const Tensor & self, const Tensor & other) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",2, false);
    THCudaHalfTensor_cmin(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAHalfType::min(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<Half>(THCudaHalfTensor_minall(context->thc_state, self_->tensor)));
}
std::tuple<Tensor &,Tensor &> CUDAHalfType::max_out(Tensor & max, Tensor & max_indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto max_ = checked_cast_tensor<CUDAHalfTensor>(max.pImpl,"max",0, false);
    auto max_indices_ = checked_cast_tensor<CUDALongTensor>(max_indices.pImpl,"max_indices",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaHalfTensor_max(context->thc_state, max_->tensor, max_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    max_->maybeScalar(maybe_scalar);
    max_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(max, max_indices);
}
std::tuple<Tensor,Tensor> CUDAHalfType::max(const Tensor & self, int64_t dim, bool keepdim) const {
    auto max_ = new CUDAHalfTensor(context);
    auto max = Tensor(max_, false);
    auto max_indices_ = new CUDALongTensor(context);
    auto max_indices = Tensor(max_indices_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaHalfTensor_max(context->thc_state, max_->tensor, max_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    max_->maybeScalar(maybe_scalar);
    max_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(max, max_indices);
}
Tensor & CUDAHalfType::s_max_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",2, false);
    THCudaHalfTensor_cmax(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAHalfType::s_max(const Tensor & self, const Tensor & other) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",2, false);
    THCudaHalfTensor_cmax(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAHalfType::max(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<Half>(THCudaHalfTensor_maxall(context->thc_state, self_->tensor)));
}
std::tuple<Tensor &,Tensor &> CUDAHalfType::mode_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = checked_cast_tensor<CUDAHalfTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaHalfTensor_mode(context->thc_state, values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CUDAHalfType::mode(const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = new CUDAHalfTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CUDALongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaHalfTensor_mode(context->thc_state, values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
std::tuple<Tensor &,Tensor &> CUDAHalfType::median_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = checked_cast_tensor<CUDAHalfTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaHalfTensor_median(context->thc_state, values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CUDAHalfType::median(const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = new CUDAHalfTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CUDALongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaHalfTensor_median(context->thc_state, values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
Tensor CUDAHalfType::median(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<Half>(THCudaHalfTensor_medianall(context->thc_state, self_->tensor)));
}
std::tuple<Tensor &,Tensor &> CUDAHalfType::sort_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool descending) const {
    auto values_ = checked_cast_tensor<CUDAHalfTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaHalfTensor_sort(context->thc_state, values_->tensor, indices_->tensor, self_->tensor, dim, descending);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CUDAHalfType::sort(const Tensor & self, int64_t dim, bool descending) const {
    auto values_ = new CUDAHalfTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CUDALongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaHalfTensor_sort(context->thc_state, values_->tensor, indices_->tensor, self_->tensor, dim, descending);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
std::tuple<Tensor &,Tensor &> CUDAHalfType::topk_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) const {
    auto values_ = checked_cast_tensor<CUDAHalfTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaHalfTensor_topk(context->thc_state, values_->tensor, indices_->tensor, self_->tensor, k, dim, largest, sorted);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CUDAHalfType::topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) const {
    auto values_ = new CUDAHalfTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CUDALongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaHalfTensor_topk(context->thc_state, values_->tensor, indices_->tensor, self_->tensor, k, dim, largest, sorted);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
int64_t CUDAHalfType::get_device(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    return static_cast<int64_t>(THCudaHalfTensor_getDevice(context->thc_state, self_->tensor));
}
Tensor & CUDAHalfType::_abs_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_abs(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::_abs(const Tensor & self) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_abs(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::sigmoid_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_sigmoid(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAHalfType::sigmoid_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_sigmoid(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::sigmoid(const Tensor & self) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_sigmoid(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::_log_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_log(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::_log(const Tensor & self) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_log(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::log10_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_log10(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAHalfType::log10_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_log10(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::log10(const Tensor & self) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_log10(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::log1p_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_log1p(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAHalfType::log1p_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_log1p(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::log1p(const Tensor & self) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_log1p(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::log2_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_log2(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAHalfType::log2_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_log2(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::log2(const Tensor & self) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_log2(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::lgamma_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_lgamma(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::lgamma(const Tensor & self) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_lgamma(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::lgamma_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_lgamma(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAHalfType::digamma_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_digamma(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::digamma(const Tensor & self) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_digamma(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::digamma_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_digamma(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAHalfType::polygamma_out(Tensor & result, int64_t n, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    THCudaHalfTensor_polygamma(context->thc_state, result_->tensor, n, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::polygamma(int64_t n, const Tensor & self) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    THCudaHalfTensor_polygamma(context->thc_state, result_->tensor, n, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::polygamma_(Tensor & self, int64_t n) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_polygamma(context->thc_state, self_->tensor, n, self_->tensor);
    return self;
}
Tensor & CUDAHalfType::_exp_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_exp(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::_exp(const Tensor & self) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_exp(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::expm1_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_expm1(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAHalfType::expm1_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_expm1(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::expm1(const Tensor & self) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_expm1(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::_cos_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_cos(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::_cos(const Tensor & self) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_cos(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::acos_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_acos(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAHalfType::acos_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_acos(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::acos(const Tensor & self) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_acos(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::cosh_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_cosh(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAHalfType::cosh_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_cosh(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::cosh(const Tensor & self) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_cosh(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::_sin_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_sin(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::_sin(const Tensor & self) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_sin(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::asin_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_asin(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAHalfType::asin_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_asin(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::asin(const Tensor & self) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_asin(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::sinh_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_sinh(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAHalfType::sinh_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_sinh(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::sinh(const Tensor & self) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_sinh(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::tan_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_tan(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAHalfType::tan_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_tan(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::tan(const Tensor & self) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_tan(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::atan_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_atan(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAHalfType::atan_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_atan(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::atan(const Tensor & self) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_atan(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::tanh_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_tanh(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAHalfType::tanh_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_tanh(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::tanh(const Tensor & self) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_tanh(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::erf_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_erf(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAHalfType::erf_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_erf(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::erf(const Tensor & self) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_erf(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::erfinv_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_erfinv(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAHalfType::erfinv_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_erfinv(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::erfinv(const Tensor & self) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_erfinv(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::_sqrt_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_sqrt(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::_sqrt(const Tensor & self) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_sqrt(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::rsqrt_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_rsqrt(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAHalfType::rsqrt_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_rsqrt(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::rsqrt(const Tensor & self) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_rsqrt(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::_ceil_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_ceil(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::_ceil(const Tensor & self) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_ceil(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::_floor_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_floor(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::_floor(const Tensor & self) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_floor(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::_round_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_round(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::_round(const Tensor & self) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_round(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::_trunc_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_trunc(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::_trunc(const Tensor & self) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_trunc(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::frac_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_frac(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAHalfType::frac_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_frac(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::frac(const Tensor & self) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_frac(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::mean_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaHalfTensor_mean(context->thc_state, result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CUDAHalfType::mean(const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaHalfTensor_mean(context->thc_state, result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CUDAHalfType::mean(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<Half>(THCudaHalfTensor_meanall(context->thc_state, self_->tensor)));
}
Tensor & CUDAHalfType::var_out(Tensor & result, const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaHalfTensor_var(context->thc_state, result_->tensor, self_->tensor, dim, (unbiased) ? 0 : 1, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CUDAHalfType::var(const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaHalfTensor_var(context->thc_state, result_->tensor, self_->tensor, dim, (unbiased) ? 0 : 1, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CUDAHalfType::var(const Tensor & self, bool unbiased) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<Half>(THCudaHalfTensor_varall(context->thc_state, self_->tensor, (unbiased) ? 0 : 1)));
}
Tensor & CUDAHalfType::std_out(Tensor & result, const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaHalfTensor_std(context->thc_state, result_->tensor, self_->tensor, dim, (unbiased) ? 0 : 1, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CUDAHalfType::std(const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaHalfTensor_std(context->thc_state, result_->tensor, self_->tensor, dim, (unbiased) ? 0 : 1, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CUDAHalfType::std(const Tensor & self, bool unbiased) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<Half>(THCudaHalfTensor_stdall(context->thc_state, self_->tensor, (unbiased) ? 0 : 1)));
}
Tensor & CUDAHalfType::norm_out(Tensor & result, const Tensor & self, Scalar p, int64_t dim, bool keepdim) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto p_ = p.toHalf();
    dim = maybe_wrap_dim(dim, self_);
    THCudaHalfTensor_norm(context->thc_state, result_->tensor, self_->tensor, convert<half>(p_), dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CUDAHalfType::norm(const Tensor & self, Scalar p, int64_t dim, bool keepdim) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto p_ = p.toHalf();
    dim = maybe_wrap_dim(dim, self_);
    THCudaHalfTensor_norm(context->thc_state, result_->tensor, self_->tensor, convert<half>(p_), dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CUDAHalfType::norm(const Tensor & self, Scalar p) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto p_ = p.toHalf();
    // norm(value) for a sparse tensor returns a DENSE 0-dim tensor
    if (self.is_sparse()) {
      auto result = THCudaHalfTensor_normall(context->thc_state,  self_->tensor, convert<half>(p_));
      return toBackend(toDense(backend())).tensor({}).fill_(result);
    }
    // aten_custom_call is followed by the generated call to normall
    return scalarTensor(convert<Half>(THCudaHalfTensor_normall(context->thc_state, self_->tensor, convert<half>(p_))));
}
Tensor & CUDAHalfType::renorm_out(Tensor & result, const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto p_ = p.toHalf();
    dim = maybe_wrap_dim(dim, self_);
    auto maxnorm_ = maxnorm.toHalf();
    THCudaHalfTensor_renorm(context->thc_state, result_->tensor, self_->tensor, convert<half>(p_), dim, convert<half>(maxnorm_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::renorm(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto p_ = p.toHalf();
    dim = maybe_wrap_dim(dim, self_);
    auto maxnorm_ = maxnorm.toHalf();
    THCudaHalfTensor_renorm(context->thc_state, result_->tensor, self_->tensor, convert<half>(p_), dim, convert<half>(maxnorm_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::renorm_(Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto p_ = p.toHalf();
    dim = maybe_wrap_dim(dim, self_);
    auto maxnorm_ = maxnorm.toHalf();
    THCudaHalfTensor_renorm(context->thc_state, self_->tensor, self_->tensor, convert<half>(p_), dim, convert<half>(maxnorm_));
    return self;
}
Tensor CUDAHalfType::s_dist(const Tensor & self, const Tensor & other, Scalar p) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",2, false);
    auto p_ = p.toHalf();
    return scalarTensor(convert<Half>(THCudaHalfTensor_dist(context->thc_state, self_->tensor, other_->tensor, convert<half>(p_))));
}
Tensor & CUDAHalfType::reciprocal_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_cinv(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::reciprocal(const Tensor & self) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_cinv(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::reciprocal_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_cinv(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAHalfType::neg_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_neg(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::neg(const Tensor & self) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_neg(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::neg_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_neg(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAHalfType::s_atan2_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",2, false);
    THCudaHalfTensor_atan2(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAHalfType::s_atan2(const Tensor & self, const Tensor & other) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",2, false);
    THCudaHalfTensor_atan2(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAHalfType::s_atan2_(Tensor & self, const Tensor & other) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",3, false);
    THCudaHalfTensor_atan2(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAHalfType::pow_out(Tensor & result, const Tensor & self, Scalar exponent) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toHalf();
    THCudaHalfTensor_pow(context->thc_state, result_->tensor, self_->tensor, convert<half>(exponent_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::pow(const Tensor & self, Scalar exponent) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toHalf();
    THCudaHalfTensor_pow(context->thc_state, result_->tensor, self_->tensor, convert<half>(exponent_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::s_pow_out(Tensor & result, const Tensor & self, const Tensor & exponent) const {
    if (exponent.dim() == 0) {
        return static_cast<const Type*>(this)->pow_out(result, self, Scalar(exponent));
    }
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto exponent_ = checked_cast_tensor<CUDAHalfTensor>(exponent.pImpl,"exponent",2, false);
    THCudaHalfTensor_cpow(context->thc_state, result_->tensor, self_->tensor, exponent_->tensor);
    result_->maybeScalar(self_->isScalar() && exponent_->isScalar());
    return result;
}
Tensor CUDAHalfType::s_pow(const Tensor & self, const Tensor & exponent) const {
    if (exponent.dim() == 0) {
        return static_cast<const Type*>(this)->pow(self, Scalar(exponent));
    }
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto exponent_ = checked_cast_tensor<CUDAHalfTensor>(exponent.pImpl,"exponent",2, false);
    THCudaHalfTensor_cpow(context->thc_state, result_->tensor, self_->tensor, exponent_->tensor);
    result_->maybeScalar(self_->isScalar() && exponent_->isScalar());
    return result;
}
Tensor & CUDAHalfType::pow_out(Tensor & result, Scalar base, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto base_ = base.toHalf();
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    THCudaHalfTensor_tpow(context->thc_state, result_->tensor, convert<half>(base_), self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::pow(Scalar base, const Tensor & self) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto base_ = base.toHalf();
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    THCudaHalfTensor_tpow(context->thc_state, result_->tensor, convert<half>(base_), self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::pow_(Tensor & self, Scalar exponent) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toHalf();
    THCudaHalfTensor_pow(context->thc_state, self_->tensor, self_->tensor, convert<half>(exponent_));
    return self;
}
Tensor & CUDAHalfType::s_pow_(Tensor & self, const Tensor & exponent) const {
    if (exponent.dim() == 0) {
        return static_cast<const Type*>(this)->pow_(self, Scalar(exponent));
    }
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto exponent_ = checked_cast_tensor<CUDAHalfTensor>(exponent.pImpl,"exponent",3, false);
    THCudaHalfTensor_cpow(context->thc_state, self_->tensor, self_->tensor, exponent_->tensor);
    return self;
}
Tensor & CUDAHalfType::s_lerp_out(Tensor & result, const Tensor & self, const Tensor & end, Scalar weight) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto end_ = checked_cast_tensor<CUDAHalfTensor>(end.pImpl,"end",2, false);
    auto weight_ = weight.toHalf();
    THCudaHalfTensor_lerp(context->thc_state, result_->tensor, self_->tensor, end_->tensor, convert<half>(weight_));
    result_->maybeScalar(self_->isScalar() && end_->isScalar());
    return result;
}
Tensor CUDAHalfType::s_lerp(const Tensor & self, const Tensor & end, Scalar weight) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto end_ = checked_cast_tensor<CUDAHalfTensor>(end.pImpl,"end",2, false);
    auto weight_ = weight.toHalf();
    THCudaHalfTensor_lerp(context->thc_state, result_->tensor, self_->tensor, end_->tensor, convert<half>(weight_));
    result_->maybeScalar(self_->isScalar() && end_->isScalar());
    return result;
}
Tensor & CUDAHalfType::s_lerp_(Tensor & self, const Tensor & end, Scalar weight) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto end_ = checked_cast_tensor<CUDAHalfTensor>(end.pImpl,"end",3, false);
    auto weight_ = weight.toHalf();
    THCudaHalfTensor_lerp(context->thc_state, self_->tensor, self_->tensor, end_->tensor, convert<half>(weight_));
    return self;
}
Tensor & CUDAHalfType::zero_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_zero(context->thc_state, self_->tensor);
    return self;
}
Tensor CUDAHalfType::_sumall(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<Half>(THCudaHalfTensor_sumall(context->thc_state, self_->tensor)));
}
Tensor & CUDAHalfType::_sum_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaHalfTensor_sum(context->thc_state, result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CUDAHalfType::_sum(const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaHalfTensor_sum(context->thc_state, result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CUDAHalfType::_prodall(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<Half>(THCudaHalfTensor_prodall(context->thc_state, self_->tensor)));
}
Tensor & CUDAHalfType::_prod_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaHalfTensor_prod(context->thc_state, result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CUDAHalfType::_prod(const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaHalfTensor_prod(context->thc_state, result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor & CUDAHalfType::cumsum_out(Tensor & result, const Tensor & self, int64_t dim) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaHalfTensor_cumsum(context->thc_state, result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::cumsum(const Tensor & self, int64_t dim) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaHalfTensor_cumsum(context->thc_state, result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::cumprod_out(Tensor & result, const Tensor & self, int64_t dim) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaHalfTensor_cumprod(context->thc_state, result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::cumprod(const Tensor & self, int64_t dim) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaHalfTensor_cumprod(context->thc_state, result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::sign_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_sign(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::sign(const Tensor & self) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_sign(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::sign_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_sign(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor CUDAHalfType::trace(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<Half>(THCudaHalfTensor_trace(context->thc_state, self_->tensor)));
}
Tensor & CUDAHalfType::add_out(Tensor & result, const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    auto alpha_ = alpha.toHalf();
    THCudaHalfTensor_add_scaled(context->thc_state, result_->tensor, self_->tensor, convert<half>(other_), convert<half>(alpha_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::add(const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    auto alpha_ = alpha.toHalf();
    THCudaHalfTensor_add_scaled(context->thc_state, result_->tensor, self_->tensor, convert<half>(other_), convert<half>(alpha_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::s_add_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if(other.type().is_sparse()) {
        return static_cast<const Type*>(this)->add_out(result, self, SparseTensor(other), alpha);
    }
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toHalf();
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",3, false);
    THCudaHalfTensor_cadd(context->thc_state, result_->tensor, self_->tensor, convert<half>(alpha_), other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAHalfType::s_add(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if(other.type().is_sparse()) {
        return static_cast<const Type*>(this)->add(self, SparseTensor(other), alpha);
    }
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add(self, Scalar(other), alpha);
    }
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toHalf();
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",3, false);
    THCudaHalfTensor_cadd(context->thc_state, result_->tensor, self_->tensor, convert<half>(alpha_), other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAHalfType::add_(Tensor & self, Scalar other, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    auto alpha_ = alpha.toHalf();
    THCudaHalfTensor_add_scaled(context->thc_state, self_->tensor, self_->tensor, convert<half>(other_), convert<half>(alpha_));
    return self;
}
Tensor & CUDAHalfType::s_add_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if(other.type().is_sparse()) {
        return static_cast<const Type*>(this)->add_(self, SparseTensor(other), alpha);
    }
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toHalf();
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",4, false);
    THCudaHalfTensor_cadd(context->thc_state, self_->tensor, self_->tensor, convert<half>(alpha_), other_->tensor);
    return self;
}
Tensor & CUDAHalfType::sub_out(Tensor & result, const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    auto alpha_ = alpha.toHalf();
    THCudaHalfTensor_sub_scaled(context->thc_state, result_->tensor, self_->tensor, convert<half>(other_), convert<half>(alpha_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::sub(const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    auto alpha_ = alpha.toHalf();
    THCudaHalfTensor_sub_scaled(context->thc_state, result_->tensor, self_->tensor, convert<half>(other_), convert<half>(alpha_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::s_sub_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toHalf();
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",3, false);
    THCudaHalfTensor_csub(context->thc_state, result_->tensor, self_->tensor, convert<half>(alpha_), other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAHalfType::s_sub(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub(self, Scalar(other), alpha);
    }
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toHalf();
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",3, false);
    THCudaHalfTensor_csub(context->thc_state, result_->tensor, self_->tensor, convert<half>(alpha_), other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAHalfType::sub_(Tensor & self, Scalar other, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    auto alpha_ = alpha.toHalf();
    THCudaHalfTensor_sub_scaled(context->thc_state, self_->tensor, self_->tensor, convert<half>(other_), convert<half>(alpha_));
    return self;
}
Tensor & CUDAHalfType::s_sub_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toHalf();
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",4, false);
    THCudaHalfTensor_csub(context->thc_state, self_->tensor, self_->tensor, convert<half>(alpha_), other_->tensor);
    return self;
}
Tensor & CUDAHalfType::mul_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_mul(context->thc_state, result_->tensor, self_->tensor, convert<half>(other_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::mul(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_mul(context->thc_state, result_->tensor, self_->tensor, convert<half>(other_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::s_mul_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",2, false);
    THCudaHalfTensor_cmul(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAHalfType::s_mul(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul(self, Scalar(other));
    }
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",2, false);
    THCudaHalfTensor_cmul(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAHalfType::mul_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_mul(context->thc_state, self_->tensor, self_->tensor, convert<half>(other_));
    return self;
}
Tensor & CUDAHalfType::s_mul_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",3, false);
    THCudaHalfTensor_cmul(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAHalfType::div_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_div(context->thc_state, result_->tensor, self_->tensor, convert<half>(other_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::div(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_div(context->thc_state, result_->tensor, self_->tensor, convert<half>(other_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::s_div_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->div_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",2, false);
    THCudaHalfTensor_cdiv(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAHalfType::s_div(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->div(self, Scalar(other));
    }
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",2, false);
    THCudaHalfTensor_cdiv(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAHalfType::div_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_div(context->thc_state, self_->tensor, self_->tensor, convert<half>(other_));
    return self;
}
Tensor & CUDAHalfType::s_div_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->div_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",3, false);
    THCudaHalfTensor_cdiv(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAHalfType::fmod_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_fmod(context->thc_state, result_->tensor, self_->tensor, convert<half>(other_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::fmod(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_fmod(context->thc_state, result_->tensor, self_->tensor, convert<half>(other_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::s_fmod_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->fmod_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",2, false);
    THCudaHalfTensor_cfmod(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAHalfType::s_fmod(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->fmod(self, Scalar(other));
    }
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",2, false);
    THCudaHalfTensor_cfmod(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAHalfType::fmod_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_fmod(context->thc_state, self_->tensor, self_->tensor, convert<half>(other_));
    return self;
}
Tensor & CUDAHalfType::s_fmod_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->fmod_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",3, false);
    THCudaHalfTensor_cfmod(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAHalfType::remainder_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_remainder(context->thc_state, result_->tensor, self_->tensor, convert<half>(other_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::remainder(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_remainder(context->thc_state, result_->tensor, self_->tensor, convert<half>(other_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::s_remainder_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->remainder_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",2, false);
    THCudaHalfTensor_cremainder(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAHalfType::s_remainder(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->remainder(self, Scalar(other));
    }
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",2, false);
    THCudaHalfTensor_cremainder(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAHalfType::remainder_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toHalf();
    THCudaHalfTensor_remainder(context->thc_state, self_->tensor, self_->tensor, convert<half>(other_));
    return self;
}
Tensor & CUDAHalfType::s_remainder_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->remainder_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",3, false);
    THCudaHalfTensor_cremainder(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAHalfType::clamp_out(Tensor & result, const Tensor & self, Scalar min, Scalar max) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toHalf();
    auto max_ = max.toHalf();
    THCudaHalfTensor_clamp(context->thc_state, result_->tensor, self_->tensor, convert<half>(min_), convert<half>(max_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::clamp(const Tensor & self, Scalar min, Scalar max) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toHalf();
    auto max_ = max.toHalf();
    THCudaHalfTensor_clamp(context->thc_state, result_->tensor, self_->tensor, convert<half>(min_), convert<half>(max_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::clamp_(Tensor & self, Scalar min, Scalar max) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toHalf();
    auto max_ = max.toHalf();
    THCudaHalfTensor_clamp(context->thc_state, self_->tensor, self_->tensor, convert<half>(min_), convert<half>(max_));
    return self;
}
Tensor & CUDAHalfType::clamp_min_out(Tensor & result, const Tensor & self, Scalar min) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toHalf();
    THCudaHalfTensor_cmaxValue(context->thc_state, result_->tensor, self_->tensor, convert<half>(min_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::clamp_min(const Tensor & self, Scalar min) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toHalf();
    THCudaHalfTensor_cmaxValue(context->thc_state, result_->tensor, self_->tensor, convert<half>(min_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::clamp_min_(Tensor & self, Scalar min) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toHalf();
    THCudaHalfTensor_cmaxValue(context->thc_state, self_->tensor, self_->tensor, convert<half>(min_));
    return self;
}
Tensor & CUDAHalfType::clamp_max_out(Tensor & result, const Tensor & self, Scalar max) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto max_ = max.toHalf();
    THCudaHalfTensor_cminValue(context->thc_state, result_->tensor, self_->tensor, convert<half>(max_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::clamp_max(const Tensor & self, Scalar max) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto max_ = max.toHalf();
    THCudaHalfTensor_cminValue(context->thc_state, result_->tensor, self_->tensor, convert<half>(max_));
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::clamp_max_(Tensor & self, Scalar max) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto max_ = max.toHalf();
    THCudaHalfTensor_cminValue(context->thc_state, self_->tensor, self_->tensor, convert<half>(max_));
    return self;
}
Tensor CUDAHalfType::_dot(const Tensor & self, const Tensor & tensor) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto tensor_ = checked_cast_tensor<CUDAHalfTensor>(tensor.pImpl,"tensor",2, false);
    return scalarTensor(convert<Half>(THCudaHalfTensor_dot(context->thc_state, self_->tensor, tensor_->tensor)));
}
Tensor & CUDAHalfType::tril_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_tril(context->thc_state, result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::tril(const Tensor & self, int64_t diagonal) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_tril(context->thc_state, result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::tril_(Tensor & self, int64_t diagonal) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_tril(context->thc_state, self_->tensor, self_->tensor, diagonal);
    return self;
}
Tensor & CUDAHalfType::triu_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_triu(context->thc_state, result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::triu(const Tensor & self, int64_t diagonal) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_triu(context->thc_state, result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::triu_(Tensor & self, int64_t diagonal) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    THCudaHalfTensor_triu(context->thc_state, self_->tensor, self_->tensor, diagonal);
    return self;
}
Tensor & CUDAHalfType::cross_out(Tensor & result, const Tensor & self, const Tensor & other, int64_t dim) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",2, false);
    THCudaHalfTensor_cross(context->thc_state, result_->tensor, self_->tensor, other_->tensor, dim);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAHalfType::cross(const Tensor & self, const Tensor & other, int64_t dim) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAHalfTensor>(other.pImpl,"other",2, false);
    THCudaHalfTensor_cross(context->thc_state, result_->tensor, self_->tensor, other_->tensor, dim);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAHalfType::diag_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    if (self_->isScalar()) {
      throw std::runtime_error("Input must be 1-d or 2-d");
    }
    THCudaHalfTensor_diag(context->thc_state, result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::diag(const Tensor & self, int64_t diagonal) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    if (self_->isScalar()) {
      throw std::runtime_error("Input must be 1-d or 2-d");
    }
    THCudaHalfTensor_diag(context->thc_state, result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::s_addmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    if(mat1.type().is_sparse()) {
        return static_cast<const Type*>(this)->addmm_out(result, self, SparseTensor(mat1), mat2, beta, alpha);
    }
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toHalf();
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toHalf();
    auto mat1_ = checked_cast_tensor<CUDAHalfTensor>(mat1.pImpl,"mat1",4, false);
    auto mat2_ = checked_cast_tensor<CUDAHalfTensor>(mat2.pImpl,"mat2",5, false);
    THCudaHalfTensor_addmm(context->thc_state, result_->tensor, convert<half>(beta_), self_->tensor, convert<half>(alpha_), mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CUDAHalfType::s_addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    if(mat1.type().is_sparse()) {
        return static_cast<const Type*>(this)->addmm(self, SparseTensor(mat1), mat2, beta, alpha);
    }
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toHalf();
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toHalf();
    auto mat1_ = checked_cast_tensor<CUDAHalfTensor>(mat1.pImpl,"mat1",4, false);
    auto mat2_ = checked_cast_tensor<CUDAHalfTensor>(mat2.pImpl,"mat2",5, false);
    THCudaHalfTensor_addmm(context->thc_state, result_->tensor, convert<half>(beta_), self_->tensor, convert<half>(alpha_), mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CUDAHalfType::addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    if(mat1.type().is_sparse()) {
        return static_cast<const Type*>(this)->addmm_(self, SparseTensor(mat1), mat2, beta, alpha);
    }
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toHalf();
    auto alpha_ = alpha.toHalf();
    auto mat1_ = checked_cast_tensor<CUDAHalfTensor>(mat1.pImpl,"mat1",5, false);
    auto mat2_ = checked_cast_tensor<CUDAHalfTensor>(mat2.pImpl,"mat2",6, false);
    THCudaHalfTensor_addmm(context->thc_state, self_->tensor, convert<half>(beta_), self_->tensor, convert<half>(alpha_), mat1_->tensor, mat2_->tensor);
    return self;
}
Tensor & CUDAHalfType::s__addmv_out(Tensor & result, const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toHalf();
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toHalf();
    auto mat_ = checked_cast_tensor<CUDAHalfTensor>(mat.pImpl,"mat",4, false);
    auto vec_ = checked_cast_tensor<CUDAHalfTensor>(vec.pImpl,"vec",5, false);
    THCudaHalfTensor_addmv(context->thc_state, result_->tensor, convert<half>(beta_), self_->tensor, convert<half>(alpha_), mat_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && mat_->isScalar() && vec_->isScalar());
    return result;
}
Tensor CUDAHalfType::s__addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toHalf();
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toHalf();
    auto mat_ = checked_cast_tensor<CUDAHalfTensor>(mat.pImpl,"mat",4, false);
    auto vec_ = checked_cast_tensor<CUDAHalfTensor>(vec.pImpl,"vec",5, false);
    THCudaHalfTensor_addmv(context->thc_state, result_->tensor, convert<half>(beta_), self_->tensor, convert<half>(alpha_), mat_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && mat_->isScalar() && vec_->isScalar());
    return result;
}
Tensor & CUDAHalfType::_addmv_(Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toHalf();
    auto alpha_ = alpha.toHalf();
    auto mat_ = checked_cast_tensor<CUDAHalfTensor>(mat.pImpl,"mat",5, false);
    auto vec_ = checked_cast_tensor<CUDAHalfTensor>(vec.pImpl,"vec",6, false);
    THCudaHalfTensor_addmv(context->thc_state, self_->tensor, convert<half>(beta_), self_->tensor, convert<half>(alpha_), mat_->tensor, vec_->tensor);
    return self;
}
Tensor & CUDAHalfType::s__addr_out(Tensor & result, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toHalf();
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toHalf();
    auto vec1_ = checked_cast_tensor<CUDAHalfTensor>(vec1.pImpl,"vec1",4, false);
    auto vec2_ = checked_cast_tensor<CUDAHalfTensor>(vec2.pImpl,"vec2",5, false);
    THCudaHalfTensor_addr(context->thc_state, result_->tensor, convert<half>(beta_), self_->tensor, convert<half>(alpha_), vec1_->tensor, vec2_->tensor);
    result_->maybeScalar(self_->isScalar() && vec1_->isScalar() && vec2_->isScalar());
    return result;
}
Tensor CUDAHalfType::s__addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toHalf();
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toHalf();
    auto vec1_ = checked_cast_tensor<CUDAHalfTensor>(vec1.pImpl,"vec1",4, false);
    auto vec2_ = checked_cast_tensor<CUDAHalfTensor>(vec2.pImpl,"vec2",5, false);
    THCudaHalfTensor_addr(context->thc_state, result_->tensor, convert<half>(beta_), self_->tensor, convert<half>(alpha_), vec1_->tensor, vec2_->tensor);
    result_->maybeScalar(self_->isScalar() && vec1_->isScalar() && vec2_->isScalar());
    return result;
}
Tensor & CUDAHalfType::_addr_(Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toHalf();
    auto alpha_ = alpha.toHalf();
    auto vec1_ = checked_cast_tensor<CUDAHalfTensor>(vec1.pImpl,"vec1",5, false);
    auto vec2_ = checked_cast_tensor<CUDAHalfTensor>(vec2.pImpl,"vec2",6, false);
    THCudaHalfTensor_addr(context->thc_state, self_->tensor, convert<half>(beta_), self_->tensor, convert<half>(alpha_), vec1_->tensor, vec2_->tensor);
    return self;
}
Tensor & CUDAHalfType::_ger_out(Tensor & result, const Tensor & self, const Tensor & vec2) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.dim() == 0 ? 1 : self.size(0),vec2.dim() == 0 ? 1 : vec2.size(0) });
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto vec2_ = checked_cast_tensor<CUDAHalfTensor>(vec2.pImpl,"vec2",2, false);
    THCudaHalfTensor_addr(context->thc_state, result_->tensor, convert<half,double>(0), result_->tensor, convert<half,double>(1), self_->tensor, vec2_->tensor);
    result_->maybeScalar(false);
    return result;
}
Tensor CUDAHalfType::_ger(const Tensor & self, const Tensor & vec2) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.dim() == 0 ? 1 : self.size(0),vec2.dim() == 0 ? 1 : vec2.size(0) });
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto vec2_ = checked_cast_tensor<CUDAHalfTensor>(vec2.pImpl,"vec2",2, false);
    THCudaHalfTensor_addr(context->thc_state, result_->tensor, convert<half,double>(0), result_->tensor, convert<half,double>(1), self_->tensor, vec2_->tensor);
    result_->maybeScalar(false);
    return result;
}
Tensor & CUDAHalfType::_mv_out(Tensor & result, const Tensor & self, const Tensor & vec) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.size(0) });
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto vec_ = checked_cast_tensor<CUDAHalfTensor>(vec.pImpl,"vec",2, false);
    THCudaHalfTensor_addmv(context->thc_state, result_->tensor, convert<half,double>(0), result_->tensor, convert<half,double>(1), self_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && vec_->isScalar());
    return result;
}
Tensor CUDAHalfType::_mv(const Tensor & self, const Tensor & vec) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.size(0) });
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto vec_ = checked_cast_tensor<CUDAHalfTensor>(vec.pImpl,"vec",2, false);
    THCudaHalfTensor_addmv(context->thc_state, result_->tensor, convert<half,double>(0), result_->tensor, convert<half,double>(1), self_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && vec_->isScalar());
    return result;
}
Tensor & CUDAHalfType::_mm_out(Tensor & result, const Tensor & self, const Tensor & mat2) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.size(0),mat2.size(1) });
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CUDAHalfTensor>(mat2.pImpl,"mat2",2, false);
    THCudaHalfTensor_addmm(context->thc_state, result_->tensor, convert<half,double>(0), result_->tensor, convert<half,double>(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CUDAHalfType::_mm(const Tensor & self, const Tensor & mat2) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.size(0),mat2.size(1) });
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CUDAHalfTensor>(mat2.pImpl,"mat2",2, false);
    THCudaHalfTensor_addmm(context->thc_state, result_->tensor, convert<half,double>(0), result_->tensor, convert<half,double>(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CUDAHalfType::bmm_out(Tensor & result, const Tensor & self, const Tensor & mat2) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.size(0),self.size(1),mat2.size(2) });
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CUDAHalfTensor>(mat2.pImpl,"mat2",2, false);
    THCudaHalfTensor_baddbmm(context->thc_state, result_->tensor, convert<half,double>(0), result_->tensor, convert<half,double>(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CUDAHalfType::bmm(const Tensor & self, const Tensor & mat2) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.size(0),self.size(1),mat2.size(2) });
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CUDAHalfTensor>(mat2.pImpl,"mat2",2, false);
    THCudaHalfTensor_baddbmm(context->thc_state, result_->tensor, convert<half,double>(0), result_->tensor, convert<half,double>(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CUDAHalfType::s_addbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toHalf();
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toHalf();
    auto batch1_ = checked_cast_tensor<CUDAHalfTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CUDAHalfTensor>(batch2.pImpl,"batch2",5, false);
    THCudaHalfTensor_addbmm(context->thc_state, result_->tensor, convert<half>(beta_), self_->tensor, convert<half>(alpha_), batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor CUDAHalfType::s_addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toHalf();
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toHalf();
    auto batch1_ = checked_cast_tensor<CUDAHalfTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CUDAHalfTensor>(batch2.pImpl,"batch2",5, false);
    THCudaHalfTensor_addbmm(context->thc_state, result_->tensor, convert<half>(beta_), self_->tensor, convert<half>(alpha_), batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor & CUDAHalfType::addbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toHalf();
    auto alpha_ = alpha.toHalf();
    auto batch1_ = checked_cast_tensor<CUDAHalfTensor>(batch1.pImpl,"batch1",5, false);
    auto batch2_ = checked_cast_tensor<CUDAHalfTensor>(batch2.pImpl,"batch2",6, false);
    THCudaHalfTensor_addbmm(context->thc_state, self_->tensor, convert<half>(beta_), self_->tensor, convert<half>(alpha_), batch1_->tensor, batch2_->tensor);
    return self;
}
Tensor & CUDAHalfType::s_baddbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toHalf();
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toHalf();
    auto batch1_ = checked_cast_tensor<CUDAHalfTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CUDAHalfTensor>(batch2.pImpl,"batch2",5, false);
    THCudaHalfTensor_baddbmm(context->thc_state, result_->tensor, convert<half>(beta_), self_->tensor, convert<half>(alpha_), batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor CUDAHalfType::s_baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toHalf();
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toHalf();
    auto batch1_ = checked_cast_tensor<CUDAHalfTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CUDAHalfTensor>(batch2.pImpl,"batch2",5, false);
    THCudaHalfTensor_baddbmm(context->thc_state, result_->tensor, convert<half>(beta_), self_->tensor, convert<half>(alpha_), batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor & CUDAHalfType::baddbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toHalf();
    auto alpha_ = alpha.toHalf();
    auto batch1_ = checked_cast_tensor<CUDAHalfTensor>(batch1.pImpl,"batch1",5, false);
    auto batch2_ = checked_cast_tensor<CUDAHalfTensor>(batch2.pImpl,"batch2",6, false);
    THCudaHalfTensor_baddbmm(context->thc_state, self_->tensor, convert<half>(beta_), self_->tensor, convert<half>(alpha_), batch1_->tensor, batch2_->tensor);
    return self;
}
Tensor & CUDAHalfType::s_addcmul_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toHalf();
    auto tensor1_ = checked_cast_tensor<CUDAHalfTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CUDAHalfTensor>(tensor2.pImpl,"tensor2",4, false);
    THCudaHalfTensor_addcmul(context->thc_state, result_->tensor, self_->tensor, convert<half>(value_), tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor CUDAHalfType::s_addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toHalf();
    auto tensor1_ = checked_cast_tensor<CUDAHalfTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CUDAHalfTensor>(tensor2.pImpl,"tensor2",4, false);
    THCudaHalfTensor_addcmul(context->thc_state, result_->tensor, self_->tensor, convert<half>(value_), tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor & CUDAHalfType::s_addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toHalf();
    auto tensor1_ = checked_cast_tensor<CUDAHalfTensor>(tensor1.pImpl,"tensor1",4, false);
    auto tensor2_ = checked_cast_tensor<CUDAHalfTensor>(tensor2.pImpl,"tensor2",5, false);
    THCudaHalfTensor_addcmul(context->thc_state, self_->tensor, self_->tensor, convert<half>(value_), tensor1_->tensor, tensor2_->tensor);
    return self;
}
Tensor & CUDAHalfType::s_addcdiv_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toHalf();
    auto tensor1_ = checked_cast_tensor<CUDAHalfTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CUDAHalfTensor>(tensor2.pImpl,"tensor2",4, false);
    THCudaHalfTensor_addcdiv(context->thc_state, result_->tensor, self_->tensor, convert<half>(value_), tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor CUDAHalfType::s_addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toHalf();
    auto tensor1_ = checked_cast_tensor<CUDAHalfTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CUDAHalfTensor>(tensor2.pImpl,"tensor2",4, false);
    THCudaHalfTensor_addcdiv(context->thc_state, result_->tensor, self_->tensor, convert<half>(value_), tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor & CUDAHalfType::s_addcdiv_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toHalf();
    auto tensor1_ = checked_cast_tensor<CUDAHalfTensor>(tensor1.pImpl,"tensor1",4, false);
    auto tensor2_ = checked_cast_tensor<CUDAHalfTensor>(tensor2.pImpl,"tensor2",5, false);
    THCudaHalfTensor_addcdiv(context->thc_state, self_->tensor, self_->tensor, convert<half>(value_), tensor1_->tensor, tensor2_->tensor);
    return self;
}
std::tuple<Tensor &,Tensor &> CUDAHalfType::btrifact_out(Tensor & result, Tensor & pivots, const Tensor & self, bool pivot) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto pivots_ = checked_cast_tensor<CUDAIntTensor>(pivots.pImpl,"pivots",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    THCudaHalfTensor_btrifact(context->thc_state, result_->tensor, pivots_->tensor, NULL, pivot, self_->tensor);
    bool maybe_scalar = self_->isScalar();
    result_->maybeScalar(maybe_scalar);
    pivots_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(result, pivots);
}
std::tuple<Tensor,Tensor> CUDAHalfType::btrifact(const Tensor & self, bool pivot) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto pivots_ = new CUDAIntTensor(context);
    auto pivots = Tensor(pivots_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    THCudaHalfTensor_btrifact(context->thc_state, result_->tensor, pivots_->tensor, NULL, pivot, self_->tensor);
    bool maybe_scalar = self_->isScalar();
    result_->maybeScalar(maybe_scalar);
    pivots_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(result, pivots);
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDAHalfType::btrifact_with_info_out(Tensor & result, Tensor & pivots, Tensor & info, const Tensor & self, bool pivot) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto pivots_ = checked_cast_tensor<CUDAIntTensor>(pivots.pImpl,"pivots",0, false);
    auto info_ = checked_cast_tensor<CUDAIntTensor>(info.pImpl,"info",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    THCudaHalfTensor_btrifact(context->thc_state, result_->tensor, pivots_->tensor, info_->tensor, pivot, self_->tensor);
    bool maybe_scalar = self_->isScalar();
    result_->maybeScalar(maybe_scalar);
    pivots_->maybeScalar(maybe_scalar);
    info_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(result, pivots, info);
}
std::tuple<Tensor,Tensor,Tensor> CUDAHalfType::btrifact_with_info(const Tensor & self, bool pivot) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto pivots_ = new CUDAIntTensor(context);
    auto pivots = Tensor(pivots_, false);
    auto info_ = new CUDAIntTensor(context);
    auto info = Tensor(info_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    THCudaHalfTensor_btrifact(context->thc_state, result_->tensor, pivots_->tensor, info_->tensor, pivot, self_->tensor);
    bool maybe_scalar = self_->isScalar();
    result_->maybeScalar(maybe_scalar);
    pivots_->maybeScalar(maybe_scalar);
    info_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(result, pivots, info);
}
Tensor & CUDAHalfType::btrisolve_out(Tensor & result, const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto LU_data_ = checked_cast_tensor<CUDAHalfTensor>(LU_data.pImpl,"LU_data",2, false);
    auto LU_pivots_ = checked_cast_tensor<CUDAIntTensor>(LU_pivots.pImpl,"LU_pivots",3, false);
    THCudaHalfTensor_btrisolve(context->thc_state, result_->tensor, self_->tensor, LU_data_->tensor, LU_pivots_->tensor);
    result_->maybeScalar(self_->isScalar() && LU_data_->isScalar() && LU_pivots_->isScalar());
    return result;
}
Tensor CUDAHalfType::btrisolve(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto LU_data_ = checked_cast_tensor<CUDAHalfTensor>(LU_data.pImpl,"LU_data",2, false);
    auto LU_pivots_ = checked_cast_tensor<CUDAIntTensor>(LU_pivots.pImpl,"LU_pivots",3, false);
    THCudaHalfTensor_btrisolve(context->thc_state, result_->tensor, self_->tensor, LU_data_->tensor, LU_pivots_->tensor);
    result_->maybeScalar(self_->isScalar() && LU_data_->isScalar() && LU_pivots_->isScalar());
    return result;
}
Tensor & CUDAHalfType::random_(Tensor & self, int64_t from, int64_t to, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    THCudaHalfTensor_clampedRandom(context->thc_state, self_->tensor, from, to);
    return self;
}
Tensor & CUDAHalfType::random_(Tensor & self, int64_t to, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    THCudaHalfTensor_cappedRandom(context->thc_state, self_->tensor, to);
    return self;
}
Tensor & CUDAHalfType::random_(Tensor & self, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    THCudaHalfTensor_random(context->thc_state, self_->tensor);
    return self;
}
Tensor & CUDAHalfType::multinomial_out(Tensor & result, const Tensor & self, int64_t num_samples, bool replacement, Generator * generator) const {
    auto result_ = checked_cast_tensor<CUDALongTensor>(result.pImpl,"result",0, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    THCudaHalfTensor_multinomial(context->thc_state, result_->tensor, self_->tensor, num_samples, replacement);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAHalfType::multinomial(const Tensor & self, int64_t num_samples, bool replacement, Generator * generator) const {
    auto result_ = new CUDALongTensor(context);
    auto result = Tensor(result_, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    THCudaHalfTensor_multinomial(context->thc_state, result_->tensor, self_->tensor, num_samples, replacement);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAHalfType::uniform_(Tensor & self, double from, double to, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    THCudaHalfTensor_uniform(context->thc_state, self_->tensor, from, to);
    return self;
}
Tensor & CUDAHalfType::normal_out(Tensor & output, const Tensor & mean, double std, Generator * generator) const {
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",0, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    auto mean_ = checked_cast_tensor<CUDAHalfTensor>(mean.pImpl,"mean",2, false);
    THCudaHalfTensor_normal_means(context->thc_state, output_->tensor, mean_->tensor, std);
    output_->maybeScalar(mean_->isScalar());
    return output;
}
Tensor CUDAHalfType::normal(const Tensor & mean, double std, Generator * generator) const {
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    auto mean_ = checked_cast_tensor<CUDAHalfTensor>(mean.pImpl,"mean",2, false);
    THCudaHalfTensor_normal_means(context->thc_state, output_->tensor, mean_->tensor, std);
    output_->maybeScalar(mean_->isScalar());
    return output;
}
Tensor & CUDAHalfType::normal_out(Tensor & output, double mean, const Tensor & std, Generator * generator) const {
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",0, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    auto std_ = checked_cast_tensor<CUDAHalfTensor>(std.pImpl,"std",3, false);
    THCudaHalfTensor_normal_stddevs(context->thc_state, output_->tensor, mean, std_->tensor);
    output_->maybeScalar(std_->isScalar());
    return output;
}
Tensor CUDAHalfType::normal(double mean, const Tensor & std, Generator * generator) const {
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    auto std_ = checked_cast_tensor<CUDAHalfTensor>(std.pImpl,"std",3, false);
    THCudaHalfTensor_normal_stddevs(context->thc_state, output_->tensor, mean, std_->tensor);
    output_->maybeScalar(std_->isScalar());
    return output;
}
Tensor & CUDAHalfType::normal_out(Tensor & output, const Tensor & mean, const Tensor & std, Generator * generator) const {
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",0, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    auto mean_ = checked_cast_tensor<CUDAHalfTensor>(mean.pImpl,"mean",2, false);
    auto std_ = checked_cast_tensor<CUDAHalfTensor>(std.pImpl,"std",3, false);
    THCudaHalfTensor_normal_means_stddevs(context->thc_state, output_->tensor, mean_->tensor, std_->tensor);
    output_->maybeScalar(mean_->isScalar() && std_->isScalar());
    return output;
}
Tensor CUDAHalfType::normal(const Tensor & mean, const Tensor & std, Generator * generator) const {
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    auto mean_ = checked_cast_tensor<CUDAHalfTensor>(mean.pImpl,"mean",2, false);
    auto std_ = checked_cast_tensor<CUDAHalfTensor>(std.pImpl,"std",3, false);
    THCudaHalfTensor_normal_means_stddevs(context->thc_state, output_->tensor, mean_->tensor, std_->tensor);
    output_->maybeScalar(mean_->isScalar() && std_->isScalar());
    return output;
}
Tensor & CUDAHalfType::normal_(Tensor & self, double mean, double std, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    THCudaHalfTensor_normal(context->thc_state, self_->tensor, mean, std);
    return self;
}
Tensor & CUDAHalfType::cauchy_(Tensor & self, double median, double sigma, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    THCudaHalfTensor_cauchy(context->thc_state, self_->tensor, median, sigma);
    return self;
}
Tensor & CUDAHalfType::log_normal_(Tensor & self, double mean, double std, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    THCudaHalfTensor_logNormal(context->thc_state, self_->tensor, mean, std);
    return self;
}
Tensor & CUDAHalfType::exponential_(Tensor & self, double lambd, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    THCudaHalfTensor_exponential(context->thc_state, self_->tensor, lambd);
    return self;
}
Tensor & CUDAHalfType::geometric_(Tensor & self, double p, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    THCudaHalfTensor_geometric(context->thc_state, self_->tensor, p);
    return self;
}
Tensor CUDAHalfType::tensor(Storage & storage, int64_t storageOffset, IntList size, IntList stride) const {
    auto storage_ = checked_cast_storage<CUDAHalfStorage>(&storage,"storage",1);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    return Tensor((new CUDAHalfTensor(context, THCudaHalfTensor_newWithStorage(context->thc_state, storage_->storage, storageOffset, size_, stride_)))->maybeScalar(size.size() == 0),false);
}
Tensor CUDAHalfType::tensor(IntList size) const {
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new CUDAHalfTensor(context, THCudaHalfTensor_newWithSize(context->thc_state, size_, NULL)))->maybeScalar(size.size() == 0),false);
}
Tensor CUDAHalfType::tensor(IntList size, IntList stride) const {
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    return Tensor((new CUDAHalfTensor(context, THCudaHalfTensor_newWithSize(context->thc_state, size_, stride_)))->maybeScalar(size.size() == 0),false);
}
Tensor CUDAHalfType::tensor() const {
    return Tensor((new CUDAHalfTensor(context, THCudaHalfTensor_new(context->thc_state))),false);
}
Tensor CUDAHalfType::alias(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    return Tensor((new CUDAHalfTensor(context, THCudaHalfTensor_newWithTensor(context->thc_state, self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor & CUDAHalfType::_copy_ignoring_overlaps_(Tensor & self, const Tensor & src) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto src_ = checked_cast_tensor<CUDAHalfTensor>(src.pImpl,"src",2, false);
    THCudaHalfTensor_copyIgnoringOverlaps(context->thc_state, self_->tensor, src_->tensor);
    return self;
}
Tensor & CUDAHalfType::as_strided_out(Tensor & result, const Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    auto result_ = checked_cast_tensor<CUDAHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    if (storage_offset == -1) {
      storage_offset = self_->tensor->storageOffset;
    }
    THCudaHalfTensor_setStorage(context->thc_state, result_->tensor, self_->tensor->storage, storage_offset, size_, stride_);
    result_->maybeScalar(size.size() == 0);
    return result;
}
Tensor CUDAHalfType::as_strided(const Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    auto result_ = new CUDAHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    if (storage_offset == -1) {
      storage_offset = self_->tensor->storageOffset;
    }
    THCudaHalfTensor_setStorage(context->thc_state, result_->tensor, self_->tensor->storage, storage_offset, size_, stride_);
    result_->maybeScalar(size.size() == 0);
    return result;
}
Tensor & CUDAHalfType::as_strided_(Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    if (storage_offset == -1) {
      storage_offset = self_->tensor->storageOffset;
    }
    THCudaHalfTensor_setStorage(context->thc_state, self_->tensor, self_->tensor->storage, storage_offset, size_, stride_);
    self_->maybeScalar(size.size() == 0);
    return self;
}
Tensor & CUDAHalfType::_cat_out(Tensor & self, TensorList tensors, int64_t dim) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",0, false);
    auto tensors_ = tensor_list_checked_cast<CUDAHalfTensor, Tensor, THCudaHalfTensor>(tensors,"tensors",1);
    THCudaHalfTensor_catArray(context->thc_state, self_->tensor, tensors_.data(), tensors_.size(), dim);
    return self;
}
Tensor CUDAHalfType::_cat(TensorList tensors, int64_t dim) const {
    auto self_ = new CUDAHalfTensor(context);
    auto self = Tensor(self_, false);
    auto tensors_ = tensor_list_checked_cast<CUDAHalfTensor, Tensor, THCudaHalfTensor>(tensors,"tensors",1);
    THCudaHalfTensor_catArray(context->thc_state, self_->tensor, tensors_.data(), tensors_.size(), dim);
    return self;
}
Tensor & CUDAHalfType::binary_cross_entropy_forward_out(Tensor & output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDAHalfTensor>(target.pImpl,"target",2, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",3, true);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",5, false);
    THNN_CudaHalfBCECriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, weight_ ? weight_->tensor : NULL, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor CUDAHalfType::binary_cross_entropy_forward(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDAHalfTensor>(target.pImpl,"target",2, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",3, true);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaHalfBCECriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, weight_ ? weight_->tensor : NULL, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor & CUDAHalfType::binary_cross_entropy_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDAHalfTensor>(target.pImpl,"target",3, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",4, true);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",6, false);
    THNN_CudaHalfBCECriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, weight_ ? weight_->tensor : NULL, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::binary_cross_entropy_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDAHalfTensor>(target.pImpl,"target",3, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",4, true);
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfBCECriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, weight_ ? weight_->tensor : NULL, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAHalfType::kl_div_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDAHalfTensor>(target.pImpl,"target",2, false);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",4, false);
    THNN_CudaHalfDistKLDivCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor CUDAHalfType::kl_div_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDAHalfTensor>(target.pImpl,"target",2, false);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaHalfDistKLDivCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor & CUDAHalfType::kl_div_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDAHalfTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",5, false);
    THNN_CudaHalfDistKLDivCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::kl_div_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDAHalfTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfDistKLDivCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAHalfType::l1_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDAHalfTensor>(target.pImpl,"target",2, false);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",4, false);
    THNN_CudaHalfAbsCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor CUDAHalfType::l1_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDAHalfTensor>(target.pImpl,"target",2, false);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaHalfAbsCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor & CUDAHalfType::l1_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDAHalfTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",5, false);
    THNN_CudaHalfAbsCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDAHalfTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfAbsCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAHalfType::mse_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDAHalfTensor>(target.pImpl,"target",2, false);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",4, false);
    THNN_CudaHalfMSECriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor CUDAHalfType::mse_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDAHalfTensor>(target.pImpl,"target",2, false);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaHalfMSECriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor & CUDAHalfType::mse_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDAHalfTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",5, false);
    THNN_CudaHalfMSECriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::mse_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDAHalfTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfMSECriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAHalfType::multi_margin_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",2, false);
    auto p_ = p.toDouble();
    auto margin_ = margin.toDouble();
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",5, true);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",7, false);
    THNN_CudaHalfMultiMarginCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, p_, weight_ ? weight_->tensor : NULL, margin_, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor CUDAHalfType::multi_margin_loss_forward(const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",2, false);
    auto p_ = p.toDouble();
    auto margin_ = margin.toDouble();
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",5, true);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaHalfMultiMarginCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, p_, weight_ ? weight_->tensor : NULL, margin_, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor & CUDAHalfType::multi_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",3, false);
    auto p_ = p.toDouble();
    auto margin_ = margin.toDouble();
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",6, true);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",8, false);
    THNN_CudaHalfMultiMarginCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, p_, weight_ ? weight_->tensor : NULL, margin_, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::multi_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",3, false);
    auto p_ = p.toDouble();
    auto margin_ = margin.toDouble();
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",6, true);
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfMultiMarginCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, p_, weight_ ? weight_->tensor : NULL, margin_, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &> CUDAHalfType::multilabel_margin_loss_forward_out(Tensor & output, Tensor & is_target, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",2, false);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",4, false);
    auto is_target_ = checked_cast_tensor<CUDAHalfTensor>(is_target.pImpl,"is_target",4, false);
    THNN_CudaHalfMultiLabelMarginCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, is_target_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    is_target_->maybeScalar(target_->isScalar());
    return std::tuple<Tensor &, Tensor &>(output, is_target);
}
std::tuple<Tensor,Tensor> CUDAHalfType::multilabel_margin_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",2, false);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    auto is_target_ = new CUDAHalfTensor(context);
    auto is_target = Tensor(is_target_, false);
    THNN_CudaHalfMultiLabelMarginCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, is_target_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    is_target_->maybeScalar(target_->isScalar());
    return std::tuple<Tensor, Tensor>(output, is_target);
}
Tensor & CUDAHalfType::multilabel_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce, const Tensor & is_target) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",3, false);
    auto is_target_ = checked_cast_tensor<CUDAHalfTensor>(is_target.pImpl,"is_target",6, false);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",6, false);
    THNN_CudaHalfMultiLabelMarginCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, is_target_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::multilabel_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce, const Tensor & is_target) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",3, false);
    auto is_target_ = checked_cast_tensor<CUDAHalfTensor>(is_target.pImpl,"is_target",6, false);
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfMultiLabelMarginCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, is_target_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &> CUDAHalfType::nll_loss_forward_out(Tensor & output, Tensor & total_weight, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",2, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",3, true);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",6, false);
    auto total_weight_ = checked_cast_tensor<CUDAHalfTensor>(total_weight.pImpl,"total_weight",6, false);
    THNN_CudaHalfClassNLLCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, weight_ ? weight_->tensor : NULL, total_weight_->tensor, ignore_index, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    total_weight_->maybeScalar(true);
    return std::tuple<Tensor &, Tensor &>(output, total_weight);
}
std::tuple<Tensor,Tensor> CUDAHalfType::nll_loss_forward(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",2, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",3, true);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    auto total_weight_ = new CUDAHalfTensor(context);
    auto total_weight = Tensor(total_weight_, false);
    THNN_CudaHalfClassNLLCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, weight_ ? weight_->tensor : NULL, total_weight_->tensor, ignore_index, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    total_weight_->maybeScalar(true);
    return std::tuple<Tensor, Tensor>(output, total_weight);
}
Tensor & CUDAHalfType::nll_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",3, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",4, true);
    auto total_weight_ = checked_cast_tensor<CUDAHalfTensor>(total_weight.pImpl,"total_weight",8, false);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",8, false);
    THNN_CudaHalfClassNLLCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, weight_ ? weight_->tensor : NULL, total_weight_->tensor, ignore_index, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::nll_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",3, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",4, true);
    auto total_weight_ = checked_cast_tensor<CUDAHalfTensor>(total_weight.pImpl,"total_weight",8, false);
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfClassNLLCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, weight_ ? weight_->tensor : NULL, total_weight_->tensor, ignore_index, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &> CUDAHalfType::nll_loss2d_forward_out(Tensor & output, Tensor & total_weight, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",2, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",3, true);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",6, false);
    auto total_weight_ = checked_cast_tensor<CUDAHalfTensor>(total_weight.pImpl,"total_weight",6, false);
    THNN_CudaHalfSpatialClassNLLCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, weight_ ? weight_->tensor : NULL, total_weight_->tensor, ignore_index, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    total_weight_->maybeScalar(true);
    return std::tuple<Tensor &, Tensor &>(output, total_weight);
}
std::tuple<Tensor,Tensor> CUDAHalfType::nll_loss2d_forward(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",2, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",3, true);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    auto total_weight_ = new CUDAHalfTensor(context);
    auto total_weight = Tensor(total_weight_, false);
    THNN_CudaHalfSpatialClassNLLCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, weight_ ? weight_->tensor : NULL, total_weight_->tensor, ignore_index, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    total_weight_->maybeScalar(true);
    return std::tuple<Tensor, Tensor>(output, total_weight);
}
Tensor & CUDAHalfType::nll_loss2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",3, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",4, true);
    auto total_weight_ = checked_cast_tensor<CUDAHalfTensor>(total_weight.pImpl,"total_weight",8, false);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",8, false);
    THNN_CudaHalfSpatialClassNLLCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, weight_ ? weight_->tensor : NULL, total_weight_->tensor, ignore_index, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::nll_loss2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",3, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",4, true);
    auto total_weight_ = checked_cast_tensor<CUDAHalfTensor>(total_weight.pImpl,"total_weight",8, false);
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfSpatialClassNLLCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, weight_ ? weight_->tensor : NULL, total_weight_->tensor, ignore_index, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAHalfType::smooth_l1_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDAHalfTensor>(target.pImpl,"target",2, false);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",4, false);
    THNN_CudaHalfSmoothL1Criterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor CUDAHalfType::smooth_l1_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDAHalfTensor>(target.pImpl,"target",2, false);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaHalfSmoothL1Criterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor & CUDAHalfType::smooth_l1_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDAHalfTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",5, false);
    THNN_CudaHalfSmoothL1Criterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::smooth_l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDAHalfTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfSmoothL1Criterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAHalfType::soft_margin_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDAHalfTensor>(target.pImpl,"target",2, false);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",4, false);
    THNN_CudaHalfSoftMarginCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor CUDAHalfType::soft_margin_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDAHalfTensor>(target.pImpl,"target",2, false);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaHalfSoftMarginCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor & CUDAHalfType::soft_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDAHalfTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",5, false);
    THNN_CudaHalfSoftMarginCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::soft_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDAHalfTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfSoftMarginCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAHalfType::elu_forward_out(Tensor & output, const Tensor & self, Scalar alpha, Scalar scale) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toDouble();
    auto scale_ = scale.toDouble();
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",3, false);
    THNN_CudaHalfELU_updateOutput(context->thc_state, self_->tensor, output_->tensor, alpha_, scale_, false);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAHalfType::elu_forward(const Tensor & self, Scalar alpha, Scalar scale) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toDouble();
    auto scale_ = scale.toDouble();
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaHalfELU_updateOutput(context->thc_state, self_->tensor, output_->tensor, alpha_, scale_, false);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAHalfType::elu_backward_out(Tensor & grad_input, const Tensor & grad_output, Scalar alpha, Scalar scale, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto alpha_ = alpha.toDouble();
    auto scale_ = scale.toDouble();
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",4, false);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_CudaHalfELU_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, output_->tensor, alpha_, scale_);
    grad_input_->maybeScalar(output_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::elu_backward(const Tensor & grad_output, Scalar alpha, Scalar scale, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto alpha_ = alpha.toDouble();
    auto scale_ = scale.toDouble();
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",4, false);
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfELU_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, output_->tensor, alpha_, scale_);
    grad_input_->maybeScalar(output_->isScalar());
    return grad_input;
}
Tensor & CUDAHalfType::elu_forward_(Tensor & self, Scalar alpha, Scalar scale) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toDouble();
    auto scale_ = scale.toDouble();
    THNN_CudaHalfELU_updateOutput(context->thc_state, self_->tensor, self_->tensor, alpha_, scale_, true);
    return self;
}
Tensor & CUDAHalfType::glu_forward_out(Tensor & output, const Tensor & self, int64_t dim) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",2, false);
    THNN_CudaHalfGatedLinear_updateOutput(context->thc_state, self_->tensor, output_->tensor, dim);
    output_->maybeScalar(false);
    return output;
}
Tensor CUDAHalfType::glu_forward(const Tensor & self, int64_t dim) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaHalfGatedLinear_updateOutput(context->thc_state, self_->tensor, output_->tensor, dim);
    output_->maybeScalar(false);
    return output;
}
Tensor & CUDAHalfType::glu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    dim = maybe_wrap_dim(dim, self_);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaHalfGatedLinear_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, dim);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::glu_backward(const Tensor & grad_output, const Tensor & self, int64_t dim) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    dim = maybe_wrap_dim(dim, self_);
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfGatedLinear_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, dim);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAHalfType::hardtanh_forward_out(Tensor & output, const Tensor & self, Scalar min_val, Scalar max_val) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto min_val_ = min_val.toDouble();
    auto max_val_ = max_val.toDouble();
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",3, false);
    THNN_CudaHalfHardTanh_updateOutput(context->thc_state, self_->tensor, output_->tensor, min_val_, max_val_, false);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAHalfType::hardtanh_forward(const Tensor & self, Scalar min_val, Scalar max_val) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto min_val_ = min_val.toDouble();
    auto max_val_ = max_val.toDouble();
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaHalfHardTanh_updateOutput(context->thc_state, self_->tensor, output_->tensor, min_val_, max_val_, false);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAHalfType::hardtanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto min_val_ = min_val.toDouble();
    auto max_val_ = max_val.toDouble();
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_CudaHalfHardTanh_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, min_val_, max_val_, false);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::hardtanh_backward(const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto min_val_ = min_val.toDouble();
    auto max_val_ = max_val.toDouble();
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfHardTanh_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, min_val_, max_val_, false);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAHalfType::hardtanh_forward_(Tensor & self, Scalar min_val, Scalar max_val) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto min_val_ = min_val.toDouble();
    auto max_val_ = max_val.toDouble();
    THNN_CudaHalfHardTanh_updateOutput(context->thc_state, self_->tensor, self_->tensor, min_val_, max_val_, true);
    return self;
}
Tensor & CUDAHalfType::leaky_relu_forward_out(Tensor & output, const Tensor & self, Scalar negative_slope) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto negative_slope_ = negative_slope.toDouble();
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",2, false);
    THNN_CudaHalfLeakyReLU_updateOutput(context->thc_state, self_->tensor, output_->tensor, negative_slope_, false);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAHalfType::leaky_relu_forward(const Tensor & self, Scalar negative_slope) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto negative_slope_ = negative_slope.toDouble();
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaHalfLeakyReLU_updateOutput(context->thc_state, self_->tensor, output_->tensor, negative_slope_, false);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAHalfType::leaky_relu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar negative_slope) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto negative_slope_ = negative_slope.toDouble();
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaHalfLeakyReLU_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, negative_slope_, false);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::leaky_relu_backward(const Tensor & grad_output, const Tensor & self, Scalar negative_slope) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto negative_slope_ = negative_slope.toDouble();
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfLeakyReLU_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, negative_slope_, false);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAHalfType::leaky_relu_forward_(Tensor & self, Scalar negative_slope) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto negative_slope_ = negative_slope.toDouble();
    THNN_CudaHalfLeakyReLU_updateOutput(context->thc_state, self_->tensor, self_->tensor, negative_slope_, true);
    return self;
}
std::tuple<Tensor &,Tensor &> CUDAHalfType::log_sigmoid_forward_out(Tensor & output, Tensor & buffer, const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",1, false);
    auto buffer_ = checked_cast_tensor<CUDAHalfTensor>(buffer.pImpl,"buffer",1, false);
    THNN_CudaHalfLogSigmoid_updateOutput(context->thc_state, self_->tensor, output_->tensor, buffer_->tensor);
    output_->maybeScalar(self_->isScalar());
    buffer_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &>(output, buffer);
}
std::tuple<Tensor,Tensor> CUDAHalfType::log_sigmoid_forward(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    auto buffer_ = new CUDAHalfTensor(context);
    auto buffer = Tensor(buffer_, false);
    THNN_CudaHalfLogSigmoid_updateOutput(context->thc_state, self_->tensor, output_->tensor, buffer_->tensor);
    output_->maybeScalar(self_->isScalar());
    buffer_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor>(output, buffer);
}
Tensor & CUDAHalfType::log_sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & buffer) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto buffer_ = checked_cast_tensor<CUDAHalfTensor>(buffer.pImpl,"buffer",3, false);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaHalfLogSigmoid_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, buffer_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::log_sigmoid_backward(const Tensor & grad_output, const Tensor & self, const Tensor & buffer) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto buffer_ = checked_cast_tensor<CUDAHalfTensor>(buffer.pImpl,"buffer",3, false);
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfLogSigmoid_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, buffer_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAHalfType::log_softmax_forward_out(Tensor & output, const Tensor & self, int64_t dim) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",2, false);
    THNN_CudaHalfLogSoftMax_updateOutput(context->thc_state, self_->tensor, output_->tensor, dim);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAHalfType::log_softmax_forward(const Tensor & self, int64_t dim) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaHalfLogSoftMax_updateOutput(context->thc_state, self_->tensor, output_->tensor, dim);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAHalfType::log_softmax_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",4, false);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_CudaHalfLogSoftMax_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, output_->tensor, dim);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::log_softmax_backward(const Tensor & grad_output, const Tensor & self, int64_t dim, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",4, false);
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfLogSoftMax_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, output_->tensor, dim);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAHalfType::prelu_forward_out(Tensor & output, const Tensor & self, const Tensor & weight) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",2, false);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",2, false);
    THNN_CudaHalfPReLU_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAHalfType::prelu_forward(const Tensor & self, const Tensor & weight) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",2, false);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaHalfPReLU_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
std::tuple<Tensor &,Tensor &> CUDAHalfType::prelu_backward_out(Tensor & grad_input, Tensor & grad_weight, const Tensor & grad_output, const Tensor & self, const Tensor & weight) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",3, false);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",3, true);
    auto grad_weight_ = checked_cast_tensor<CUDAHalfTensor>(grad_weight.pImpl,"grad_weight",3, true);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    if (grad_input_) THNN_CudaHalfPReLU_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor);
    if (grad_weight_) THNN_CudaHalfPReLU_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &>(grad_input, grad_weight);
}
std::tuple<Tensor,Tensor> CUDAHalfType::prelu_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, std::array<bool,2> output_mask) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",3, false);
    auto grad_input_ = output_mask[0] ? new CUDAHalfTensor(context) : nullptr;
    auto grad_input = Tensor(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_input_, false);
    auto grad_weight_ = output_mask[1] ? new CUDAHalfTensor(context) : nullptr;
    auto grad_weight = Tensor(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_weight_, false);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    if (grad_input_) THNN_CudaHalfPReLU_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor);
    if (grad_weight_) THNN_CudaHalfPReLU_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor>(grad_input, grad_weight);
}
Tensor & CUDAHalfType::rrelu_with_noise_forward_out(Tensor & output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto noise_ = checked_cast_tensor<CUDAHalfTensor>(noise.pImpl,"noise",2, false);
    auto lower_ = lower.toDouble();
    auto upper_ = upper.toDouble();
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",6, false);
    THNN_CudaHalfRReLU_updateOutput(context->thc_state, self_->tensor, output_->tensor, noise_->tensor, lower_, upper_, training, false, NULL);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAHalfType::rrelu_with_noise_forward(const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto noise_ = checked_cast_tensor<CUDAHalfTensor>(noise.pImpl,"noise",2, false);
    auto lower_ = lower.toDouble();
    auto upper_ = upper.toDouble();
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaHalfRReLU_updateOutput(context->thc_state, self_->tensor, output_->tensor, noise_->tensor, lower_, upper_, training, false, NULL);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAHalfType::rrelu_with_noise_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto noise_ = checked_cast_tensor<CUDAHalfTensor>(noise.pImpl,"noise",3, false);
    auto lower_ = lower.toDouble();
    auto upper_ = upper.toDouble();
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",6, false);
    THNN_CudaHalfRReLU_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, noise_->tensor, lower_, upper_, training, false);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::rrelu_with_noise_backward(const Tensor & grad_output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto noise_ = checked_cast_tensor<CUDAHalfTensor>(noise.pImpl,"noise",3, false);
    auto lower_ = lower.toDouble();
    auto upper_ = upper.toDouble();
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfRReLU_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, noise_->tensor, lower_, upper_, training, false);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAHalfType::rrelu_with_noise_forward_(Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto noise_ = checked_cast_tensor<CUDAHalfTensor>(noise.pImpl,"noise",2, false);
    auto lower_ = lower.toDouble();
    auto upper_ = upper.toDouble();
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    THNN_CudaHalfRReLU_updateOutput(context->thc_state, self_->tensor, self_->tensor, noise_->tensor, lower_, upper_, training, true, NULL);
    return self;
}
Tensor & CUDAHalfType::softmax_forward_out(Tensor & output, const Tensor & self, int64_t dim) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",2, false);
    THNN_CudaHalfSoftMax_updateOutput(context->thc_state, self_->tensor, output_->tensor, dim);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAHalfType::softmax_forward(const Tensor & self, int64_t dim) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaHalfSoftMax_updateOutput(context->thc_state, self_->tensor, output_->tensor, dim);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAHalfType::softmax_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",4, false);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_CudaHalfSoftMax_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, output_->tensor, dim);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::softmax_backward(const Tensor & grad_output, const Tensor & self, int64_t dim, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",4, false);
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfSoftMax_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, output_->tensor, dim);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAHalfType::softplus_forward_out(Tensor & output, const Tensor & self, Scalar beta, Scalar threshold) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toDouble();
    auto threshold_ = threshold.toDouble();
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",3, false);
    THNN_CudaHalfSoftPlus_updateOutput(context->thc_state, self_->tensor, output_->tensor, beta_, threshold_);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAHalfType::softplus_forward(const Tensor & self, Scalar beta, Scalar threshold) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toDouble();
    auto threshold_ = threshold.toDouble();
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaHalfSoftPlus_updateOutput(context->thc_state, self_->tensor, output_->tensor, beta_, threshold_);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAHalfType::softplus_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto beta_ = beta.toDouble();
    auto threshold_ = threshold.toDouble();
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",5, false);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",5, false);
    THNN_CudaHalfSoftPlus_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, output_->tensor, beta_, threshold_);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::softplus_backward(const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto beta_ = beta.toDouble();
    auto threshold_ = threshold.toDouble();
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",5, false);
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfSoftPlus_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, output_->tensor, beta_, threshold_);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAHalfType::softshrink_forward_out(Tensor & output, const Tensor & self, Scalar lambd) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto lambd_ = lambd.toDouble();
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",2, false);
    THNN_CudaHalfSoftShrink_updateOutput(context->thc_state, self_->tensor, output_->tensor, lambd_);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAHalfType::softshrink_forward(const Tensor & self, Scalar lambd) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto lambd_ = lambd.toDouble();
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaHalfSoftShrink_updateOutput(context->thc_state, self_->tensor, output_->tensor, lambd_);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAHalfType::softshrink_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar lambd) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto lambd_ = lambd.toDouble();
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaHalfSoftShrink_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, lambd_);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::softshrink_backward(const Tensor & grad_output, const Tensor & self, Scalar lambd) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto lambd_ = lambd.toDouble();
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfSoftShrink_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, lambd_);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAHalfType::threshold_forward_out(Tensor & output, const Tensor & self, Scalar threshold, Scalar value) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto threshold_ = threshold.toDouble();
    auto value_ = value.toDouble();
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",3, false);
    THNN_CudaHalfThreshold_updateOutput(context->thc_state, self_->tensor, output_->tensor, threshold_, value_, false);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAHalfType::threshold_forward(const Tensor & self, Scalar threshold, Scalar value) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto threshold_ = threshold.toDouble();
    auto value_ = value.toDouble();
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaHalfThreshold_updateOutput(context->thc_state, self_->tensor, output_->tensor, threshold_, value_, false);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAHalfType::threshold_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar threshold, Scalar value) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto threshold_ = threshold.toDouble();
    auto value_ = value.toDouble();
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_CudaHalfThreshold_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, threshold_, value_, false);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::threshold_backward(const Tensor & grad_output, const Tensor & self, Scalar threshold, Scalar value) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto threshold_ = threshold.toDouble();
    auto value_ = value.toDouble();
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfThreshold_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, threshold_, value_, false);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAHalfType::threshold_forward_(Tensor & self, Scalar threshold, Scalar value) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto threshold_ = threshold.toDouble();
    auto value_ = value.toDouble();
    THNN_CudaHalfThreshold_updateOutput(context->thc_state, self_->tensor, self_->tensor, threshold_, value_, true);
    return self;
}
Tensor & CUDAHalfType::adaptive_avg_pool2d_forward_out(Tensor & output, const Tensor & self, IntList output_size) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 2);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",2, false);
    THNN_CudaHalfSpatialAdaptiveAveragePooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[1], output_size_[0]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAHalfType::adaptive_avg_pool2d_forward(const Tensor & self, IntList output_size) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 2);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaHalfSpatialAdaptiveAveragePooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[1], output_size_[0]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAHalfType::adaptive_avg_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",2, false);
    THNN_CudaHalfSpatialAdaptiveAveragePooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::adaptive_avg_pool2d_backward(const Tensor & grad_output, const Tensor & self) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfSpatialAdaptiveAveragePooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAHalfType::adaptive_avg_pool3d_forward_out(Tensor & output, const Tensor & self, IntList output_size) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 2);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",2, false);
    THNN_CudaHalfVolumetricAdaptiveAveragePooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[0], output_size_[2], output_size_[1]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAHalfType::adaptive_avg_pool3d_forward(const Tensor & self, IntList output_size) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 2);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaHalfVolumetricAdaptiveAveragePooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[0], output_size_[2], output_size_[1]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAHalfType::adaptive_avg_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",2, false);
    THNN_CudaHalfVolumetricAdaptiveAveragePooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::adaptive_avg_pool3d_backward(const Tensor & grad_output, const Tensor & self) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfVolumetricAdaptiveAveragePooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &> CUDAHalfType::adaptive_max_pool2d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList output_size) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 2);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",2, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",2, false);
    THNN_CudaHalfSpatialAdaptiveMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, output_size_[1], output_size_[0]);
    bool maybe_scalar = self_->isScalar();
    output_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(output, indices);
}
std::tuple<Tensor,Tensor> CUDAHalfType::adaptive_max_pool2d_forward(const Tensor & self, IntList output_size) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 2);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    auto indices_ = new CUDALongTensor(context);
    auto indices = Tensor(indices_, false);
    THNN_CudaHalfSpatialAdaptiveMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, output_size_[1], output_size_[0]);
    bool maybe_scalar = self_->isScalar();
    output_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(output, indices);
}
Tensor & CUDAHalfType::adaptive_max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",3, false);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaHalfSpatialAdaptiveMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::adaptive_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",3, false);
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfSpatialAdaptiveMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &> CUDAHalfType::adaptive_max_pool3d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList output_size) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 2);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",2, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",2, false);
    THNN_CudaHalfVolumetricAdaptiveMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, output_size_[0], output_size_[2], output_size_[1]);
    bool maybe_scalar = self_->isScalar();
    output_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(output, indices);
}
std::tuple<Tensor,Tensor> CUDAHalfType::adaptive_max_pool3d_forward(const Tensor & self, IntList output_size) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 2);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    auto indices_ = new CUDALongTensor(context);
    auto indices = Tensor(indices_, false);
    THNN_CudaHalfVolumetricAdaptiveMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, output_size_[0], output_size_[2], output_size_[1]);
    bool maybe_scalar = self_->isScalar();
    output_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(output, indices);
}
Tensor & CUDAHalfType::adaptive_max_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",3, false);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaHalfVolumetricAdaptiveMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::adaptive_max_pool3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",3, false);
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfVolumetricAdaptiveMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAHalfType::avg_pool2d_forward_out(Tensor & output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 2);
    auto stride_ = check_intlist<2>(stride, "stride", 3, kernel_size);
    auto padding_ = check_intlist<2>(padding, "padding", 4);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",6, false);
    THNN_CudaHalfSpatialAveragePooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], ceil_mode, count_include_pad);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAHalfType::avg_pool2d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 2);
    auto stride_ = check_intlist<2>(stride, "stride", 3, kernel_size);
    auto padding_ = check_intlist<2>(padding, "padding", 4);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaHalfSpatialAveragePooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], ceil_mode, count_include_pad);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAHalfType::avg_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto stride_ = check_intlist<2>(stride, "stride", 4, kernel_size);
    auto padding_ = check_intlist<2>(padding, "padding", 5);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",7, false);
    THNN_CudaHalfSpatialAveragePooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], ceil_mode, count_include_pad);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::avg_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto stride_ = check_intlist<2>(stride, "stride", 4, kernel_size);
    auto padding_ = check_intlist<2>(padding, "padding", 5);
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfSpatialAveragePooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], ceil_mode, count_include_pad);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAHalfType::avg_pool3d_forward_out(Tensor & output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 2);
    auto stride_ = check_intlist<3>(stride, "stride", 3, kernel_size);
    auto padding_ = check_intlist<3>(padding, "padding", 4);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",6, false);
    THNN_CudaHalfVolumetricAveragePooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], ceil_mode, count_include_pad);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAHalfType::avg_pool3d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 2);
    auto stride_ = check_intlist<3>(stride, "stride", 3, kernel_size);
    auto padding_ = check_intlist<3>(padding, "padding", 4);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaHalfVolumetricAveragePooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], ceil_mode, count_include_pad);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAHalfType::avg_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto stride_ = check_intlist<3>(stride, "stride", 4, kernel_size);
    auto padding_ = check_intlist<3>(padding, "padding", 5);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",7, false);
    THNN_CudaHalfVolumetricAveragePooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], ceil_mode, count_include_pad);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::avg_pool3d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto stride_ = check_intlist<3>(stride, "stride", 4, kernel_size);
    auto padding_ = check_intlist<3>(padding, "padding", 5);
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfVolumetricAveragePooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], ceil_mode, count_include_pad);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &> CUDAHalfType::fractional_max_pool2d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & random_samples) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 2);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 3);
    auto random_samples_ = checked_cast_tensor<CUDAHalfTensor>(random_samples.pImpl,"random_samples",4, false);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",4, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",4, false);
    THNN_CudaHalfSpatialFractionalMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[1], output_size_[0], kernel_size_[1], kernel_size_[0], indices_->tensor, random_samples_->tensor);
    output_->maybeScalar(false);
    return std::tuple<Tensor &, Tensor &>(output, indices);
}
std::tuple<Tensor,Tensor> CUDAHalfType::fractional_max_pool2d_forward(const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & random_samples) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 2);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 3);
    auto random_samples_ = checked_cast_tensor<CUDAHalfTensor>(random_samples.pImpl,"random_samples",4, false);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    auto indices_ = new CUDALongTensor(context);
    auto indices = Tensor(indices_, false);
    THNN_CudaHalfSpatialFractionalMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[1], output_size_[0], kernel_size_[1], kernel_size_[0], indices_->tensor, random_samples_->tensor);
    output_->maybeScalar(false);
    return std::tuple<Tensor, Tensor>(output, indices);
}
Tensor & CUDAHalfType::fractional_max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 4);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",5, false);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",5, false);
    THNN_CudaHalfSpatialFractionalMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, output_size_[1], output_size_[0], kernel_size_[1], kernel_size_[0], indices_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::fractional_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 4);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",5, false);
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfSpatialFractionalMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, output_size_[1], output_size_[0], kernel_size_[1], kernel_size_[0], indices_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &> CUDAHalfType::max_pool2d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 2);
    auto stride_ = check_intlist<2>(stride, "stride", 3, kernel_size);
    auto padding_ = check_intlist<2>(padding, "padding", 4);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 5);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",6, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",6, false);
    THNN_CudaHalfSpatialDilatedMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], ceil_mode);
    bool maybe_scalar = self_->isScalar();
    output_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(output, indices);
}
std::tuple<Tensor,Tensor> CUDAHalfType::max_pool2d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 2);
    auto stride_ = check_intlist<2>(stride, "stride", 3, kernel_size);
    auto padding_ = check_intlist<2>(padding, "padding", 4);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 5);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    auto indices_ = new CUDALongTensor(context);
    auto indices = Tensor(indices_, false);
    THNN_CudaHalfSpatialDilatedMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], ceil_mode);
    bool maybe_scalar = self_->isScalar();
    output_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(output, indices);
}
Tensor & CUDAHalfType::max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto stride_ = check_intlist<2>(stride, "stride", 4, kernel_size);
    auto padding_ = check_intlist<2>(padding, "padding", 5);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 6);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",8, false);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",8, false);
    THNN_CudaHalfSpatialDilatedMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], ceil_mode);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::max_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto stride_ = check_intlist<2>(stride, "stride", 4, kernel_size);
    auto padding_ = check_intlist<2>(padding, "padding", 5);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 6);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",8, false);
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfSpatialDilatedMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], ceil_mode);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &> CUDAHalfType::max_pool3d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 2);
    auto stride_ = check_intlist<3>(stride, "stride", 3, kernel_size);
    auto padding_ = check_intlist<3>(padding, "padding", 4);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 5);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",6, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",6, false);
    THNN_CudaHalfVolumetricDilatedMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], ceil_mode);
    bool maybe_scalar = self_->isScalar();
    output_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(output, indices);
}
std::tuple<Tensor,Tensor> CUDAHalfType::max_pool3d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 2);
    auto stride_ = check_intlist<3>(stride, "stride", 3, kernel_size);
    auto padding_ = check_intlist<3>(padding, "padding", 4);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 5);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    auto indices_ = new CUDALongTensor(context);
    auto indices = Tensor(indices_, false);
    THNN_CudaHalfVolumetricDilatedMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], ceil_mode);
    bool maybe_scalar = self_->isScalar();
    output_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(output, indices);
}
Tensor & CUDAHalfType::max_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto stride_ = check_intlist<3>(stride, "stride", 4, kernel_size);
    auto padding_ = check_intlist<3>(padding, "padding", 5);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 6);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",8, false);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",8, false);
    THNN_CudaHalfVolumetricDilatedMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], ceil_mode);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::max_pool3d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto stride_ = check_intlist<3>(stride, "stride", 4, kernel_size);
    auto padding_ = check_intlist<3>(padding, "padding", 5);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 6);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",8, false);
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfVolumetricDilatedMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], ceil_mode);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAHalfType::max_unpool2d_forward_out(Tensor & output, const Tensor & self, const Tensor & indices, IntList output_size) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",2, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 3);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",3, false);
    THNN_CudaHalfSpatialMaxUnpooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, output_size_[1], output_size_[0]);
    output_->maybeScalar(self_->isScalar() && indices_->isScalar());
    return output;
}
Tensor CUDAHalfType::max_unpool2d_forward(const Tensor & self, const Tensor & indices, IntList output_size) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",2, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 3);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaHalfSpatialMaxUnpooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, output_size_[1], output_size_[0]);
    output_->maybeScalar(self_->isScalar() && indices_->isScalar());
    return output;
}
Tensor & CUDAHalfType::max_unpool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",3, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 4);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_CudaHalfSpatialMaxUnpooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor, output_size_[1], output_size_[0]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::max_unpool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",3, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 4);
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfSpatialMaxUnpooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor, output_size_[1], output_size_[0]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAHalfType::max_unpool3d_forward_out(Tensor & output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",2, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 3);
    auto stride_ = check_intlist<3>(stride, "stride", 4);
    auto padding_ = check_intlist<3>(padding, "padding", 5);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",5, false);
    THNN_CudaHalfVolumetricMaxUnpooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, output_size_[0], output_size_[2], output_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1]);
    output_->maybeScalar(self_->isScalar() && indices_->isScalar());
    return output;
}
Tensor CUDAHalfType::max_unpool3d_forward(const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",2, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 3);
    auto stride_ = check_intlist<3>(stride, "stride", 4);
    auto padding_ = check_intlist<3>(padding, "padding", 5);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaHalfVolumetricMaxUnpooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, output_size_[0], output_size_[2], output_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1]);
    output_->maybeScalar(self_->isScalar() && indices_->isScalar());
    return output;
}
Tensor & CUDAHalfType::max_unpool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",3, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 4);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",6, false);
    THNN_CudaHalfVolumetricMaxUnpooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor, output_size_[0], output_size_[2], output_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::max_unpool3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",3, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 4);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfVolumetricMaxUnpooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor, output_size_[0], output_size_[2], output_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAHalfType::reflection_pad1d_forward_out(Tensor & output, const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<2>(padding, "padding", 2);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",2, false);
    THNN_CudaHalfTemporalReflectionPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAHalfType::reflection_pad1d_forward(const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<2>(padding, "padding", 2);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaHalfTemporalReflectionPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAHalfType::reflection_pad1d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<2>(padding, "padding", 3);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaHalfTemporalReflectionPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::reflection_pad1d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<2>(padding, "padding", 3);
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfTemporalReflectionPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAHalfType::reflection_pad2d_forward_out(Tensor & output, const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<4>(padding, "padding", 2);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",2, false);
    THNN_CudaHalfSpatialReflectionPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1], padding_[2], padding_[3]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAHalfType::reflection_pad2d_forward(const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<4>(padding, "padding", 2);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaHalfSpatialReflectionPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1], padding_[2], padding_[3]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAHalfType::reflection_pad2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<4>(padding, "padding", 3);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaHalfSpatialReflectionPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1], padding_[2], padding_[3]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::reflection_pad2d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<4>(padding, "padding", 3);
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfSpatialReflectionPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1], padding_[2], padding_[3]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAHalfType::replication_pad1d_forward_out(Tensor & output, const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<2>(padding, "padding", 2);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",2, false);
    THNN_CudaHalfTemporalReplicationPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAHalfType::replication_pad1d_forward(const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<2>(padding, "padding", 2);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaHalfTemporalReplicationPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAHalfType::replication_pad1d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<2>(padding, "padding", 3);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaHalfTemporalReplicationPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::replication_pad1d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<2>(padding, "padding", 3);
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfTemporalReplicationPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAHalfType::replication_pad2d_forward_out(Tensor & output, const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<4>(padding, "padding", 2);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",2, false);
    THNN_CudaHalfSpatialReplicationPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1], padding_[2], padding_[3]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAHalfType::replication_pad2d_forward(const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<4>(padding, "padding", 2);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaHalfSpatialReplicationPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1], padding_[2], padding_[3]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAHalfType::replication_pad2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<4>(padding, "padding", 3);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaHalfSpatialReplicationPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1], padding_[2], padding_[3]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::replication_pad2d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<4>(padding, "padding", 3);
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfSpatialReplicationPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1], padding_[2], padding_[3]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAHalfType::replication_pad3d_forward_out(Tensor & output, const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<6>(padding, "padding", 2);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",2, false);
    THNN_CudaHalfVolumetricReplicationPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1], padding_[2], padding_[3], padding_[4], padding_[5]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAHalfType::replication_pad3d_forward(const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<6>(padding, "padding", 2);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaHalfVolumetricReplicationPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1], padding_[2], padding_[3], padding_[4], padding_[5]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAHalfType::replication_pad3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<6>(padding, "padding", 3);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaHalfVolumetricReplicationPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1], padding_[2], padding_[3], padding_[4], padding_[5]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::replication_pad3d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<6>(padding, "padding", 3);
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfVolumetricReplicationPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1], padding_[2], padding_[3], padding_[4], padding_[5]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAHalfType::upsample_linear1d_forward_out(Tensor & output, const Tensor & self, IntList output_size, bool align_corners) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<1>(output_size, "output_size", 2);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",3, false);
    THNN_CudaHalfTemporalUpSamplingLinear_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[0], align_corners);
    return output;
}
Tensor CUDAHalfType::upsample_linear1d_forward(const Tensor & self, IntList output_size, bool align_corners) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<1>(output_size, "output_size", 2);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaHalfTemporalUpSamplingLinear_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[0], align_corners);
    return output;
}
Tensor & CUDAHalfType::upsample_linear1d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_size_ = check_intlist<1>(output_size, "output_size", 2);
    auto input_size_ = check_intlist<3>(input_size, "input_size", 3);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_CudaHalfTemporalUpSamplingLinear_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, input_size_[0], input_size_[1], input_size_[2], output_size_[0], align_corners);
    grad_input_->maybeScalar(false);
    return grad_input;
}
Tensor CUDAHalfType::upsample_linear1d_backward(const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_size_ = check_intlist<1>(output_size, "output_size", 2);
    auto input_size_ = check_intlist<3>(input_size, "input_size", 3);
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfTemporalUpSamplingLinear_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, input_size_[0], input_size_[1], input_size_[2], output_size_[0], align_corners);
    grad_input_->maybeScalar(false);
    return grad_input;
}
Tensor & CUDAHalfType::upsample_bilinear2d_forward_out(Tensor & output, const Tensor & self, IntList output_size, bool align_corners) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 2);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",3, false);
    THNN_CudaHalfSpatialUpSamplingBilinear_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[0], output_size_[1], align_corners);
    return output;
}
Tensor CUDAHalfType::upsample_bilinear2d_forward(const Tensor & self, IntList output_size, bool align_corners) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 2);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaHalfSpatialUpSamplingBilinear_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[0], output_size_[1], align_corners);
    return output;
}
Tensor & CUDAHalfType::upsample_bilinear2d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 2);
    auto input_size_ = check_intlist<4>(input_size, "input_size", 3);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_CudaHalfSpatialUpSamplingBilinear_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, input_size_[0], input_size_[1], input_size_[2], input_size_[3], output_size_[0], output_size_[1], align_corners);
    grad_input_->maybeScalar(false);
    return grad_input;
}
Tensor CUDAHalfType::upsample_bilinear2d_backward(const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 2);
    auto input_size_ = check_intlist<4>(input_size, "input_size", 3);
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfSpatialUpSamplingBilinear_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, input_size_[0], input_size_[1], input_size_[2], input_size_[3], output_size_[0], output_size_[1], align_corners);
    grad_input_->maybeScalar(false);
    return grad_input;
}
Tensor & CUDAHalfType::upsample_trilinear3d_forward_out(Tensor & output, const Tensor & self, IntList output_size, bool align_corners) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 2);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",3, false);
    THNN_CudaHalfVolumetricUpSamplingTrilinear_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[0], output_size_[1], output_size_[2], align_corners);
    return output;
}
Tensor CUDAHalfType::upsample_trilinear3d_forward(const Tensor & self, IntList output_size, bool align_corners) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 2);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaHalfVolumetricUpSamplingTrilinear_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[0], output_size_[1], output_size_[2], align_corners);
    return output;
}
Tensor & CUDAHalfType::upsample_trilinear3d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 2);
    auto input_size_ = check_intlist<5>(input_size, "input_size", 3);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_CudaHalfVolumetricUpSamplingTrilinear_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, input_size_[0], input_size_[1], input_size_[2], input_size_[3], input_size_[4], output_size_[0], output_size_[1], output_size_[2], align_corners);
    grad_input_->maybeScalar(false);
    return grad_input;
}
Tensor CUDAHalfType::upsample_trilinear3d_backward(const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 2);
    auto input_size_ = check_intlist<5>(input_size, "input_size", 3);
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfVolumetricUpSamplingTrilinear_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, input_size_[0], input_size_[1], input_size_[2], input_size_[3], input_size_[4], output_size_[0], output_size_[1], output_size_[2], align_corners);
    grad_input_->maybeScalar(false);
    return grad_input;
}
Tensor & CUDAHalfType::upsample_nearest1d_forward_out(Tensor & output, const Tensor & self, int64_t scale_factor) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",2, false);
    THNN_CudaHalfTemporalUpSamplingNearest_updateOutput(context->thc_state, self_->tensor, output_->tensor, scale_factor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAHalfType::upsample_nearest1d_forward(const Tensor & self, int64_t scale_factor) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaHalfTemporalUpSamplingNearest_updateOutput(context->thc_state, self_->tensor, output_->tensor, scale_factor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAHalfType::upsample_nearest1d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaHalfTemporalUpSamplingNearest_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, scale_factor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::upsample_nearest1d_backward(const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfTemporalUpSamplingNearest_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, scale_factor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAHalfType::upsample_nearest2d_forward_out(Tensor & output, const Tensor & self, int64_t scale_factor) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",2, false);
    THNN_CudaHalfSpatialUpSamplingNearest_updateOutput(context->thc_state, self_->tensor, output_->tensor, scale_factor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAHalfType::upsample_nearest2d_forward(const Tensor & self, int64_t scale_factor) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaHalfSpatialUpSamplingNearest_updateOutput(context->thc_state, self_->tensor, output_->tensor, scale_factor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAHalfType::upsample_nearest2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaHalfSpatialUpSamplingNearest_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, scale_factor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::upsample_nearest2d_backward(const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfSpatialUpSamplingNearest_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, scale_factor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAHalfType::upsample_nearest3d_forward_out(Tensor & output, const Tensor & self, int64_t scale_factor) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",2, false);
    THNN_CudaHalfVolumetricUpSamplingNearest_updateOutput(context->thc_state, self_->tensor, output_->tensor, scale_factor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAHalfType::upsample_nearest3d_forward(const Tensor & self, int64_t scale_factor) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaHalfVolumetricUpSamplingNearest_updateOutput(context->thc_state, self_->tensor, output_->tensor, scale_factor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAHalfType::upsample_nearest3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaHalfVolumetricUpSamplingNearest_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, scale_factor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::upsample_nearest3d_backward(const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfVolumetricUpSamplingNearest_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, scale_factor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDAHalfType::_sigmoid_forward_out(Tensor & output, const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",1, false);
    THNN_CudaHalfSigmoid_updateOutput(context->thc_state, self_->tensor, output_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAHalfType::_sigmoid_forward(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaHalfSigmoid_updateOutput(context->thc_state, self_->tensor, output_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAHalfType::_sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",2, false);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",2, false);
    THNN_CudaHalfSigmoid_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, output_->tensor);
    grad_input_->maybeScalar(output_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::_sigmoid_backward(const Tensor & grad_output, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",2, false);
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfSigmoid_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, output_->tensor);
    grad_input_->maybeScalar(output_->isScalar());
    return grad_input;
}
Tensor & CUDAHalfType::_tanh_forward_out(Tensor & output, const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",1, false);
    THNN_CudaHalfTanh_updateOutput(context->thc_state, self_->tensor, output_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDAHalfType::_tanh_forward(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaHalfTanh_updateOutput(context->thc_state, self_->tensor, output_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDAHalfType::_tanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",2, false);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",2, false);
    THNN_CudaHalfTanh_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, output_->tensor);
    grad_input_->maybeScalar(output_->isScalar());
    return grad_input;
}
Tensor CUDAHalfType::_tanh_backward(const Tensor & grad_output, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",2, false);
    auto grad_input_ = new CUDAHalfTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaHalfTanh_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, output_->tensor);
    grad_input_->maybeScalar(output_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDAHalfType::thnn_batch_norm_forward_out(Tensor & output, Tensor & save_mean, Tensor & save_std, const Tensor & self, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",2, true);
    auto bias_ = checked_cast_tensor<CUDAHalfTensor>(bias.pImpl,"bias",3, true);
    auto running_mean_ = checked_cast_tensor<CUDAHalfTensor>(running_mean.pImpl,"running_mean",4, true);
    auto running_var_ = checked_cast_tensor<CUDAHalfTensor>(running_var.pImpl,"running_var",5, true);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",8, false);
    auto save_mean_ = checked_cast_tensor<CUDAHalfTensor>(save_mean.pImpl,"save_mean",8, false);
    auto save_std_ = checked_cast_tensor<CUDAHalfTensor>(save_std.pImpl,"save_std",8, false);
    THNN_CudaHalfBatchNormalization_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_ ? weight_->tensor : NULL, bias_ ? bias_->tensor : NULL, running_mean_ ? running_mean_->tensor : NULL, running_var_ ? running_var_->tensor : NULL, save_mean_->tensor, save_std_->tensor, training, momentum, eps);
    bool maybe_scalar = self_->isScalar() && (!weight_ || weight_->isScalar()) && (!bias_ || bias_->isScalar()) && (!running_mean_ || running_mean_->isScalar()) && (!running_var_ || running_var_->isScalar());
    output_->maybeScalar(maybe_scalar);
    save_mean_->maybeScalar(maybe_scalar);
    save_std_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(output, save_mean, save_std);
}
std::tuple<Tensor,Tensor,Tensor> CUDAHalfType::thnn_batch_norm_forward(const Tensor & self, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",2, true);
    auto bias_ = checked_cast_tensor<CUDAHalfTensor>(bias.pImpl,"bias",3, true);
    auto running_mean_ = checked_cast_tensor<CUDAHalfTensor>(running_mean.pImpl,"running_mean",4, true);
    auto running_var_ = checked_cast_tensor<CUDAHalfTensor>(running_var.pImpl,"running_var",5, true);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    auto save_mean_ = new CUDAHalfTensor(context);
    auto save_mean = Tensor(save_mean_, false);
    auto save_std_ = new CUDAHalfTensor(context);
    auto save_std = Tensor(save_std_, false);
    THNN_CudaHalfBatchNormalization_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_ ? weight_->tensor : NULL, bias_ ? bias_->tensor : NULL, running_mean_ ? running_mean_->tensor : NULL, running_var_ ? running_var_->tensor : NULL, save_mean_->tensor, save_std_->tensor, training, momentum, eps);
    bool maybe_scalar = self_->isScalar() && (!weight_ || weight_->isScalar()) && (!bias_ || bias_->isScalar()) && (!running_mean_ || running_mean_->isScalar()) && (!running_var_ || running_var_->isScalar());
    output_->maybeScalar(maybe_scalar);
    save_mean_->maybeScalar(maybe_scalar);
    save_std_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(output, save_mean, save_std);
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDAHalfType::thnn_batch_norm_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, bool training, double eps, const Tensor & save_mean, const Tensor & save_std) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",3, true);
    auto running_mean_ = checked_cast_tensor<CUDAHalfTensor>(running_mean.pImpl,"running_mean",4, true);
    auto running_var_ = checked_cast_tensor<CUDAHalfTensor>(running_var.pImpl,"running_var",5, true);
    auto save_mean_ = checked_cast_tensor<CUDAHalfTensor>(save_mean.pImpl,"save_mean",8, true);
    auto save_std_ = checked_cast_tensor<CUDAHalfTensor>(save_std.pImpl,"save_std",9, true);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",9, true);
    auto grad_weight_ = checked_cast_tensor<CUDAHalfTensor>(grad_weight.pImpl,"grad_weight",9, true);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = checked_cast_tensor<CUDAHalfTensor>(grad_bias.pImpl,"grad_bias",9, true);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    THNN_CudaHalfBatchNormalization_backward(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, weight_ ? weight_->tensor : NULL, running_mean_ ? running_mean_->tensor : NULL, running_var_ ? running_var_->tensor : NULL, save_mean_ ? save_mean_->tensor : NULL, save_std_ ? save_std_->tensor : NULL, training, 1, eps);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &, Tensor &>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> CUDAHalfType::thnn_batch_norm_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, bool training, double eps, const Tensor & save_mean, const Tensor & save_std, std::array<bool,3> output_mask) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",3, true);
    auto running_mean_ = checked_cast_tensor<CUDAHalfTensor>(running_mean.pImpl,"running_mean",4, true);
    auto running_var_ = checked_cast_tensor<CUDAHalfTensor>(running_var.pImpl,"running_var",5, true);
    auto save_mean_ = checked_cast_tensor<CUDAHalfTensor>(save_mean.pImpl,"save_mean",8, true);
    auto save_std_ = checked_cast_tensor<CUDAHalfTensor>(save_std.pImpl,"save_std",9, true);
    auto grad_input_ = output_mask[0] ? new CUDAHalfTensor(context) : nullptr;
    auto grad_input = Tensor(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_input_, false);
    auto grad_weight_ = output_mask[1] ? new CUDAHalfTensor(context) : nullptr;
    auto grad_weight = Tensor(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_weight_, false);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = output_mask[2] ? new CUDAHalfTensor(context) : nullptr;
    auto grad_bias = Tensor(grad_bias_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_bias_, false);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    THNN_CudaHalfBatchNormalization_backward(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, weight_ ? weight_->tensor : NULL, running_mean_ ? running_mean_->tensor : NULL, running_var_ ? running_var_->tensor : NULL, save_mean_ ? save_mean_->tensor : NULL, save_std_ ? save_std_->tensor : NULL, training, 1, eps);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDAHalfType::thnn_conv_transpose2d_forward_out(Tensor & output, Tensor & columns, Tensor & ones, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CUDAHalfTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto output_padding_ = check_intlist<2>(output_padding, "output_padding", 7);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 8);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",8, false);
    auto columns_ = checked_cast_tensor<CUDAHalfTensor>(columns.pImpl,"columns",8, false);
    auto ones_ = checked_cast_tensor<CUDAHalfTensor>(ones.pImpl,"ones",8, false);
    THNN_CudaHalfSpatialFullDilatedConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], output_padding_[1], output_padding_[0]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    columns_->maybeScalar(maybe_scalar);
    ones_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(output, columns, ones);
}
std::tuple<Tensor,Tensor,Tensor> CUDAHalfType::thnn_conv_transpose2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CUDAHalfTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto output_padding_ = check_intlist<2>(output_padding, "output_padding", 7);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 8);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    auto columns_ = new CUDAHalfTensor(context);
    auto columns = Tensor(columns_, false);
    auto ones_ = new CUDAHalfTensor(context);
    auto ones = Tensor(ones_, false);
    THNN_CudaHalfSpatialFullDilatedConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], output_padding_[1], output_padding_[0]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    columns_->maybeScalar(maybe_scalar);
    ones_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(output, columns, ones);
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDAHalfType::thnn_conv_transpose2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & columns, const Tensor & ones) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto output_padding_ = check_intlist<2>(output_padding, "output_padding", 7);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 8);
    auto columns_ = checked_cast_tensor<CUDAHalfTensor>(columns.pImpl,"columns",9, false);
    auto ones_ = checked_cast_tensor<CUDAHalfTensor>(ones.pImpl,"ones",10, false);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",10, true);
    auto grad_weight_ = checked_cast_tensor<CUDAHalfTensor>(grad_weight.pImpl,"grad_weight",10, true);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = checked_cast_tensor<CUDAHalfTensor>(grad_bias.pImpl,"grad_bias",10, true);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(1) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_CudaHalfSpatialFullDilatedConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, columns_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], output_padding_[1], output_padding_[0]);
    if (grad_weight_ || grad_bias_) THNN_CudaHalfSpatialFullDilatedConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], output_padding_[1], output_padding_[0], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &, Tensor &>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> CUDAHalfType::thnn_conv_transpose2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto output_padding_ = check_intlist<2>(output_padding, "output_padding", 7);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 8);
    auto columns_ = checked_cast_tensor<CUDAHalfTensor>(columns.pImpl,"columns",9, false);
    auto ones_ = checked_cast_tensor<CUDAHalfTensor>(ones.pImpl,"ones",10, false);
    auto grad_input_ = output_mask[0] ? new CUDAHalfTensor(context) : nullptr;
    auto grad_input = Tensor(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_input_, false);
    auto grad_weight_ = output_mask[1] ? new CUDAHalfTensor(context) : nullptr;
    auto grad_weight = Tensor(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_weight_, false);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = output_mask[2] ? new CUDAHalfTensor(context) : nullptr;
    auto grad_bias = Tensor(grad_bias_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_bias_, false);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(1) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_CudaHalfSpatialFullDilatedConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, columns_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], output_padding_[1], output_padding_[0]);
    if (grad_weight_ || grad_bias_) THNN_CudaHalfSpatialFullDilatedConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], output_padding_[1], output_padding_[0], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDAHalfType::thnn_conv_transpose3d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CUDAHalfTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto output_padding_ = check_intlist<3>(output_padding, "output_padding", 7);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 8);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",8, false);
    auto finput_ = checked_cast_tensor<CUDAHalfTensor>(finput.pImpl,"finput",8, false);
    auto fgrad_input_ = checked_cast_tensor<CUDAHalfTensor>(fgrad_input.pImpl,"fgrad_input",8, false);
    THNN_CudaHalfVolumetricFullDilatedConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], output_padding_[0], output_padding_[2], output_padding_[1]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    finput_->maybeScalar(maybe_scalar);
    fgrad_input_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(output, finput, fgrad_input);
}
std::tuple<Tensor,Tensor,Tensor> CUDAHalfType::thnn_conv_transpose3d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CUDAHalfTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto output_padding_ = check_intlist<3>(output_padding, "output_padding", 7);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 8);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    auto finput_ = new CUDAHalfTensor(context);
    auto finput = Tensor(finput_, false);
    auto fgrad_input_ = new CUDAHalfTensor(context);
    auto fgrad_input = Tensor(fgrad_input_, false);
    THNN_CudaHalfVolumetricFullDilatedConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], output_padding_[0], output_padding_[2], output_padding_[1]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    finput_->maybeScalar(maybe_scalar);
    fgrad_input_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(output, finput, fgrad_input);
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDAHalfType::thnn_conv_transpose3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & finput, const Tensor & fgrad_input) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto output_padding_ = check_intlist<3>(output_padding, "output_padding", 7);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 8);
    auto finput_ = checked_cast_tensor<CUDAHalfTensor>(finput.pImpl,"finput",9, false);
    auto fgrad_input_ = checked_cast_tensor<CUDAHalfTensor>(fgrad_input.pImpl,"fgrad_input",10, false);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",10, true);
    auto grad_weight_ = checked_cast_tensor<CUDAHalfTensor>(grad_weight.pImpl,"grad_weight",10, true);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = checked_cast_tensor<CUDAHalfTensor>(grad_bias.pImpl,"grad_bias",10, true);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(1) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_CudaHalfVolumetricFullDilatedConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], output_padding_[0], output_padding_[2], output_padding_[1]);
    if (grad_weight_ || grad_bias_) THNN_CudaHalfVolumetricFullDilatedConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], output_padding_[0], output_padding_[2], output_padding_[1], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &, Tensor &>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> CUDAHalfType::thnn_conv_transpose3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto output_padding_ = check_intlist<3>(output_padding, "output_padding", 7);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 8);
    auto finput_ = checked_cast_tensor<CUDAHalfTensor>(finput.pImpl,"finput",9, false);
    auto fgrad_input_ = checked_cast_tensor<CUDAHalfTensor>(fgrad_input.pImpl,"fgrad_input",10, false);
    auto grad_input_ = output_mask[0] ? new CUDAHalfTensor(context) : nullptr;
    auto grad_input = Tensor(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_input_, false);
    auto grad_weight_ = output_mask[1] ? new CUDAHalfTensor(context) : nullptr;
    auto grad_weight = Tensor(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_weight_, false);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = output_mask[2] ? new CUDAHalfTensor(context) : nullptr;
    auto grad_bias = Tensor(grad_bias_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_bias_, false);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(1) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_CudaHalfVolumetricFullDilatedConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], output_padding_[0], output_padding_[2], output_padding_[1]);
    if (grad_weight_ || grad_bias_) THNN_CudaHalfVolumetricFullDilatedConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], output_padding_[0], output_padding_[2], output_padding_[1], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDAHalfType::thnn_conv2d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CUDAHalfTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",6, false);
    auto finput_ = checked_cast_tensor<CUDAHalfTensor>(finput.pImpl,"finput",6, false);
    auto fgrad_input_ = checked_cast_tensor<CUDAHalfTensor>(fgrad_input.pImpl,"fgrad_input",6, false);
    THNN_CudaHalfSpatialConvolutionMM_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    finput_->maybeScalar(maybe_scalar);
    fgrad_input_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(output, finput, fgrad_input);
}
std::tuple<Tensor,Tensor,Tensor> CUDAHalfType::thnn_conv2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CUDAHalfTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    auto finput_ = new CUDAHalfTensor(context);
    auto finput = Tensor(finput_, false);
    auto fgrad_input_ = new CUDAHalfTensor(context);
    auto fgrad_input = Tensor(fgrad_input_, false);
    THNN_CudaHalfSpatialConvolutionMM_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    finput_->maybeScalar(maybe_scalar);
    fgrad_input_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(output, finput, fgrad_input);
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDAHalfType::thnn_conv2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto finput_ = checked_cast_tensor<CUDAHalfTensor>(finput.pImpl,"finput",7, false);
    auto fgrad_input_ = checked_cast_tensor<CUDAHalfTensor>(fgrad_input.pImpl,"fgrad_input",8, false);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",8, true);
    auto grad_weight_ = checked_cast_tensor<CUDAHalfTensor>(grad_weight.pImpl,"grad_weight",8, true);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = checked_cast_tensor<CUDAHalfTensor>(grad_bias.pImpl,"grad_bias",8, true);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_CudaHalfSpatialConvolutionMM_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, finput_->tensor, fgrad_input_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
    if (grad_weight_ || grad_bias_) THNN_CudaHalfSpatialConvolutionMM_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &, Tensor &>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> CUDAHalfType::thnn_conv2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto finput_ = checked_cast_tensor<CUDAHalfTensor>(finput.pImpl,"finput",7, false);
    auto fgrad_input_ = checked_cast_tensor<CUDAHalfTensor>(fgrad_input.pImpl,"fgrad_input",8, false);
    auto grad_input_ = output_mask[0] ? new CUDAHalfTensor(context) : nullptr;
    auto grad_input = Tensor(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_input_, false);
    auto grad_weight_ = output_mask[1] ? new CUDAHalfTensor(context) : nullptr;
    auto grad_weight = Tensor(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_weight_, false);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = output_mask[2] ? new CUDAHalfTensor(context) : nullptr;
    auto grad_bias = Tensor(grad_bias_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_bias_, false);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_CudaHalfSpatialConvolutionMM_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, finput_->tensor, fgrad_input_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
    if (grad_weight_ || grad_bias_) THNN_CudaHalfSpatialConvolutionMM_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}
Tensor & CUDAHalfType::thnn_conv_depthwise2d_forward_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CUDAHalfTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",7, false);
    THNN_CudaHalfSpatialDepthwiseConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
    output_->maybeScalar(self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar()));
    return output;
}
Tensor CUDAHalfType::thnn_conv_depthwise2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CUDAHalfTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaHalfSpatialDepthwiseConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
    output_->maybeScalar(self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar()));
    return output;
}
std::tuple<Tensor &,Tensor &> CUDAHalfType::thnn_conv_depthwise2d_backward_out(Tensor & grad_input, Tensor & grad_weight, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",7, true);
    auto grad_weight_ = checked_cast_tensor<CUDAHalfTensor>(grad_weight.pImpl,"grad_weight",7, true);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    if (grad_input_) THNN_CudaHalfSpatialDepthwiseConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
    if (grad_weight_) THNN_CudaHalfSpatialDepthwiseConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &>(grad_input, grad_weight);
}
std::tuple<Tensor,Tensor> CUDAHalfType::thnn_conv_depthwise2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, std::array<bool,2> output_mask) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
    auto grad_input_ = output_mask[0] ? new CUDAHalfTensor(context) : nullptr;
    auto grad_input = Tensor(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_input_, false);
    auto grad_weight_ = output_mask[1] ? new CUDAHalfTensor(context) : nullptr;
    auto grad_weight = Tensor(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_weight_, false);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    if (grad_input_) THNN_CudaHalfSpatialDepthwiseConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
    if (grad_weight_) THNN_CudaHalfSpatialDepthwiseConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor>(grad_input, grad_weight);
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDAHalfType::thnn_conv_dilated2d_forward_out(Tensor & output, Tensor & columns, Tensor & ones, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CUDAHalfTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",7, false);
    auto columns_ = checked_cast_tensor<CUDAHalfTensor>(columns.pImpl,"columns",7, false);
    auto ones_ = checked_cast_tensor<CUDAHalfTensor>(ones.pImpl,"ones",7, false);
    THNN_CudaHalfSpatialDilatedConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    columns_->maybeScalar(maybe_scalar);
    ones_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(output, columns, ones);
}
std::tuple<Tensor,Tensor,Tensor> CUDAHalfType::thnn_conv_dilated2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CUDAHalfTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    auto columns_ = new CUDAHalfTensor(context);
    auto columns = Tensor(columns_, false);
    auto ones_ = new CUDAHalfTensor(context);
    auto ones = Tensor(ones_, false);
    THNN_CudaHalfSpatialDilatedConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    columns_->maybeScalar(maybe_scalar);
    ones_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(output, columns, ones);
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDAHalfType::thnn_conv_dilated2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
    auto columns_ = checked_cast_tensor<CUDAHalfTensor>(columns.pImpl,"columns",8, false);
    auto ones_ = checked_cast_tensor<CUDAHalfTensor>(ones.pImpl,"ones",9, false);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",9, true);
    auto grad_weight_ = checked_cast_tensor<CUDAHalfTensor>(grad_weight.pImpl,"grad_weight",9, true);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = checked_cast_tensor<CUDAHalfTensor>(grad_bias.pImpl,"grad_bias",9, true);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_CudaHalfSpatialDilatedConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, columns_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
    if (grad_weight_ || grad_bias_) THNN_CudaHalfSpatialDilatedConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &, Tensor &>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> CUDAHalfType::thnn_conv_dilated2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
    auto columns_ = checked_cast_tensor<CUDAHalfTensor>(columns.pImpl,"columns",8, false);
    auto ones_ = checked_cast_tensor<CUDAHalfTensor>(ones.pImpl,"ones",9, false);
    auto grad_input_ = output_mask[0] ? new CUDAHalfTensor(context) : nullptr;
    auto grad_input = Tensor(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_input_, false);
    auto grad_weight_ = output_mask[1] ? new CUDAHalfTensor(context) : nullptr;
    auto grad_weight = Tensor(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_weight_, false);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = output_mask[2] ? new CUDAHalfTensor(context) : nullptr;
    auto grad_bias = Tensor(grad_bias_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_bias_, false);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_CudaHalfSpatialDilatedConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, columns_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
    if (grad_weight_ || grad_bias_) THNN_CudaHalfSpatialDilatedConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDAHalfType::thnn_conv_dilated3d_forward_out(Tensor & output, Tensor & columns, Tensor & ones, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CUDAHalfTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 7);
    auto output_ = checked_cast_tensor<CUDAHalfTensor>(output.pImpl,"output",7, false);
    auto columns_ = checked_cast_tensor<CUDAHalfTensor>(columns.pImpl,"columns",7, false);
    auto ones_ = checked_cast_tensor<CUDAHalfTensor>(ones.pImpl,"ones",7, false);
    THNN_CudaHalfVolumetricDilatedConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    columns_->maybeScalar(maybe_scalar);
    ones_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(output, columns, ones);
}
std::tuple<Tensor,Tensor,Tensor> CUDAHalfType::thnn_conv_dilated3d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CUDAHalfTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 7);
    auto output_ = new CUDAHalfTensor(context);
    auto output = Tensor(output_, false);
    auto columns_ = new CUDAHalfTensor(context);
    auto columns = Tensor(columns_, false);
    auto ones_ = new CUDAHalfTensor(context);
    auto ones = Tensor(ones_, false);
    THNN_CudaHalfVolumetricDilatedConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    columns_->maybeScalar(maybe_scalar);
    ones_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(output, columns, ones);
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDAHalfType::thnn_conv_dilated3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 7);
    auto columns_ = checked_cast_tensor<CUDAHalfTensor>(columns.pImpl,"columns",8, false);
    auto ones_ = checked_cast_tensor<CUDAHalfTensor>(ones.pImpl,"ones",9, false);
    auto grad_input_ = checked_cast_tensor<CUDAHalfTensor>(grad_input.pImpl,"grad_input",9, true);
    auto grad_weight_ = checked_cast_tensor<CUDAHalfTensor>(grad_weight.pImpl,"grad_weight",9, true);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = checked_cast_tensor<CUDAHalfTensor>(grad_bias.pImpl,"grad_bias",9, true);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_CudaHalfVolumetricDilatedConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, columns_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1]);
    if (grad_weight_ || grad_bias_) THNN_CudaHalfVolumetricDilatedConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &, Tensor &>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> CUDAHalfType::thnn_conv_dilated3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) const {
    auto grad_output_ = checked_cast_tensor<CUDAHalfTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDAHalfTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDAHalfTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 7);
    auto columns_ = checked_cast_tensor<CUDAHalfTensor>(columns.pImpl,"columns",8, false);
    auto ones_ = checked_cast_tensor<CUDAHalfTensor>(ones.pImpl,"ones",9, false);
    auto grad_input_ = output_mask[0] ? new CUDAHalfTensor(context) : nullptr;
    auto grad_input = Tensor(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_input_, false);
    auto grad_weight_ = output_mask[1] ? new CUDAHalfTensor(context) : nullptr;
    auto grad_weight = Tensor(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_weight_, false);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = output_mask[2] ? new CUDAHalfTensor(context) : nullptr;
    auto grad_bias = Tensor(grad_bias_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_bias_, false);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_CudaHalfVolumetricDilatedConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, columns_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1]);
    if (grad_weight_ || grad_bias_) THNN_CudaHalfVolumetricDilatedConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}
Tensor & CUDAHalfType::abs_out(Tensor & result, const Tensor & self) const {
    return  at::native::_abs_out_cuda(result, self);
}
Tensor & CUDAHalfType::ceil_out(Tensor & result, const Tensor & self) const {
    return  at::native::_ceil_out_cuda(result, self);
}
Tensor & CUDAHalfType::cos_out(Tensor & result, const Tensor & self) const {
    return  at::native::_cos_out_cuda(result, self);
}
Tensor CUDAHalfType::embedding_dense_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) const {
    return  at::native::embedding_backward_cuda(grad, indices, num_weights, padding_idx, scale_grad_by_freq);
}
Tensor & CUDAHalfType::embedding_renorm_(Tensor & self, const Tensor & indices, double max_norm, double norm_type) const {
    return  at::native::embedding_renorm_cuda_(self, indices, max_norm, norm_type);
}
std::tuple<Tensor,Tensor,Tensor> CUDAHalfType::embedding_bag(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse) const {
    return  at::native::embedding_bag_cuda(weight, indices, offsets, scale_grad_by_freq, mode, sparse);
}
Tensor CUDAHalfType::embedding_bag_dense_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode) const {
    return  at::native::embedding_bag_backward_cuda(grad, indices, offsets, offset2bag, bag_size, num_weights, scale_grad_by_freq, mode);
}
Tensor & CUDAHalfType::exp_out(Tensor & result, const Tensor & self) const {
    return  at::native::_exp_out_cuda(result, self);
}
Tensor & CUDAHalfType::eye_out(Tensor & result, int64_t n, int64_t m) const {
    return  at::native::eye_out_cuda(result, n, m);
}
Tensor & CUDAHalfType::floor_out(Tensor & result, const Tensor & self) const {
    return  at::native::_floor_out_cuda(result, self);
}
Tensor CUDAHalfType::_fft_with_size(const Tensor & self, int64_t signal_ndim, bool complex_input, bool complex_output, bool inverse, IntList checked_signal_sizes, bool normalized, bool onesided, IntList output_sizes) const {
    return  at::native::_fft_cufft(self, signal_ndim, complex_input, complex_output, inverse, checked_signal_sizes, normalized, onesided, output_sizes);
}
Tensor & CUDAHalfType::log_out(Tensor & result, const Tensor & self) const {
    return  at::native::_log_out_cuda(result, self);
}
std::tuple<Tensor,Tensor> CUDAHalfType::RoiPooling2d_forward(const Tensor & input, const Tensor & rois, int64_t pooledHeight, int64_t pooledWidth, double spatialScale) const {
    return  at::native::RoiPooling2d_forward_cuda(input, rois, pooledHeight, pooledWidth, spatialScale);
}
Tensor CUDAHalfType::RoiPooling2d_backward(const Tensor & input, const Tensor & rois, int64_t pooledHeight, int64_t pooledWidth, double spatialScale, const Tensor & gradOutput, const Tensor & argmaxes) const {
    return  at::native::RoiPooling2d_backward_cuda(input, rois, pooledHeight, pooledWidth, spatialScale, gradOutput, argmaxes);
}
Tensor & CUDAHalfType::round_out(Tensor & result, const Tensor & self) const {
    return  at::native::_round_out_cuda(result, self);
}
Tensor & CUDAHalfType::sin_out(Tensor & result, const Tensor & self) const {
    return  at::native::_sin_out_cuda(result, self);
}
Tensor & CUDAHalfType::sspaddmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return  at::native::_sspaddmm_out_only_sparse(result, self, mat1, mat2, beta, alpha);
}
Tensor CUDAHalfType::sum(const Tensor & self) const {
    return  at::native::_sum_cuda(self);
}
Tensor & CUDAHalfType::sum_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    return  at::native::_sum_out_cuda(result, self, dim, keepdim);
}
Tensor & CUDAHalfType::sqrt_out(Tensor & result, const Tensor & self) const {
    return  at::native::_sqrt_out_cuda(result, self);
}
Tensor CUDAHalfType::prod(const Tensor & self) const {
    return  at::native::_prod_cuda(self);
}
Tensor & CUDAHalfType::prod_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    return  at::native::_prod_out_cuda(result, self, dim, keepdim);
}
Tensor & CUDAHalfType::trunc_out(Tensor & result, const Tensor & self) const {
    return  at::native::_trunc_out_cuda(result, self);
}
std::tuple<Tensor,Tensor> CUDAHalfType::_unique(const Tensor & self, bool sorted, bool return_inverse) const {
    return  at::native::_unique_cuda(self, sorted, return_inverse);
}
Tensor CUDAHalfType::_s_where(const Tensor & condition, const Tensor & self, const Tensor & other) const {
    return  at::native::_s_where_cuda(condition, self, other);
}
Tensor CUDAHalfType::_standard_gamma_grad(const Tensor & self, const Tensor & output) const {
    return  at::native::_standard_gamma_grad_cuda(self, output);
}
Tensor CUDAHalfType::poisson(const Tensor & self, Generator * generator) const {
    return  at::native::_s_poisson_cuda(self, generator);
}

}
