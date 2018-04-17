// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/CUDAShortType.h"
#include "ATen/CUDAShortStorage.h"
#include "ATen/CUDAShortTensor.h"
#include "ATen/CUDAGenerator.h"
#include "ATen/CUDAByteTensor.h"
#include "ATen/CUDAIntTensor.h"
#include "ATen/CUDALongTensor.h"
#include "ATen/SparseCUDAShortTensor.h"
#include "ATen/CUDAShortTensor.h"
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

CUDAShortType::CUDAShortType(Context* context)
: Type(context, /*is_variable_or_undefined=*/false) {}
ScalarType CUDAShortType::scalarType() const {
  return ScalarType::Short;
}
Backend CUDAShortType::backend() const {
  return Backend::CUDA;
}
bool CUDAShortType::is_cuda() const { return backend() == kCUDA || backend() == kSparseCUDA; }
bool CUDAShortType::is_sparse() const { return backend() == kSparseCPU || backend() == kSparseCUDA; }
bool CUDAShortType::is_distributed() const { return false; }

std::unique_ptr<Storage> CUDAShortType::storage() const {
  return std::unique_ptr<Storage>(new CUDAShortStorage(context));
}
std::unique_ptr<Storage> CUDAShortType::storage(size_t size) const {
  return std::unique_ptr<Storage>(new CUDAShortStorage(context,size));
}
std::unique_ptr<Storage> CUDAShortType::storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const {
    return std::unique_ptr<Storage>(
      new CUDAShortStorage(context,data,size,deleter));
}
std::unique_ptr<Storage> CUDAShortType::storageWithAllocator(int64_t size, std::unique_ptr<Allocator> allocator) const {
    return std::unique_ptr<Storage>(
        new CUDAShortStorage(context, size, std::move(allocator)));
}
Tensor CUDAShortType::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THCudaShortTensor_retain(context->thc_state,  (THCudaShortTensor*) th_pointer);
  return Tensor(new CUDAShortTensor(context,(THCudaShortTensor*)(th_pointer)), false);
}
std::unique_ptr<Storage> CUDAShortType::unsafeStorageFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THCudaShortStorage_retain(context->thc_state,  (THCudaShortStorage*) th_pointer);
  return std::unique_ptr<Storage>(new CUDAShortStorage(context, (THCudaShortStorage*) th_pointer));
}
std::unique_ptr<Generator> CUDAShortType::generator() const {
  return std::unique_ptr<Generator>(new CUDAGenerator(context));
}

const char * CUDAShortType::toString() const {
  return CUDAShortType::typeString();
}
TypeID CUDAShortType::ID() const {
  return TypeID::CUDAShort;
}

std::size_t CUDAShortType::elementSizeInBytes() const {
  return sizeof(int16_t);
}

const char * CUDAShortType::typeString() {
  return "CUDAShortType";
}

/* example
Tensor * CUDAShortType::add(Tensor & a, Tensor & b) {
  std::cout << "add CUDAShortTensor\n";
  return &a;
}
*/

int64_t CUDAShortType::storage_offset(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    return static_cast<int64_t>(THCudaShortTensor_storageOffset(context->thc_state, self_->tensor));
}
Tensor & CUDAShortType::resize_(Tensor & self, IntList size) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    THCudaShortTensor_resize(context->thc_state, self_->tensor, size_, NULL);
    self_->maybeScalar(size.size() == 0);
    return self;
}
int64_t CUDAShortType::numel(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    return static_cast<int64_t>(THCudaShortTensor_nElement(context->thc_state, self_->tensor));
}
Tensor & CUDAShortType::set_(Tensor & self, Storage & storage) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto storage_ = checked_cast_storage<CUDAShortStorage>(&storage,"storage",2);
    THCudaShortTensor_setStorage(context->thc_state, self_->tensor, storage_->storage, 0, THLongStorageView::makeFromLength(static_cast<int64_t>(storage.size())), NULL);
    self_->maybeScalar(false);
    return self;
}
Tensor & CUDAShortType::set_(Tensor & self, Storage & sourceStorage, int64_t storage_offset, IntList size, IntList stride) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto sourceStorage_ = checked_cast_storage<CUDAShortStorage>(&sourceStorage,"sourceStorage",2);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    THCudaShortTensor_setStorage(context->thc_state, self_->tensor, sourceStorage_->storage, storage_offset, size_, stride_);
    self_->maybeScalar(size.size() == 0);
    return self;
}
Tensor & CUDAShortType::set_(Tensor & self, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto source_ = checked_cast_tensor<CUDAShortTensor>(source.pImpl,"source",2, false);
    THCudaShortTensor_set(context->thc_state, self_->tensor, source_->tensor);
    self_->maybeScalar(source_->isScalar());
    return self;
}
Tensor & CUDAShortType::set_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    THCudaShortTensor_setStorage(context->thc_state, self_->tensor, NULL, 0, NULL, NULL);
    self_->maybeScalar(false);
    return self;
}
Tensor & CUDAShortType::fill_(Tensor & self, Scalar value) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toShort();
    THCudaShortTensor_fill(context->thc_state, self_->tensor, value_);
    return self;
}
Tensor & CUDAShortType::fill_(Tensor & self, const Tensor & value) const {
    if (value.dim() == 0) {
        return static_cast<const Type*>(this)->fill_(self, Scalar(value));
    }
    AT_ERROR("fill_ only supports a 0-dimensional value tensor, but got tensor "
        "with %" PRId64 " dimension(s)", value.dim());
}
bool CUDAShortType::is_contiguous(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    return THCudaShortTensor_isContiguous(context->thc_state, self_->tensor);
}
bool CUDAShortType::is_set_to(const Tensor & self, const Tensor & tensor) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto tensor_ = checked_cast_tensor<CUDAShortTensor>(tensor.pImpl,"tensor",2, false);
    return THCudaShortTensor_isSetTo(context->thc_state, self_->tensor, tensor_->tensor);
}
Tensor & CUDAShortType::s_masked_fill_(Tensor & self, const Tensor & mask, Scalar value) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CUDAByteTensor>(mask.pImpl,"mask",2, false);
    auto value_ = value.toShort();
    THCudaShortTensor_maskedFill(context->thc_state, self_->tensor, mask_->tensor, value_);
    return self;
}
Tensor & CUDAShortType::s_masked_fill_(Tensor & self, const Tensor & mask, const Tensor & value) const {
    if (value.dim() == 0) {
        return static_cast<const Type*>(this)->masked_fill_(self, mask, Scalar(value));
    }
    AT_ERROR("masked_fill_ only supports a 0-dimensional value tensor, but got tensor "
        "with %" PRId64 " dimension(s)", value.dim());
}
Tensor & CUDAShortType::s_masked_scatter_(Tensor & self, const Tensor & mask, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CUDAByteTensor>(mask.pImpl,"mask",2, false);
    auto source_ = checked_cast_tensor<CUDAShortTensor>(source.pImpl,"source",3, false);
    THCudaShortTensor_maskedCopy(context->thc_state, self_->tensor, mask_->tensor, source_->tensor);
    return self;
}
Tensor & CUDAShortType::s_masked_select_out(Tensor & result, const Tensor & self, const Tensor & mask) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CUDAByteTensor>(mask.pImpl,"mask",2, false);
    THCudaShortTensor_maskedSelect(context->thc_state, result_->tensor, self_->tensor, mask_->tensor);
    result_->maybeScalar(self_->isScalar() && mask_->isScalar());
    return result;
}
Tensor CUDAShortType::s_masked_select(const Tensor & self, const Tensor & mask) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CUDAByteTensor>(mask.pImpl,"mask",2, false);
    THCudaShortTensor_maskedSelect(context->thc_state, result_->tensor, self_->tensor, mask_->tensor);
    result_->maybeScalar(self_->isScalar() && mask_->isScalar());
    return result;
}
Tensor CUDAShortType::transpose(const Tensor & self, int64_t dim0, int64_t dim1) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    dim0 = maybe_wrap_dim(dim0, self_);
    dim1 = maybe_wrap_dim(dim1, self_);
    return Tensor((new CUDAShortTensor(context, THCudaShortTensor_newTranspose(context->thc_state, self_->tensor, dim0, dim1)))->maybeScalar(self_->isScalar()),false);
}
Tensor CUDAShortType::t(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    return Tensor((new CUDAShortTensor(context, THCudaShortTensor_newTranspose(context->thc_state, self_->tensor, 0, 1)))->maybeScalar(self_->isScalar()),false);
}
Tensor & CUDAShortType::nonzero_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDALongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    THCudaShortTensor_nonzero(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAShortType::nonzero(const Tensor & self) const {
    auto result_ = new CUDALongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    THCudaShortTensor_nonzero(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAShortType::contiguous(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    return Tensor((new CUDAShortTensor(context, THCudaShortTensor_newContiguous(context->thc_state, self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor CUDAShortType::clone(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    return Tensor((new CUDAShortTensor(context, THCudaShortTensor_newClone(context->thc_state, self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor CUDAShortType::view(const Tensor & self, IntList size) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new CUDAShortTensor(context, THCudaShortTensor_newView(context->thc_state, self_->tensor, size_)))->maybeScalar(size.size() == 0),false);
}
Tensor & CUDAShortType::resize_as_(Tensor & self, const Tensor & the_template) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto the_template_ = checked_cast_tensor<CUDAShortTensor>(the_template.pImpl,"the_template",2, false);
    THCudaShortTensor_resizeAs(context->thc_state, self_->tensor, the_template_->tensor);
    self_->maybeScalar(the_template_->isScalar());
    return self;
}
Tensor & CUDAShortType::index_select_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    THCudaShortTensor_indexSelect(context->thc_state, result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
Tensor CUDAShortType::index_select(const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    THCudaShortTensor_indexSelect(context->thc_state, result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
Tensor & CUDAShortType::_indexCopy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    auto source_ = checked_cast_tensor<CUDAShortTensor>(source.pImpl,"source",4, false);
    THCudaShortTensor_indexCopy(context->thc_state, self_->tensor, dim, index_->tensor, source_->tensor);
    return self;
}
Tensor & CUDAShortType::take_out(Tensor & result, const Tensor & self, const Tensor & index) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",2, false);
    THCudaShortTensor_take(context->thc_state, result_->tensor, self_->tensor, index_->tensor);
    result_->maybeScalar(index_->isScalar());
    return result;
}
Tensor CUDAShortType::take(const Tensor & self, const Tensor & index) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",2, false);
    THCudaShortTensor_take(context->thc_state, result_->tensor, self_->tensor, index_->tensor);
    result_->maybeScalar(index_->isScalar());
    return result;
}
Tensor & CUDAShortType::put_(Tensor & self, const Tensor & index, const Tensor & source, bool accumulate) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",2, false);
    auto source_ = checked_cast_tensor<CUDAShortTensor>(source.pImpl,"source",3, false);
    THCudaShortTensor_put(context->thc_state, self_->tensor, index_->tensor, source_->tensor, accumulate);
    return self;
}
Tensor & CUDAShortType::index_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    auto source_ = checked_cast_tensor<CUDAShortTensor>(source.pImpl,"source",4, false);
    THCudaShortTensor_indexAdd(context->thc_state, self_->tensor, dim, index_->tensor, source_->tensor);
    return self;
}
Tensor & CUDAShortType::index_fill_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    auto value_ = value.toShort();
    THCudaShortTensor_indexFill(context->thc_state, self_->tensor, dim, index_->tensor, value_);
    return self;
}
Tensor & CUDAShortType::index_fill_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & value) const {
    if (value.dim() == 0) {
        return static_cast<const Type*>(this)->index_fill_(self, dim, index, Scalar(value));
    }
    AT_ERROR("index_fill_ only supports a 0-dimensional value tensor, but got tensor "
        "with %" PRId64 " dimension(s)", value.dim());
}
Tensor CUDAShortType::unfold(const Tensor & self, int64_t dimension, int64_t size, int64_t step) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    dimension = maybe_wrap_dim(dimension, self_);
    THCudaShortTensor_unfold(context->thc_state, result_->tensor, self_->tensor, dimension, size, step);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAShortType::_range_out(Tensor & result, Scalar start, Scalar end, Scalar step) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto start_ = start.toLong();
    auto end_ = end.toLong();
    auto step_ = step.toLong();
    THCudaShortTensor_range(context->thc_state, result_->tensor, start_, end_, step_);
    return result;
}
Tensor CUDAShortType::_range(Scalar start, Scalar end, Scalar step) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto start_ = start.toLong();
    auto end_ = end.toLong();
    auto step_ = step.toLong();
    THCudaShortTensor_range(context->thc_state, result_->tensor, start_, end_, step_);
    return result;
}
Tensor & CUDAShortType::_arange_out(Tensor & result, Scalar start, Scalar end, Scalar step) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto start_ = start.toLong();
    auto end_ = end.toLong();
    auto step_ = step.toLong();
    THCudaShortTensor_arange(context->thc_state, result_->tensor, start_, end_, step_);
    return result;
}
Tensor CUDAShortType::_arange(Scalar start, Scalar end, Scalar step) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto start_ = start.toLong();
    auto end_ = end.toLong();
    auto step_ = step.toLong();
    THCudaShortTensor_arange(context->thc_state, result_->tensor, start_, end_, step_);
    return result;
}
Tensor & CUDAShortType::_arange_out(Tensor & result, Scalar end) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto end_ = end.toLong();
    THCudaShortTensor_arange(context->thc_state, result_->tensor, 0, end_, 1);
    return result;
}
Tensor CUDAShortType::_arange(Scalar end) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto end_ = end.toLong();
    THCudaShortTensor_arange(context->thc_state, result_->tensor, 0, end_, 1);
    return result;
}
Tensor & CUDAShortType::scatter_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
    if (src.dim() == 0) {
        return static_cast<const Type*>(this)->scatter_(self, dim, index, Scalar(src));
    }
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    auto src_ = checked_cast_tensor<CUDAShortTensor>(src.pImpl,"src",4, false);
    THCudaShortTensor_scatter(context->thc_state, self_->tensor, dim, index_->tensor, src_->tensor);
    return self;
}
Tensor & CUDAShortType::scatter_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    auto value_ = value.toShort();
    THCudaShortTensor_scatterFill(context->thc_state, self_->tensor, dim, index_->tensor, value_);
    return self;
}
Tensor & CUDAShortType::scatter_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    auto src_ = checked_cast_tensor<CUDAShortTensor>(src.pImpl,"src",4, false);
    THCudaShortTensor_scatterAdd(context->thc_state, self_->tensor, dim, index_->tensor, src_->tensor);
    return self;
}
Tensor & CUDAShortType::gather_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    result.resize_(index.sizes());
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    THCudaShortTensor_gather(context->thc_state, result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
Tensor CUDAShortType::gather(const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    result.resize_(index.sizes());
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    THCudaShortTensor_gather(context->thc_state, result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
void* CUDAShortType::data_ptr(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    return THCudaShortTensor_data(context->thc_state, self_->tensor);
}
bool CUDAShortType::equal(const Tensor & self, const Tensor & other) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",2, false);
    return THCudaShortTensor_equal(context->thc_state, self_->tensor, other_->tensor);
}
Tensor & CUDAShortType::__and___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_bitand(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAShortType::__and__(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_bitand(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAShortType::s___and___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__and___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",2, false);
    THCudaShortTensor_cbitand(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAShortType::s___and__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__and__(self, Scalar(other));
    }
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",2, false);
    THCudaShortTensor_cbitand(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAShortType::__iand__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_bitand(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAShortType::s___iand__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__iand__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",3, false);
    THCudaShortTensor_cbitand(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAShortType::__or___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_bitor(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAShortType::__or__(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_bitor(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAShortType::s___or___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__or___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",2, false);
    THCudaShortTensor_cbitor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAShortType::s___or__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__or__(self, Scalar(other));
    }
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",2, false);
    THCudaShortTensor_cbitor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAShortType::__ior__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_bitor(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAShortType::s___ior__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__ior__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",3, false);
    THCudaShortTensor_cbitor(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAShortType::__xor___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_bitxor(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAShortType::__xor__(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_bitxor(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAShortType::s___xor___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__xor___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",2, false);
    THCudaShortTensor_cbitxor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAShortType::s___xor__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__xor__(self, Scalar(other));
    }
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",2, false);
    THCudaShortTensor_cbitxor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAShortType::__ixor__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_bitxor(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAShortType::s___ixor__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__ixor__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",3, false);
    THCudaShortTensor_cbitxor(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAShortType::__lshift___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_lshift(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAShortType::__lshift__(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_lshift(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAShortType::s___lshift___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__lshift___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",2, false);
    THCudaShortTensor_clshift(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAShortType::s___lshift__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__lshift__(self, Scalar(other));
    }
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",2, false);
    THCudaShortTensor_clshift(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAShortType::__ilshift__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_lshift(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAShortType::s___ilshift__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__ilshift__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",3, false);
    THCudaShortTensor_clshift(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAShortType::__rshift___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_rshift(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAShortType::__rshift__(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_rshift(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAShortType::s___rshift___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__rshift___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",2, false);
    THCudaShortTensor_crshift(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAShortType::s___rshift__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__rshift__(self, Scalar(other));
    }
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",2, false);
    THCudaShortTensor_crshift(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAShortType::__irshift__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_rshift(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAShortType::s___irshift__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__irshift__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",3, false);
    THCudaShortTensor_crshift(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAShortType::lt_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_ltValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAShortType::lt(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_ltValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAShortType::s_lt_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->lt_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",2, false);
    THCudaShortTensor_ltTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAShortType::s_lt(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->lt(self, Scalar(other));
    }
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",2, false);
    THCudaShortTensor_ltTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAShortType::lt_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_ltValueT(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAShortType::s_lt_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->lt_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",3, false);
    THCudaShortTensor_ltTensorT(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAShortType::gt_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_gtValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAShortType::gt(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_gtValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAShortType::s_gt_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->gt_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",2, false);
    THCudaShortTensor_gtTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAShortType::s_gt(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->gt(self, Scalar(other));
    }
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",2, false);
    THCudaShortTensor_gtTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAShortType::gt_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_gtValueT(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAShortType::s_gt_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->gt_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",3, false);
    THCudaShortTensor_gtTensorT(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAShortType::le_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_leValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAShortType::le(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_leValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAShortType::s_le_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->le_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",2, false);
    THCudaShortTensor_leTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAShortType::s_le(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->le(self, Scalar(other));
    }
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",2, false);
    THCudaShortTensor_leTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAShortType::le_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_leValueT(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAShortType::s_le_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->le_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",3, false);
    THCudaShortTensor_leTensorT(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAShortType::ge_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_geValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAShortType::ge(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_geValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAShortType::s_ge_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ge_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",2, false);
    THCudaShortTensor_geTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAShortType::s_ge(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ge(self, Scalar(other));
    }
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",2, false);
    THCudaShortTensor_geTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAShortType::ge_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_geValueT(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAShortType::s_ge_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ge_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",3, false);
    THCudaShortTensor_geTensorT(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAShortType::eq_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_eqValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAShortType::eq(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_eqValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAShortType::s_eq_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->eq_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",2, false);
    THCudaShortTensor_eqTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAShortType::s_eq(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->eq(self, Scalar(other));
    }
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",2, false);
    THCudaShortTensor_eqTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAShortType::eq_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_eqValueT(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAShortType::s_eq_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->eq_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",3, false);
    THCudaShortTensor_eqTensorT(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAShortType::ne_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_neValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAShortType::ne(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_neValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAShortType::s_ne_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ne_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",2, false);
    THCudaShortTensor_neTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAShortType::s_ne(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ne(self, Scalar(other));
    }
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",2, false);
    THCudaShortTensor_neTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAShortType::ne_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_neValueT(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAShortType::s_ne_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ne_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",3, false);
    THCudaShortTensor_neTensorT(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
std::tuple<Tensor &,Tensor &> CUDAShortType::min_out(Tensor & min, Tensor & min_indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto min_ = checked_cast_tensor<CUDAShortTensor>(min.pImpl,"min",0, false);
    auto min_indices_ = checked_cast_tensor<CUDALongTensor>(min_indices.pImpl,"min_indices",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaShortTensor_min(context->thc_state, min_->tensor, min_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    min_->maybeScalar(maybe_scalar);
    min_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(min, min_indices);
}
std::tuple<Tensor,Tensor> CUDAShortType::min(const Tensor & self, int64_t dim, bool keepdim) const {
    auto min_ = new CUDAShortTensor(context);
    auto min = Tensor(min_, false);
    auto min_indices_ = new CUDALongTensor(context);
    auto min_indices = Tensor(min_indices_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaShortTensor_min(context->thc_state, min_->tensor, min_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    min_->maybeScalar(maybe_scalar);
    min_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(min, min_indices);
}
Tensor & CUDAShortType::s_min_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",2, false);
    THCudaShortTensor_cmin(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAShortType::s_min(const Tensor & self, const Tensor & other) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",2, false);
    THCudaShortTensor_cmin(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAShortType::min(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<int16_t>(THCudaShortTensor_minall(context->thc_state, self_->tensor)));
}
std::tuple<Tensor &,Tensor &> CUDAShortType::max_out(Tensor & max, Tensor & max_indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto max_ = checked_cast_tensor<CUDAShortTensor>(max.pImpl,"max",0, false);
    auto max_indices_ = checked_cast_tensor<CUDALongTensor>(max_indices.pImpl,"max_indices",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaShortTensor_max(context->thc_state, max_->tensor, max_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    max_->maybeScalar(maybe_scalar);
    max_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(max, max_indices);
}
std::tuple<Tensor,Tensor> CUDAShortType::max(const Tensor & self, int64_t dim, bool keepdim) const {
    auto max_ = new CUDAShortTensor(context);
    auto max = Tensor(max_, false);
    auto max_indices_ = new CUDALongTensor(context);
    auto max_indices = Tensor(max_indices_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaShortTensor_max(context->thc_state, max_->tensor, max_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    max_->maybeScalar(maybe_scalar);
    max_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(max, max_indices);
}
Tensor & CUDAShortType::s_max_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",2, false);
    THCudaShortTensor_cmax(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAShortType::s_max(const Tensor & self, const Tensor & other) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",2, false);
    THCudaShortTensor_cmax(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAShortType::max(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<int16_t>(THCudaShortTensor_maxall(context->thc_state, self_->tensor)));
}
std::tuple<Tensor &,Tensor &> CUDAShortType::mode_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = checked_cast_tensor<CUDAShortTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaShortTensor_mode(context->thc_state, values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CUDAShortType::mode(const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = new CUDAShortTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CUDALongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaShortTensor_mode(context->thc_state, values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
std::tuple<Tensor &,Tensor &> CUDAShortType::median_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = checked_cast_tensor<CUDAShortTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaShortTensor_median(context->thc_state, values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CUDAShortType::median(const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = new CUDAShortTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CUDALongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaShortTensor_median(context->thc_state, values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
Tensor CUDAShortType::median(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<int16_t>(THCudaShortTensor_medianall(context->thc_state, self_->tensor)));
}
std::tuple<Tensor &,Tensor &> CUDAShortType::sort_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool descending) const {
    auto values_ = checked_cast_tensor<CUDAShortTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaShortTensor_sort(context->thc_state, values_->tensor, indices_->tensor, self_->tensor, dim, descending);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CUDAShortType::sort(const Tensor & self, int64_t dim, bool descending) const {
    auto values_ = new CUDAShortTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CUDALongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaShortTensor_sort(context->thc_state, values_->tensor, indices_->tensor, self_->tensor, dim, descending);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
std::tuple<Tensor &,Tensor &> CUDAShortType::topk_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) const {
    auto values_ = checked_cast_tensor<CUDAShortTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaShortTensor_topk(context->thc_state, values_->tensor, indices_->tensor, self_->tensor, k, dim, largest, sorted);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CUDAShortType::topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) const {
    auto values_ = new CUDAShortTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CUDALongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaShortTensor_topk(context->thc_state, values_->tensor, indices_->tensor, self_->tensor, k, dim, largest, sorted);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
int64_t CUDAShortType::get_device(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    return static_cast<int64_t>(THCudaShortTensor_getDevice(context->thc_state, self_->tensor));
}
Tensor & CUDAShortType::_abs_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    THCudaShortTensor_abs(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAShortType::_abs(const Tensor & self) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    THCudaShortTensor_abs(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAShortType::neg_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    THCudaShortTensor_neg(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAShortType::neg(const Tensor & self) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    THCudaShortTensor_neg(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAShortType::neg_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    THCudaShortTensor_neg(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAShortType::pow_out(Tensor & result, const Tensor & self, Scalar exponent) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toShort();
    THCudaShortTensor_pow(context->thc_state, result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAShortType::pow(const Tensor & self, Scalar exponent) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toShort();
    THCudaShortTensor_pow(context->thc_state, result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAShortType::s_pow_out(Tensor & result, const Tensor & self, const Tensor & exponent) const {
    if (exponent.dim() == 0) {
        return static_cast<const Type*>(this)->pow_out(result, self, Scalar(exponent));
    }
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto exponent_ = checked_cast_tensor<CUDAShortTensor>(exponent.pImpl,"exponent",2, false);
    THCudaShortTensor_cpow(context->thc_state, result_->tensor, self_->tensor, exponent_->tensor);
    result_->maybeScalar(self_->isScalar() && exponent_->isScalar());
    return result;
}
Tensor CUDAShortType::s_pow(const Tensor & self, const Tensor & exponent) const {
    if (exponent.dim() == 0) {
        return static_cast<const Type*>(this)->pow(self, Scalar(exponent));
    }
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto exponent_ = checked_cast_tensor<CUDAShortTensor>(exponent.pImpl,"exponent",2, false);
    THCudaShortTensor_cpow(context->thc_state, result_->tensor, self_->tensor, exponent_->tensor);
    result_->maybeScalar(self_->isScalar() && exponent_->isScalar());
    return result;
}
Tensor & CUDAShortType::pow_out(Tensor & result, Scalar base, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto base_ = base.toShort();
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",2, false);
    THCudaShortTensor_tpow(context->thc_state, result_->tensor, base_, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAShortType::pow(Scalar base, const Tensor & self) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto base_ = base.toShort();
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",2, false);
    THCudaShortTensor_tpow(context->thc_state, result_->tensor, base_, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAShortType::pow_(Tensor & self, Scalar exponent) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toShort();
    THCudaShortTensor_pow(context->thc_state, self_->tensor, self_->tensor, exponent_);
    return self;
}
Tensor & CUDAShortType::s_pow_(Tensor & self, const Tensor & exponent) const {
    if (exponent.dim() == 0) {
        return static_cast<const Type*>(this)->pow_(self, Scalar(exponent));
    }
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto exponent_ = checked_cast_tensor<CUDAShortTensor>(exponent.pImpl,"exponent",3, false);
    THCudaShortTensor_cpow(context->thc_state, self_->tensor, self_->tensor, exponent_->tensor);
    return self;
}
Tensor & CUDAShortType::zero_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    THCudaShortTensor_zero(context->thc_state, self_->tensor);
    return self;
}
Tensor CUDAShortType::_sumall(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<int16_t>(THCudaShortTensor_sumall(context->thc_state, self_->tensor)));
}
Tensor & CUDAShortType::_sum_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaShortTensor_sum(context->thc_state, result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CUDAShortType::_sum(const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaShortTensor_sum(context->thc_state, result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CUDAShortType::_prodall(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<int16_t>(THCudaShortTensor_prodall(context->thc_state, self_->tensor)));
}
Tensor & CUDAShortType::_prod_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaShortTensor_prod(context->thc_state, result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CUDAShortType::_prod(const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaShortTensor_prod(context->thc_state, result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor & CUDAShortType::cumsum_out(Tensor & result, const Tensor & self, int64_t dim) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaShortTensor_cumsum(context->thc_state, result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAShortType::cumsum(const Tensor & self, int64_t dim) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaShortTensor_cumsum(context->thc_state, result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAShortType::cumprod_out(Tensor & result, const Tensor & self, int64_t dim) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaShortTensor_cumprod(context->thc_state, result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAShortType::cumprod(const Tensor & self, int64_t dim) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaShortTensor_cumprod(context->thc_state, result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAShortType::sign_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    THCudaShortTensor_sign(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAShortType::sign(const Tensor & self) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    THCudaShortTensor_sign(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAShortType::sign_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    THCudaShortTensor_sign(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor CUDAShortType::trace(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<int16_t>(THCudaShortTensor_trace(context->thc_state, self_->tensor)));
}
Tensor & CUDAShortType::add_out(Tensor & result, const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    auto alpha_ = alpha.toShort();
    THCudaShortTensor_add_scaled(context->thc_state, result_->tensor, self_->tensor, other_, alpha_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAShortType::add(const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    auto alpha_ = alpha.toShort();
    THCudaShortTensor_add_scaled(context->thc_state, result_->tensor, self_->tensor, other_, alpha_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAShortType::s_add_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if(other.type().is_sparse()) {
        return static_cast<const Type*>(this)->add_out(result, self, SparseTensor(other), alpha);
    }
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toShort();
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",3, false);
    THCudaShortTensor_cadd(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAShortType::s_add(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if(other.type().is_sparse()) {
        return static_cast<const Type*>(this)->add(self, SparseTensor(other), alpha);
    }
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add(self, Scalar(other), alpha);
    }
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toShort();
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",3, false);
    THCudaShortTensor_cadd(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAShortType::add_out(Tensor & result, const Tensor & self, SparseTensor other, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toShort();
    auto other_ = checked_cast_tensor<SparseCUDAShortTensor>(other.tref.pImpl,"other",3,false);
    THCSShortTensor_spcadd(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAShortType::add(const Tensor & self, SparseTensor other, Scalar alpha) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toShort();
    auto other_ = checked_cast_tensor<SparseCUDAShortTensor>(other.tref.pImpl,"other",3,false);
    THCSShortTensor_spcadd(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAShortType::add_(Tensor & self, Scalar other, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    auto alpha_ = alpha.toShort();
    THCudaShortTensor_add_scaled(context->thc_state, self_->tensor, self_->tensor, other_, alpha_);
    return self;
}
Tensor & CUDAShortType::s_add_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if(other.type().is_sparse()) {
        return static_cast<const Type*>(this)->add_(self, SparseTensor(other), alpha);
    }
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toShort();
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",4, false);
    THCudaShortTensor_cadd(context->thc_state, self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & CUDAShortType::add_(Tensor & self, SparseTensor other, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toShort();
    auto other_ = checked_cast_tensor<SparseCUDAShortTensor>(other.tref.pImpl,"other",4,false);
    THCSShortTensor_spcadd(context->thc_state, self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & CUDAShortType::sub_out(Tensor & result, const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    auto alpha_ = alpha.toShort();
    THCudaShortTensor_sub_scaled(context->thc_state, result_->tensor, self_->tensor, other_, alpha_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAShortType::sub(const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    auto alpha_ = alpha.toShort();
    THCudaShortTensor_sub_scaled(context->thc_state, result_->tensor, self_->tensor, other_, alpha_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAShortType::s_sub_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toShort();
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",3, false);
    THCudaShortTensor_csub(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAShortType::s_sub(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub(self, Scalar(other), alpha);
    }
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toShort();
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",3, false);
    THCudaShortTensor_csub(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAShortType::sub_(Tensor & self, Scalar other, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    auto alpha_ = alpha.toShort();
    THCudaShortTensor_sub_scaled(context->thc_state, self_->tensor, self_->tensor, other_, alpha_);
    return self;
}
Tensor & CUDAShortType::s_sub_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toShort();
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",4, false);
    THCudaShortTensor_csub(context->thc_state, self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & CUDAShortType::mul_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_mul(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAShortType::mul(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_mul(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAShortType::s_mul_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",2, false);
    THCudaShortTensor_cmul(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAShortType::s_mul(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul(self, Scalar(other));
    }
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",2, false);
    THCudaShortTensor_cmul(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAShortType::mul_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_mul(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAShortType::s_mul_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",3, false);
    THCudaShortTensor_cmul(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAShortType::div_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_div(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAShortType::div(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_div(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAShortType::s_div_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->div_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",2, false);
    THCudaShortTensor_cdiv(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAShortType::s_div(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->div(self, Scalar(other));
    }
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",2, false);
    THCudaShortTensor_cdiv(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAShortType::div_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_div(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAShortType::s_div_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->div_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",3, false);
    THCudaShortTensor_cdiv(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAShortType::fmod_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_fmod(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAShortType::fmod(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_fmod(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAShortType::s_fmod_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->fmod_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",2, false);
    THCudaShortTensor_cfmod(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAShortType::s_fmod(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->fmod(self, Scalar(other));
    }
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",2, false);
    THCudaShortTensor_cfmod(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAShortType::fmod_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_fmod(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAShortType::s_fmod_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->fmod_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",3, false);
    THCudaShortTensor_cfmod(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAShortType::remainder_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_remainder(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAShortType::remainder(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_remainder(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAShortType::s_remainder_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->remainder_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",2, false);
    THCudaShortTensor_cremainder(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAShortType::s_remainder(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->remainder(self, Scalar(other));
    }
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",2, false);
    THCudaShortTensor_cremainder(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAShortType::remainder_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THCudaShortTensor_remainder(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAShortType::s_remainder_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->remainder_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",3, false);
    THCudaShortTensor_cremainder(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAShortType::clamp_out(Tensor & result, const Tensor & self, Scalar min, Scalar max) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toShort();
    auto max_ = max.toShort();
    THCudaShortTensor_clamp(context->thc_state, result_->tensor, self_->tensor, min_, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAShortType::clamp(const Tensor & self, Scalar min, Scalar max) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toShort();
    auto max_ = max.toShort();
    THCudaShortTensor_clamp(context->thc_state, result_->tensor, self_->tensor, min_, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAShortType::clamp_(Tensor & self, Scalar min, Scalar max) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toShort();
    auto max_ = max.toShort();
    THCudaShortTensor_clamp(context->thc_state, self_->tensor, self_->tensor, min_, max_);
    return self;
}
Tensor & CUDAShortType::clamp_min_out(Tensor & result, const Tensor & self, Scalar min) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toShort();
    THCudaShortTensor_cmaxValue(context->thc_state, result_->tensor, self_->tensor, min_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAShortType::clamp_min(const Tensor & self, Scalar min) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toShort();
    THCudaShortTensor_cmaxValue(context->thc_state, result_->tensor, self_->tensor, min_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAShortType::clamp_min_(Tensor & self, Scalar min) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toShort();
    THCudaShortTensor_cmaxValue(context->thc_state, self_->tensor, self_->tensor, min_);
    return self;
}
Tensor & CUDAShortType::clamp_max_out(Tensor & result, const Tensor & self, Scalar max) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto max_ = max.toShort();
    THCudaShortTensor_cminValue(context->thc_state, result_->tensor, self_->tensor, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAShortType::clamp_max(const Tensor & self, Scalar max) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto max_ = max.toShort();
    THCudaShortTensor_cminValue(context->thc_state, result_->tensor, self_->tensor, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAShortType::clamp_max_(Tensor & self, Scalar max) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto max_ = max.toShort();
    THCudaShortTensor_cminValue(context->thc_state, self_->tensor, self_->tensor, max_);
    return self;
}
Tensor & CUDAShortType::tril_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    THCudaShortTensor_tril(context->thc_state, result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAShortType::tril(const Tensor & self, int64_t diagonal) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    THCudaShortTensor_tril(context->thc_state, result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAShortType::tril_(Tensor & self, int64_t diagonal) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    THCudaShortTensor_tril(context->thc_state, self_->tensor, self_->tensor, diagonal);
    return self;
}
Tensor & CUDAShortType::triu_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    THCudaShortTensor_triu(context->thc_state, result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAShortType::triu(const Tensor & self, int64_t diagonal) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    THCudaShortTensor_triu(context->thc_state, result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAShortType::triu_(Tensor & self, int64_t diagonal) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    THCudaShortTensor_triu(context->thc_state, self_->tensor, self_->tensor, diagonal);
    return self;
}
Tensor & CUDAShortType::cross_out(Tensor & result, const Tensor & self, const Tensor & other, int64_t dim) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",2, false);
    THCudaShortTensor_cross(context->thc_state, result_->tensor, self_->tensor, other_->tensor, dim);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAShortType::cross(const Tensor & self, const Tensor & other, int64_t dim) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAShortTensor>(other.pImpl,"other",2, false);
    THCudaShortTensor_cross(context->thc_state, result_->tensor, self_->tensor, other_->tensor, dim);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAShortType::diag_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    if (self_->isScalar()) {
      throw std::runtime_error("Input must be 1-d or 2-d");
    }
    THCudaShortTensor_diag(context->thc_state, result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAShortType::diag(const Tensor & self, int64_t diagonal) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    if (self_->isScalar()) {
      throw std::runtime_error("Input must be 1-d or 2-d");
    }
    THCudaShortTensor_diag(context->thc_state, result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAShortType::s_addmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    if(mat1.type().is_sparse()) {
        return static_cast<const Type*>(this)->addmm_out(result, self, SparseTensor(mat1), mat2, beta, alpha);
    }
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toShort();
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toShort();
    auto mat1_ = checked_cast_tensor<CUDAShortTensor>(mat1.pImpl,"mat1",4, false);
    auto mat2_ = checked_cast_tensor<CUDAShortTensor>(mat2.pImpl,"mat2",5, false);
    THCudaShortTensor_addmm(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CUDAShortType::s_addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    if(mat1.type().is_sparse()) {
        return static_cast<const Type*>(this)->addmm(self, SparseTensor(mat1), mat2, beta, alpha);
    }
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toShort();
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toShort();
    auto mat1_ = checked_cast_tensor<CUDAShortTensor>(mat1.pImpl,"mat1",4, false);
    auto mat2_ = checked_cast_tensor<CUDAShortTensor>(mat2.pImpl,"mat2",5, false);
    THCudaShortTensor_addmm(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CUDAShortType::addmm_out(Tensor & result, const Tensor & self, SparseTensor mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toShort();
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toShort();
    auto mat1_ = checked_cast_tensor<SparseCUDAShortTensor>(mat1.tref.pImpl,"mat1",4,false);
    auto mat2_ = checked_cast_tensor<CUDAShortTensor>(mat2.pImpl,"mat2",5, false);
    THCSShortTensor_spaddmm(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CUDAShortType::addmm(const Tensor & self, SparseTensor mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toShort();
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toShort();
    auto mat1_ = checked_cast_tensor<SparseCUDAShortTensor>(mat1.tref.pImpl,"mat1",4,false);
    auto mat2_ = checked_cast_tensor<CUDAShortTensor>(mat2.pImpl,"mat2",5, false);
    THCSShortTensor_spaddmm(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CUDAShortType::addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    if(mat1.type().is_sparse()) {
        return static_cast<const Type*>(this)->addmm_(self, SparseTensor(mat1), mat2, beta, alpha);
    }
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toShort();
    auto alpha_ = alpha.toShort();
    auto mat1_ = checked_cast_tensor<CUDAShortTensor>(mat1.pImpl,"mat1",5, false);
    auto mat2_ = checked_cast_tensor<CUDAShortTensor>(mat2.pImpl,"mat2",6, false);
    THCudaShortTensor_addmm(context->thc_state, self_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    return self;
}
Tensor & CUDAShortType::addmm_(Tensor & self, SparseTensor mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toShort();
    auto alpha_ = alpha.toShort();
    auto mat1_ = checked_cast_tensor<SparseCUDAShortTensor>(mat1.tref.pImpl,"mat1",5,false);
    auto mat2_ = checked_cast_tensor<CUDAShortTensor>(mat2.pImpl,"mat2",6, false);
    THCSShortTensor_spaddmm(context->thc_state, self_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    return self;
}
Tensor & CUDAShortType::s__addmv_out(Tensor & result, const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toShort();
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toShort();
    auto mat_ = checked_cast_tensor<CUDAShortTensor>(mat.pImpl,"mat",4, false);
    auto vec_ = checked_cast_tensor<CUDAShortTensor>(vec.pImpl,"vec",5, false);
    THCudaShortTensor_addmv(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, mat_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && mat_->isScalar() && vec_->isScalar());
    return result;
}
Tensor CUDAShortType::s__addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toShort();
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toShort();
    auto mat_ = checked_cast_tensor<CUDAShortTensor>(mat.pImpl,"mat",4, false);
    auto vec_ = checked_cast_tensor<CUDAShortTensor>(vec.pImpl,"vec",5, false);
    THCudaShortTensor_addmv(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, mat_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && mat_->isScalar() && vec_->isScalar());
    return result;
}
Tensor & CUDAShortType::_addmv_(Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toShort();
    auto alpha_ = alpha.toShort();
    auto mat_ = checked_cast_tensor<CUDAShortTensor>(mat.pImpl,"mat",5, false);
    auto vec_ = checked_cast_tensor<CUDAShortTensor>(vec.pImpl,"vec",6, false);
    THCudaShortTensor_addmv(context->thc_state, self_->tensor, beta_, self_->tensor, alpha_, mat_->tensor, vec_->tensor);
    return self;
}
Tensor & CUDAShortType::s__addr_out(Tensor & result, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toShort();
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toShort();
    auto vec1_ = checked_cast_tensor<CUDAShortTensor>(vec1.pImpl,"vec1",4, false);
    auto vec2_ = checked_cast_tensor<CUDAShortTensor>(vec2.pImpl,"vec2",5, false);
    THCudaShortTensor_addr(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, vec1_->tensor, vec2_->tensor);
    result_->maybeScalar(self_->isScalar() && vec1_->isScalar() && vec2_->isScalar());
    return result;
}
Tensor CUDAShortType::s__addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toShort();
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toShort();
    auto vec1_ = checked_cast_tensor<CUDAShortTensor>(vec1.pImpl,"vec1",4, false);
    auto vec2_ = checked_cast_tensor<CUDAShortTensor>(vec2.pImpl,"vec2",5, false);
    THCudaShortTensor_addr(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, vec1_->tensor, vec2_->tensor);
    result_->maybeScalar(self_->isScalar() && vec1_->isScalar() && vec2_->isScalar());
    return result;
}
Tensor & CUDAShortType::_addr_(Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toShort();
    auto alpha_ = alpha.toShort();
    auto vec1_ = checked_cast_tensor<CUDAShortTensor>(vec1.pImpl,"vec1",5, false);
    auto vec2_ = checked_cast_tensor<CUDAShortTensor>(vec2.pImpl,"vec2",6, false);
    THCudaShortTensor_addr(context->thc_state, self_->tensor, beta_, self_->tensor, alpha_, vec1_->tensor, vec2_->tensor);
    return self;
}
Tensor & CUDAShortType::_ger_out(Tensor & result, const Tensor & self, const Tensor & vec2) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.dim() == 0 ? 1 : self.size(0),vec2.dim() == 0 ? 1 : vec2.size(0) });
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto vec2_ = checked_cast_tensor<CUDAShortTensor>(vec2.pImpl,"vec2",2, false);
    THCudaShortTensor_addr(context->thc_state, result_->tensor, int16_t(0), result_->tensor, int16_t(1), self_->tensor, vec2_->tensor);
    result_->maybeScalar(false);
    return result;
}
Tensor CUDAShortType::_ger(const Tensor & self, const Tensor & vec2) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.dim() == 0 ? 1 : self.size(0),vec2.dim() == 0 ? 1 : vec2.size(0) });
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto vec2_ = checked_cast_tensor<CUDAShortTensor>(vec2.pImpl,"vec2",2, false);
    THCudaShortTensor_addr(context->thc_state, result_->tensor, int16_t(0), result_->tensor, int16_t(1), self_->tensor, vec2_->tensor);
    result_->maybeScalar(false);
    return result;
}
Tensor & CUDAShortType::_mv_out(Tensor & result, const Tensor & self, const Tensor & vec) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.size(0) });
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto vec_ = checked_cast_tensor<CUDAShortTensor>(vec.pImpl,"vec",2, false);
    THCudaShortTensor_addmv(context->thc_state, result_->tensor, int16_t(0), result_->tensor, int16_t(1), self_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && vec_->isScalar());
    return result;
}
Tensor CUDAShortType::_mv(const Tensor & self, const Tensor & vec) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.size(0) });
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto vec_ = checked_cast_tensor<CUDAShortTensor>(vec.pImpl,"vec",2, false);
    THCudaShortTensor_addmv(context->thc_state, result_->tensor, int16_t(0), result_->tensor, int16_t(1), self_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && vec_->isScalar());
    return result;
}
Tensor & CUDAShortType::_mm_out(Tensor & result, const Tensor & self, const Tensor & mat2) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.size(0),mat2.size(1) });
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CUDAShortTensor>(mat2.pImpl,"mat2",2, false);
    THCudaShortTensor_addmm(context->thc_state, result_->tensor, int16_t(0), result_->tensor, int16_t(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CUDAShortType::_mm(const Tensor & self, const Tensor & mat2) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.size(0),mat2.size(1) });
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CUDAShortTensor>(mat2.pImpl,"mat2",2, false);
    THCudaShortTensor_addmm(context->thc_state, result_->tensor, int16_t(0), result_->tensor, int16_t(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CUDAShortType::bmm_out(Tensor & result, const Tensor & self, const Tensor & mat2) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.size(0),self.size(1),mat2.size(2) });
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CUDAShortTensor>(mat2.pImpl,"mat2",2, false);
    THCudaShortTensor_baddbmm(context->thc_state, result_->tensor, int16_t(0), result_->tensor, int16_t(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CUDAShortType::bmm(const Tensor & self, const Tensor & mat2) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.size(0),self.size(1),mat2.size(2) });
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CUDAShortTensor>(mat2.pImpl,"mat2",2, false);
    THCudaShortTensor_baddbmm(context->thc_state, result_->tensor, int16_t(0), result_->tensor, int16_t(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CUDAShortType::s_addbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toShort();
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toShort();
    auto batch1_ = checked_cast_tensor<CUDAShortTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CUDAShortTensor>(batch2.pImpl,"batch2",5, false);
    THCudaShortTensor_addbmm(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor CUDAShortType::s_addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toShort();
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toShort();
    auto batch1_ = checked_cast_tensor<CUDAShortTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CUDAShortTensor>(batch2.pImpl,"batch2",5, false);
    THCudaShortTensor_addbmm(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor & CUDAShortType::addbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toShort();
    auto alpha_ = alpha.toShort();
    auto batch1_ = checked_cast_tensor<CUDAShortTensor>(batch1.pImpl,"batch1",5, false);
    auto batch2_ = checked_cast_tensor<CUDAShortTensor>(batch2.pImpl,"batch2",6, false);
    THCudaShortTensor_addbmm(context->thc_state, self_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    return self;
}
Tensor & CUDAShortType::s_baddbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toShort();
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toShort();
    auto batch1_ = checked_cast_tensor<CUDAShortTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CUDAShortTensor>(batch2.pImpl,"batch2",5, false);
    THCudaShortTensor_baddbmm(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor CUDAShortType::s_baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toShort();
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toShort();
    auto batch1_ = checked_cast_tensor<CUDAShortTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CUDAShortTensor>(batch2.pImpl,"batch2",5, false);
    THCudaShortTensor_baddbmm(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor & CUDAShortType::baddbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toShort();
    auto alpha_ = alpha.toShort();
    auto batch1_ = checked_cast_tensor<CUDAShortTensor>(batch1.pImpl,"batch1",5, false);
    auto batch2_ = checked_cast_tensor<CUDAShortTensor>(batch2.pImpl,"batch2",6, false);
    THCudaShortTensor_baddbmm(context->thc_state, self_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    return self;
}
Tensor & CUDAShortType::s_addcmul_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toShort();
    auto tensor1_ = checked_cast_tensor<CUDAShortTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CUDAShortTensor>(tensor2.pImpl,"tensor2",4, false);
    THCudaShortTensor_addcmul(context->thc_state, result_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor CUDAShortType::s_addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toShort();
    auto tensor1_ = checked_cast_tensor<CUDAShortTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CUDAShortTensor>(tensor2.pImpl,"tensor2",4, false);
    THCudaShortTensor_addcmul(context->thc_state, result_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor & CUDAShortType::s_addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toShort();
    auto tensor1_ = checked_cast_tensor<CUDAShortTensor>(tensor1.pImpl,"tensor1",4, false);
    auto tensor2_ = checked_cast_tensor<CUDAShortTensor>(tensor2.pImpl,"tensor2",5, false);
    THCudaShortTensor_addcmul(context->thc_state, self_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    return self;
}
Tensor & CUDAShortType::s_addcdiv_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toShort();
    auto tensor1_ = checked_cast_tensor<CUDAShortTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CUDAShortTensor>(tensor2.pImpl,"tensor2",4, false);
    THCudaShortTensor_addcdiv(context->thc_state, result_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor CUDAShortType::s_addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toShort();
    auto tensor1_ = checked_cast_tensor<CUDAShortTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CUDAShortTensor>(tensor2.pImpl,"tensor2",4, false);
    THCudaShortTensor_addcdiv(context->thc_state, result_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor & CUDAShortType::s_addcdiv_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toShort();
    auto tensor1_ = checked_cast_tensor<CUDAShortTensor>(tensor1.pImpl,"tensor1",4, false);
    auto tensor2_ = checked_cast_tensor<CUDAShortTensor>(tensor2.pImpl,"tensor2",5, false);
    THCudaShortTensor_addcdiv(context->thc_state, self_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    return self;
}
Tensor & CUDAShortType::random_(Tensor & self, int64_t from, int64_t to, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    THCudaShortTensor_clampedRandom(context->thc_state, self_->tensor, from, to);
    return self;
}
Tensor & CUDAShortType::random_(Tensor & self, int64_t to, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    THCudaShortTensor_cappedRandom(context->thc_state, self_->tensor, to);
    return self;
}
Tensor & CUDAShortType::random_(Tensor & self, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    THCudaShortTensor_random(context->thc_state, self_->tensor);
    return self;
}
Tensor & CUDAShortType::geometric_(Tensor & self, double p, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    THCudaShortTensor_geometric(context->thc_state, self_->tensor, p);
    return self;
}
Tensor CUDAShortType::tensor(Storage & storage, int64_t storageOffset, IntList size, IntList stride) const {
    auto storage_ = checked_cast_storage<CUDAShortStorage>(&storage,"storage",1);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    return Tensor((new CUDAShortTensor(context, THCudaShortTensor_newWithStorage(context->thc_state, storage_->storage, storageOffset, size_, stride_)))->maybeScalar(size.size() == 0),false);
}
Tensor CUDAShortType::tensor(IntList size) const {
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new CUDAShortTensor(context, THCudaShortTensor_newWithSize(context->thc_state, size_, NULL)))->maybeScalar(size.size() == 0),false);
}
Tensor CUDAShortType::tensor(IntList size, IntList stride) const {
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    return Tensor((new CUDAShortTensor(context, THCudaShortTensor_newWithSize(context->thc_state, size_, stride_)))->maybeScalar(size.size() == 0),false);
}
Tensor CUDAShortType::tensor() const {
    return Tensor((new CUDAShortTensor(context, THCudaShortTensor_new(context->thc_state))),false);
}
Tensor CUDAShortType::alias(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    return Tensor((new CUDAShortTensor(context, THCudaShortTensor_newWithTensor(context->thc_state, self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor & CUDAShortType::_copy_ignoring_overlaps_(Tensor & self, const Tensor & src) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto src_ = checked_cast_tensor<CUDAShortTensor>(src.pImpl,"src",2, false);
    THCudaShortTensor_copyIgnoringOverlaps(context->thc_state, self_->tensor, src_->tensor);
    return self;
}
Tensor & CUDAShortType::as_strided_out(Tensor & result, const Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    auto result_ = checked_cast_tensor<CUDAShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    if (storage_offset == -1) {
      storage_offset = self_->tensor->storageOffset;
    }
    THCudaShortTensor_setStorage(context->thc_state, result_->tensor, self_->tensor->storage, storage_offset, size_, stride_);
    result_->maybeScalar(size.size() == 0);
    return result;
}
Tensor CUDAShortType::as_strided(const Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    auto result_ = new CUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    if (storage_offset == -1) {
      storage_offset = self_->tensor->storageOffset;
    }
    THCudaShortTensor_setStorage(context->thc_state, result_->tensor, self_->tensor->storage, storage_offset, size_, stride_);
    result_->maybeScalar(size.size() == 0);
    return result;
}
Tensor & CUDAShortType::as_strided_(Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    if (storage_offset == -1) {
      storage_offset = self_->tensor->storageOffset;
    }
    THCudaShortTensor_setStorage(context->thc_state, self_->tensor, self_->tensor->storage, storage_offset, size_, stride_);
    self_->maybeScalar(size.size() == 0);
    return self;
}
Tensor & CUDAShortType::_cat_out(Tensor & self, TensorList tensors, int64_t dim) const {
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",0, false);
    auto tensors_ = tensor_list_checked_cast<CUDAShortTensor, Tensor, THCudaShortTensor>(tensors,"tensors",1);
    THCudaShortTensor_catArray(context->thc_state, self_->tensor, tensors_.data(), tensors_.size(), dim);
    return self;
}
Tensor CUDAShortType::_cat(TensorList tensors, int64_t dim) const {
    auto self_ = new CUDAShortTensor(context);
    auto self = Tensor(self_, false);
    auto tensors_ = tensor_list_checked_cast<CUDAShortTensor, Tensor, THCudaShortTensor>(tensors,"tensors",1);
    THCudaShortTensor_catArray(context->thc_state, self_->tensor, tensors_.data(), tensors_.size(), dim);
    return self;
}
Tensor CUDAShortType::_sparse_mask(const Tensor & self, SparseTensor mask) const {
    auto result_ = new SparseCUDAShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAShortTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<SparseCUDAShortTensor>(mask.tref.pImpl,"mask",2,false);
    THCudaShortTensor_sparseMask(context->thc_state, result_->tensor, self_->tensor, mask_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAShortType::abs_out(Tensor & result, const Tensor & self) const {
    return  at::native::_abs_out_cuda(result, self);
}
Tensor & CUDAShortType::ceil_out(Tensor & result, const Tensor & self) const {
    return  at::native::_ceil_out_cuda(result, self);
}
Tensor & CUDAShortType::cos_out(Tensor & result, const Tensor & self) const {
    return  at::native::_cos_out_cuda(result, self);
}
Tensor CUDAShortType::embedding_dense_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) const {
    return  at::native::embedding_backward_cuda(grad, indices, num_weights, padding_idx, scale_grad_by_freq);
}
Tensor & CUDAShortType::embedding_renorm_(Tensor & self, const Tensor & indices, double max_norm, double norm_type) const {
    return  at::native::embedding_renorm_cuda_(self, indices, max_norm, norm_type);
}
std::tuple<Tensor,Tensor,Tensor> CUDAShortType::embedding_bag(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse) const {
    return  at::native::embedding_bag_cuda(weight, indices, offsets, scale_grad_by_freq, mode, sparse);
}
Tensor CUDAShortType::embedding_bag_dense_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode) const {
    return  at::native::embedding_bag_backward_cuda(grad, indices, offsets, offset2bag, bag_size, num_weights, scale_grad_by_freq, mode);
}
Tensor & CUDAShortType::exp_out(Tensor & result, const Tensor & self) const {
    return  at::native::_exp_out_cuda(result, self);
}
Tensor & CUDAShortType::eye_out(Tensor & result, int64_t n, int64_t m) const {
    return  at::native::eye_out_cuda(result, n, m);
}
Tensor & CUDAShortType::floor_out(Tensor & result, const Tensor & self) const {
    return  at::native::_floor_out_cuda(result, self);
}
Tensor CUDAShortType::_fft_with_size(const Tensor & self, int64_t signal_ndim, bool complex_input, bool complex_output, bool inverse, IntList checked_signal_sizes, bool normalized, bool onesided, IntList output_sizes) const {
    return  at::native::_fft_cufft(self, signal_ndim, complex_input, complex_output, inverse, checked_signal_sizes, normalized, onesided, output_sizes);
}
Tensor & CUDAShortType::log_out(Tensor & result, const Tensor & self) const {
    return  at::native::_log_out_cuda(result, self);
}
std::tuple<Tensor,Tensor> CUDAShortType::RoiPooling2d_forward(const Tensor & input, const Tensor & rois, int64_t pooledHeight, int64_t pooledWidth, double spatialScale) const {
    return  at::native::RoiPooling2d_forward_cuda(input, rois, pooledHeight, pooledWidth, spatialScale);
}
Tensor CUDAShortType::RoiPooling2d_backward(const Tensor & input, const Tensor & rois, int64_t pooledHeight, int64_t pooledWidth, double spatialScale, const Tensor & gradOutput, const Tensor & argmaxes) const {
    return  at::native::RoiPooling2d_backward_cuda(input, rois, pooledHeight, pooledWidth, spatialScale, gradOutput, argmaxes);
}
Tensor & CUDAShortType::round_out(Tensor & result, const Tensor & self) const {
    return  at::native::_round_out_cuda(result, self);
}
Tensor & CUDAShortType::sin_out(Tensor & result, const Tensor & self) const {
    return  at::native::_sin_out_cuda(result, self);
}
Tensor & CUDAShortType::sspaddmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return  at::native::_sspaddmm_out_only_sparse(result, self, mat1, mat2, beta, alpha);
}
Tensor CUDAShortType::sum(const Tensor & self) const {
    return  at::native::_sum_cuda(self);
}
Tensor & CUDAShortType::sum_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    return  at::native::_sum_out_cuda(result, self, dim, keepdim);
}
Tensor & CUDAShortType::sqrt_out(Tensor & result, const Tensor & self) const {
    return  at::native::_sqrt_out_cuda(result, self);
}
Tensor CUDAShortType::prod(const Tensor & self) const {
    return  at::native::_prod_cuda(self);
}
Tensor & CUDAShortType::prod_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    return  at::native::_prod_out_cuda(result, self, dim, keepdim);
}
Tensor & CUDAShortType::trunc_out(Tensor & result, const Tensor & self) const {
    return  at::native::_trunc_out_cuda(result, self);
}
std::tuple<Tensor,Tensor> CUDAShortType::_unique(const Tensor & self, bool sorted, bool return_inverse) const {
    return  at::native::_unique_cuda(self, sorted, return_inverse);
}
Tensor CUDAShortType::_s_where(const Tensor & condition, const Tensor & self, const Tensor & other) const {
    return  at::native::_s_where_cuda(condition, self, other);
}
Tensor CUDAShortType::_standard_gamma_grad(const Tensor & self, const Tensor & output) const {
    return  at::native::_standard_gamma_grad_cuda(self, output);
}
Tensor CUDAShortType::poisson(const Tensor & self, Generator * generator) const {
    return  at::native::_s_poisson_cuda(self, generator);
}

}
