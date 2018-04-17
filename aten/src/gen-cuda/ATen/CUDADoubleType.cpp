// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/CUDADoubleType.h"
#include "ATen/CUDADoubleStorage.h"
#include "ATen/CUDADoubleTensor.h"
#include "ATen/CUDAGenerator.h"
#include "ATen/CUDAByteTensor.h"
#include "ATen/CUDAIntTensor.h"
#include "ATen/CUDALongTensor.h"
#include "ATen/SparseCUDADoubleTensor.h"
#include "ATen/CUDADoubleTensor.h"
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

CUDADoubleType::CUDADoubleType(Context* context)
: Type(context, /*is_variable_or_undefined=*/false) {}
ScalarType CUDADoubleType::scalarType() const {
  return ScalarType::Double;
}
Backend CUDADoubleType::backend() const {
  return Backend::CUDA;
}
bool CUDADoubleType::is_cuda() const { return backend() == kCUDA || backend() == kSparseCUDA; }
bool CUDADoubleType::is_sparse() const { return backend() == kSparseCPU || backend() == kSparseCUDA; }
bool CUDADoubleType::is_distributed() const { return false; }

std::unique_ptr<Storage> CUDADoubleType::storage() const {
  return std::unique_ptr<Storage>(new CUDADoubleStorage(context));
}
std::unique_ptr<Storage> CUDADoubleType::storage(size_t size) const {
  return std::unique_ptr<Storage>(new CUDADoubleStorage(context,size));
}
std::unique_ptr<Storage> CUDADoubleType::storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const {
    return std::unique_ptr<Storage>(
      new CUDADoubleStorage(context,data,size,deleter));
}
std::unique_ptr<Storage> CUDADoubleType::storageWithAllocator(int64_t size, std::unique_ptr<Allocator> allocator) const {
    return std::unique_ptr<Storage>(
        new CUDADoubleStorage(context, size, std::move(allocator)));
}
Tensor CUDADoubleType::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THCudaDoubleTensor_retain(context->thc_state,  (THCudaDoubleTensor*) th_pointer);
  return Tensor(new CUDADoubleTensor(context,(THCudaDoubleTensor*)(th_pointer)), false);
}
std::unique_ptr<Storage> CUDADoubleType::unsafeStorageFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THCudaDoubleStorage_retain(context->thc_state,  (THCudaDoubleStorage*) th_pointer);
  return std::unique_ptr<Storage>(new CUDADoubleStorage(context, (THCudaDoubleStorage*) th_pointer));
}
std::unique_ptr<Generator> CUDADoubleType::generator() const {
  return std::unique_ptr<Generator>(new CUDAGenerator(context));
}

const char * CUDADoubleType::toString() const {
  return CUDADoubleType::typeString();
}
TypeID CUDADoubleType::ID() const {
  return TypeID::CUDADouble;
}

std::size_t CUDADoubleType::elementSizeInBytes() const {
  return sizeof(double);
}

const char * CUDADoubleType::typeString() {
  return "CUDADoubleType";
}

/* example
Tensor * CUDADoubleType::add(Tensor & a, Tensor & b) {
  std::cout << "add CUDADoubleTensor\n";
  return &a;
}
*/

int64_t CUDADoubleType::storage_offset(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    return static_cast<int64_t>(THCudaDoubleTensor_storageOffset(context->thc_state, self_->tensor));
}
Tensor & CUDADoubleType::resize_(Tensor & self, IntList size) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    THCudaDoubleTensor_resize(context->thc_state, self_->tensor, size_, NULL);
    self_->maybeScalar(size.size() == 0);
    return self;
}
int64_t CUDADoubleType::numel(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    return static_cast<int64_t>(THCudaDoubleTensor_nElement(context->thc_state, self_->tensor));
}
Tensor & CUDADoubleType::set_(Tensor & self, Storage & storage) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto storage_ = checked_cast_storage<CUDADoubleStorage>(&storage,"storage",2);
    THCudaDoubleTensor_setStorage(context->thc_state, self_->tensor, storage_->storage, 0, THLongStorageView::makeFromLength(static_cast<int64_t>(storage.size())), NULL);
    self_->maybeScalar(false);
    return self;
}
Tensor & CUDADoubleType::set_(Tensor & self, Storage & sourceStorage, int64_t storage_offset, IntList size, IntList stride) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto sourceStorage_ = checked_cast_storage<CUDADoubleStorage>(&sourceStorage,"sourceStorage",2);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    THCudaDoubleTensor_setStorage(context->thc_state, self_->tensor, sourceStorage_->storage, storage_offset, size_, stride_);
    self_->maybeScalar(size.size() == 0);
    return self;
}
Tensor & CUDADoubleType::set_(Tensor & self, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto source_ = checked_cast_tensor<CUDADoubleTensor>(source.pImpl,"source",2, false);
    THCudaDoubleTensor_set(context->thc_state, self_->tensor, source_->tensor);
    self_->maybeScalar(source_->isScalar());
    return self;
}
Tensor & CUDADoubleType::set_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_setStorage(context->thc_state, self_->tensor, NULL, 0, NULL, NULL);
    self_->maybeScalar(false);
    return self;
}
Tensor & CUDADoubleType::fill_(Tensor & self, Scalar value) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toDouble();
    THCudaDoubleTensor_fill(context->thc_state, self_->tensor, value_);
    return self;
}
Tensor & CUDADoubleType::fill_(Tensor & self, const Tensor & value) const {
    if (value.dim() == 0) {
        return static_cast<const Type*>(this)->fill_(self, Scalar(value));
    }
    AT_ERROR("fill_ only supports a 0-dimensional value tensor, but got tensor "
        "with %" PRId64 " dimension(s)", value.dim());
}
bool CUDADoubleType::is_contiguous(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    return THCudaDoubleTensor_isContiguous(context->thc_state, self_->tensor);
}
bool CUDADoubleType::is_set_to(const Tensor & self, const Tensor & tensor) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto tensor_ = checked_cast_tensor<CUDADoubleTensor>(tensor.pImpl,"tensor",2, false);
    return THCudaDoubleTensor_isSetTo(context->thc_state, self_->tensor, tensor_->tensor);
}
Tensor & CUDADoubleType::s_masked_fill_(Tensor & self, const Tensor & mask, Scalar value) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CUDAByteTensor>(mask.pImpl,"mask",2, false);
    auto value_ = value.toDouble();
    THCudaDoubleTensor_maskedFill(context->thc_state, self_->tensor, mask_->tensor, value_);
    return self;
}
Tensor & CUDADoubleType::s_masked_fill_(Tensor & self, const Tensor & mask, const Tensor & value) const {
    if (value.dim() == 0) {
        return static_cast<const Type*>(this)->masked_fill_(self, mask, Scalar(value));
    }
    AT_ERROR("masked_fill_ only supports a 0-dimensional value tensor, but got tensor "
        "with %" PRId64 " dimension(s)", value.dim());
}
Tensor & CUDADoubleType::s_masked_scatter_(Tensor & self, const Tensor & mask, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CUDAByteTensor>(mask.pImpl,"mask",2, false);
    auto source_ = checked_cast_tensor<CUDADoubleTensor>(source.pImpl,"source",3, false);
    THCudaDoubleTensor_maskedCopy(context->thc_state, self_->tensor, mask_->tensor, source_->tensor);
    return self;
}
Tensor & CUDADoubleType::s_masked_select_out(Tensor & result, const Tensor & self, const Tensor & mask) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CUDAByteTensor>(mask.pImpl,"mask",2, false);
    THCudaDoubleTensor_maskedSelect(context->thc_state, result_->tensor, self_->tensor, mask_->tensor);
    result_->maybeScalar(self_->isScalar() && mask_->isScalar());
    return result;
}
Tensor CUDADoubleType::s_masked_select(const Tensor & self, const Tensor & mask) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CUDAByteTensor>(mask.pImpl,"mask",2, false);
    THCudaDoubleTensor_maskedSelect(context->thc_state, result_->tensor, self_->tensor, mask_->tensor);
    result_->maybeScalar(self_->isScalar() && mask_->isScalar());
    return result;
}
Tensor CUDADoubleType::transpose(const Tensor & self, int64_t dim0, int64_t dim1) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim0 = maybe_wrap_dim(dim0, self_);
    dim1 = maybe_wrap_dim(dim1, self_);
    return Tensor((new CUDADoubleTensor(context, THCudaDoubleTensor_newTranspose(context->thc_state, self_->tensor, dim0, dim1)))->maybeScalar(self_->isScalar()),false);
}
Tensor CUDADoubleType::t(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    return Tensor((new CUDADoubleTensor(context, THCudaDoubleTensor_newTranspose(context->thc_state, self_->tensor, 0, 1)))->maybeScalar(self_->isScalar()),false);
}
Tensor & CUDADoubleType::nonzero_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDALongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_nonzero(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::nonzero(const Tensor & self) const {
    auto result_ = new CUDALongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_nonzero(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::contiguous(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    return Tensor((new CUDADoubleTensor(context, THCudaDoubleTensor_newContiguous(context->thc_state, self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor CUDADoubleType::clone(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    return Tensor((new CUDADoubleTensor(context, THCudaDoubleTensor_newClone(context->thc_state, self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor CUDADoubleType::view(const Tensor & self, IntList size) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new CUDADoubleTensor(context, THCudaDoubleTensor_newView(context->thc_state, self_->tensor, size_)))->maybeScalar(size.size() == 0),false);
}
Tensor & CUDADoubleType::resize_as_(Tensor & self, const Tensor & the_template) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto the_template_ = checked_cast_tensor<CUDADoubleTensor>(the_template.pImpl,"the_template",2, false);
    THCudaDoubleTensor_resizeAs(context->thc_state, self_->tensor, the_template_->tensor);
    self_->maybeScalar(the_template_->isScalar());
    return self;
}
Tensor & CUDADoubleType::index_select_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    THCudaDoubleTensor_indexSelect(context->thc_state, result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
Tensor CUDADoubleType::index_select(const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    THCudaDoubleTensor_indexSelect(context->thc_state, result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
Tensor & CUDADoubleType::_indexCopy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    auto source_ = checked_cast_tensor<CUDADoubleTensor>(source.pImpl,"source",4, false);
    THCudaDoubleTensor_indexCopy(context->thc_state, self_->tensor, dim, index_->tensor, source_->tensor);
    return self;
}
Tensor & CUDADoubleType::take_out(Tensor & result, const Tensor & self, const Tensor & index) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",2, false);
    THCudaDoubleTensor_take(context->thc_state, result_->tensor, self_->tensor, index_->tensor);
    result_->maybeScalar(index_->isScalar());
    return result;
}
Tensor CUDADoubleType::take(const Tensor & self, const Tensor & index) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",2, false);
    THCudaDoubleTensor_take(context->thc_state, result_->tensor, self_->tensor, index_->tensor);
    result_->maybeScalar(index_->isScalar());
    return result;
}
Tensor & CUDADoubleType::put_(Tensor & self, const Tensor & index, const Tensor & source, bool accumulate) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",2, false);
    auto source_ = checked_cast_tensor<CUDADoubleTensor>(source.pImpl,"source",3, false);
    THCudaDoubleTensor_put(context->thc_state, self_->tensor, index_->tensor, source_->tensor, accumulate);
    return self;
}
Tensor & CUDADoubleType::index_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    auto source_ = checked_cast_tensor<CUDADoubleTensor>(source.pImpl,"source",4, false);
    THCudaDoubleTensor_indexAdd(context->thc_state, self_->tensor, dim, index_->tensor, source_->tensor);
    return self;
}
Tensor & CUDADoubleType::index_fill_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    auto value_ = value.toDouble();
    THCudaDoubleTensor_indexFill(context->thc_state, self_->tensor, dim, index_->tensor, value_);
    return self;
}
Tensor & CUDADoubleType::index_fill_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & value) const {
    if (value.dim() == 0) {
        return static_cast<const Type*>(this)->index_fill_(self, dim, index, Scalar(value));
    }
    AT_ERROR("index_fill_ only supports a 0-dimensional value tensor, but got tensor "
        "with %" PRId64 " dimension(s)", value.dim());
}
Tensor CUDADoubleType::unfold(const Tensor & self, int64_t dimension, int64_t size, int64_t step) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dimension = maybe_wrap_dim(dimension, self_);
    THCudaDoubleTensor_unfold(context->thc_state, result_->tensor, self_->tensor, dimension, size, step);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::_range_out(Tensor & result, Scalar start, Scalar end, Scalar step) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto start_ = start.toDouble();
    auto end_ = end.toDouble();
    auto step_ = step.toDouble();
    THCudaDoubleTensor_range(context->thc_state, result_->tensor, start_, end_, step_);
    return result;
}
Tensor CUDADoubleType::_range(Scalar start, Scalar end, Scalar step) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto start_ = start.toDouble();
    auto end_ = end.toDouble();
    auto step_ = step.toDouble();
    THCudaDoubleTensor_range(context->thc_state, result_->tensor, start_, end_, step_);
    return result;
}
Tensor & CUDADoubleType::_arange_out(Tensor & result, Scalar start, Scalar end, Scalar step) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto start_ = start.toDouble();
    auto end_ = end.toDouble();
    auto step_ = step.toDouble();
    THCudaDoubleTensor_arange(context->thc_state, result_->tensor, start_, end_, step_);
    return result;
}
Tensor CUDADoubleType::_arange(Scalar start, Scalar end, Scalar step) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto start_ = start.toDouble();
    auto end_ = end.toDouble();
    auto step_ = step.toDouble();
    THCudaDoubleTensor_arange(context->thc_state, result_->tensor, start_, end_, step_);
    return result;
}
Tensor & CUDADoubleType::_arange_out(Tensor & result, Scalar end) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto end_ = end.toDouble();
    THCudaDoubleTensor_arange(context->thc_state, result_->tensor, 0, end_, 1);
    return result;
}
Tensor CUDADoubleType::_arange(Scalar end) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto end_ = end.toDouble();
    THCudaDoubleTensor_arange(context->thc_state, result_->tensor, 0, end_, 1);
    return result;
}
Tensor & CUDADoubleType::scatter_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
    if (src.dim() == 0) {
        return static_cast<const Type*>(this)->scatter_(self, dim, index, Scalar(src));
    }
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    auto src_ = checked_cast_tensor<CUDADoubleTensor>(src.pImpl,"src",4, false);
    THCudaDoubleTensor_scatter(context->thc_state, self_->tensor, dim, index_->tensor, src_->tensor);
    return self;
}
Tensor & CUDADoubleType::scatter_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    auto value_ = value.toDouble();
    THCudaDoubleTensor_scatterFill(context->thc_state, self_->tensor, dim, index_->tensor, value_);
    return self;
}
Tensor & CUDADoubleType::scatter_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    auto src_ = checked_cast_tensor<CUDADoubleTensor>(src.pImpl,"src",4, false);
    THCudaDoubleTensor_scatterAdd(context->thc_state, self_->tensor, dim, index_->tensor, src_->tensor);
    return self;
}
Tensor & CUDADoubleType::gather_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    result.resize_(index.sizes());
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    THCudaDoubleTensor_gather(context->thc_state, result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
Tensor CUDADoubleType::gather(const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    result.resize_(index.sizes());
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    THCudaDoubleTensor_gather(context->thc_state, result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
void* CUDADoubleType::data_ptr(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    return THCudaDoubleTensor_data(context->thc_state, self_->tensor);
}
bool CUDADoubleType::equal(const Tensor & self, const Tensor & other) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",2, false);
    return THCudaDoubleTensor_equal(context->thc_state, self_->tensor, other_->tensor);
}
Tensor & CUDADoubleType::__and___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_bitand(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::__and__(const Tensor & self, Scalar other) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_bitand(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::s___and___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__and___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",2, false);
    THCudaDoubleTensor_cbitand(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDADoubleType::s___and__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__and__(self, Scalar(other));
    }
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",2, false);
    THCudaDoubleTensor_cbitand(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDADoubleType::__iand__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_bitand(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDADoubleType::s___iand__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__iand__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",3, false);
    THCudaDoubleTensor_cbitand(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDADoubleType::__or___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_bitor(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::__or__(const Tensor & self, Scalar other) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_bitor(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::s___or___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__or___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",2, false);
    THCudaDoubleTensor_cbitor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDADoubleType::s___or__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__or__(self, Scalar(other));
    }
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",2, false);
    THCudaDoubleTensor_cbitor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDADoubleType::__ior__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_bitor(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDADoubleType::s___ior__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__ior__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",3, false);
    THCudaDoubleTensor_cbitor(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDADoubleType::__xor___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_bitxor(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::__xor__(const Tensor & self, Scalar other) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_bitxor(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::s___xor___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__xor___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",2, false);
    THCudaDoubleTensor_cbitxor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDADoubleType::s___xor__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__xor__(self, Scalar(other));
    }
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",2, false);
    THCudaDoubleTensor_cbitxor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDADoubleType::__ixor__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_bitxor(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDADoubleType::s___ixor__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__ixor__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",3, false);
    THCudaDoubleTensor_cbitxor(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDADoubleType::__lshift___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_lshift(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::__lshift__(const Tensor & self, Scalar other) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_lshift(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::s___lshift___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__lshift___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",2, false);
    THCudaDoubleTensor_clshift(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDADoubleType::s___lshift__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__lshift__(self, Scalar(other));
    }
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",2, false);
    THCudaDoubleTensor_clshift(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDADoubleType::__ilshift__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_lshift(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDADoubleType::s___ilshift__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__ilshift__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",3, false);
    THCudaDoubleTensor_clshift(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDADoubleType::__rshift___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_rshift(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::__rshift__(const Tensor & self, Scalar other) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_rshift(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::s___rshift___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__rshift___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",2, false);
    THCudaDoubleTensor_crshift(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDADoubleType::s___rshift__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__rshift__(self, Scalar(other));
    }
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",2, false);
    THCudaDoubleTensor_crshift(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDADoubleType::__irshift__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_rshift(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDADoubleType::s___irshift__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__irshift__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",3, false);
    THCudaDoubleTensor_crshift(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDADoubleType::lt_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_ltValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::lt(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_ltValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::s_lt_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->lt_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",2, false);
    THCudaDoubleTensor_ltTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDADoubleType::s_lt(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->lt(self, Scalar(other));
    }
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",2, false);
    THCudaDoubleTensor_ltTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDADoubleType::lt_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_ltValueT(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDADoubleType::s_lt_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->lt_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",3, false);
    THCudaDoubleTensor_ltTensorT(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDADoubleType::gt_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_gtValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::gt(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_gtValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::s_gt_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->gt_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",2, false);
    THCudaDoubleTensor_gtTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDADoubleType::s_gt(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->gt(self, Scalar(other));
    }
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",2, false);
    THCudaDoubleTensor_gtTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDADoubleType::gt_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_gtValueT(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDADoubleType::s_gt_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->gt_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",3, false);
    THCudaDoubleTensor_gtTensorT(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDADoubleType::le_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_leValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::le(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_leValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::s_le_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->le_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",2, false);
    THCudaDoubleTensor_leTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDADoubleType::s_le(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->le(self, Scalar(other));
    }
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",2, false);
    THCudaDoubleTensor_leTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDADoubleType::le_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_leValueT(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDADoubleType::s_le_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->le_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",3, false);
    THCudaDoubleTensor_leTensorT(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDADoubleType::ge_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_geValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::ge(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_geValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::s_ge_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ge_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",2, false);
    THCudaDoubleTensor_geTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDADoubleType::s_ge(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ge(self, Scalar(other));
    }
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",2, false);
    THCudaDoubleTensor_geTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDADoubleType::ge_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_geValueT(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDADoubleType::s_ge_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ge_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",3, false);
    THCudaDoubleTensor_geTensorT(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDADoubleType::eq_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_eqValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::eq(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_eqValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::s_eq_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->eq_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",2, false);
    THCudaDoubleTensor_eqTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDADoubleType::s_eq(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->eq(self, Scalar(other));
    }
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",2, false);
    THCudaDoubleTensor_eqTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDADoubleType::eq_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_eqValueT(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDADoubleType::s_eq_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->eq_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",3, false);
    THCudaDoubleTensor_eqTensorT(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDADoubleType::ne_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_neValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::ne(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_neValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::s_ne_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ne_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",2, false);
    THCudaDoubleTensor_neTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDADoubleType::s_ne(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ne(self, Scalar(other));
    }
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",2, false);
    THCudaDoubleTensor_neTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDADoubleType::ne_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_neValueT(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDADoubleType::s_ne_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ne_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",3, false);
    THCudaDoubleTensor_neTensorT(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
std::tuple<Tensor &,Tensor &> CUDADoubleType::min_out(Tensor & min, Tensor & min_indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto min_ = checked_cast_tensor<CUDADoubleTensor>(min.pImpl,"min",0, false);
    auto min_indices_ = checked_cast_tensor<CUDALongTensor>(min_indices.pImpl,"min_indices",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaDoubleTensor_min(context->thc_state, min_->tensor, min_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    min_->maybeScalar(maybe_scalar);
    min_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(min, min_indices);
}
std::tuple<Tensor,Tensor> CUDADoubleType::min(const Tensor & self, int64_t dim, bool keepdim) const {
    auto min_ = new CUDADoubleTensor(context);
    auto min = Tensor(min_, false);
    auto min_indices_ = new CUDALongTensor(context);
    auto min_indices = Tensor(min_indices_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaDoubleTensor_min(context->thc_state, min_->tensor, min_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    min_->maybeScalar(maybe_scalar);
    min_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(min, min_indices);
}
Tensor & CUDADoubleType::s_min_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",2, false);
    THCudaDoubleTensor_cmin(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDADoubleType::s_min(const Tensor & self, const Tensor & other) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",2, false);
    THCudaDoubleTensor_cmin(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDADoubleType::min(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<double>(THCudaDoubleTensor_minall(context->thc_state, self_->tensor)));
}
std::tuple<Tensor &,Tensor &> CUDADoubleType::max_out(Tensor & max, Tensor & max_indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto max_ = checked_cast_tensor<CUDADoubleTensor>(max.pImpl,"max",0, false);
    auto max_indices_ = checked_cast_tensor<CUDALongTensor>(max_indices.pImpl,"max_indices",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaDoubleTensor_max(context->thc_state, max_->tensor, max_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    max_->maybeScalar(maybe_scalar);
    max_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(max, max_indices);
}
std::tuple<Tensor,Tensor> CUDADoubleType::max(const Tensor & self, int64_t dim, bool keepdim) const {
    auto max_ = new CUDADoubleTensor(context);
    auto max = Tensor(max_, false);
    auto max_indices_ = new CUDALongTensor(context);
    auto max_indices = Tensor(max_indices_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaDoubleTensor_max(context->thc_state, max_->tensor, max_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    max_->maybeScalar(maybe_scalar);
    max_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(max, max_indices);
}
Tensor & CUDADoubleType::s_max_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",2, false);
    THCudaDoubleTensor_cmax(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDADoubleType::s_max(const Tensor & self, const Tensor & other) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",2, false);
    THCudaDoubleTensor_cmax(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDADoubleType::max(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<double>(THCudaDoubleTensor_maxall(context->thc_state, self_->tensor)));
}
std::tuple<Tensor &,Tensor &> CUDADoubleType::mode_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = checked_cast_tensor<CUDADoubleTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaDoubleTensor_mode(context->thc_state, values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CUDADoubleType::mode(const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = new CUDADoubleTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CUDALongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaDoubleTensor_mode(context->thc_state, values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
std::tuple<Tensor &,Tensor &> CUDADoubleType::median_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = checked_cast_tensor<CUDADoubleTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaDoubleTensor_median(context->thc_state, values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CUDADoubleType::median(const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = new CUDADoubleTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CUDALongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaDoubleTensor_median(context->thc_state, values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
Tensor CUDADoubleType::median(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<double>(THCudaDoubleTensor_medianall(context->thc_state, self_->tensor)));
}
std::tuple<Tensor &,Tensor &> CUDADoubleType::sort_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool descending) const {
    auto values_ = checked_cast_tensor<CUDADoubleTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaDoubleTensor_sort(context->thc_state, values_->tensor, indices_->tensor, self_->tensor, dim, descending);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CUDADoubleType::sort(const Tensor & self, int64_t dim, bool descending) const {
    auto values_ = new CUDADoubleTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CUDALongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaDoubleTensor_sort(context->thc_state, values_->tensor, indices_->tensor, self_->tensor, dim, descending);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
std::tuple<Tensor &,Tensor &> CUDADoubleType::topk_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) const {
    auto values_ = checked_cast_tensor<CUDADoubleTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaDoubleTensor_topk(context->thc_state, values_->tensor, indices_->tensor, self_->tensor, k, dim, largest, sorted);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CUDADoubleType::topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) const {
    auto values_ = new CUDADoubleTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CUDALongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaDoubleTensor_topk(context->thc_state, values_->tensor, indices_->tensor, self_->tensor, k, dim, largest, sorted);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
int64_t CUDADoubleType::get_device(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    return static_cast<int64_t>(THCudaDoubleTensor_getDevice(context->thc_state, self_->tensor));
}
Tensor & CUDADoubleType::_abs_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_abs(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::_abs(const Tensor & self) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_abs(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::sigmoid_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_sigmoid(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDADoubleType::sigmoid_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_sigmoid(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::sigmoid(const Tensor & self) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_sigmoid(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::_log_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_log(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::_log(const Tensor & self) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_log(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::log10_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_log10(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDADoubleType::log10_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_log10(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::log10(const Tensor & self) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_log10(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::log1p_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_log1p(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDADoubleType::log1p_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_log1p(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::log1p(const Tensor & self) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_log1p(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::log2_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_log2(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDADoubleType::log2_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_log2(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::log2(const Tensor & self) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_log2(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::lgamma_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_lgamma(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::lgamma(const Tensor & self) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_lgamma(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::lgamma_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_lgamma(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDADoubleType::digamma_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_digamma(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::digamma(const Tensor & self) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_digamma(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::digamma_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_digamma(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDADoubleType::polygamma_out(Tensor & result, int64_t n, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    THCudaDoubleTensor_polygamma(context->thc_state, result_->tensor, n, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::polygamma(int64_t n, const Tensor & self) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    THCudaDoubleTensor_polygamma(context->thc_state, result_->tensor, n, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::polygamma_(Tensor & self, int64_t n) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_polygamma(context->thc_state, self_->tensor, n, self_->tensor);
    return self;
}
Tensor & CUDADoubleType::_exp_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_exp(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::_exp(const Tensor & self) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_exp(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::expm1_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_expm1(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDADoubleType::expm1_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_expm1(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::expm1(const Tensor & self) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_expm1(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::_cos_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_cos(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::_cos(const Tensor & self) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_cos(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::acos_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_acos(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDADoubleType::acos_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_acos(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::acos(const Tensor & self) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_acos(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::cosh_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_cosh(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDADoubleType::cosh_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_cosh(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::cosh(const Tensor & self) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_cosh(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::_sin_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_sin(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::_sin(const Tensor & self) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_sin(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::asin_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_asin(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDADoubleType::asin_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_asin(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::asin(const Tensor & self) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_asin(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::sinh_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_sinh(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDADoubleType::sinh_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_sinh(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::sinh(const Tensor & self) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_sinh(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::tan_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_tan(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDADoubleType::tan_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_tan(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::tan(const Tensor & self) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_tan(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::atan_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_atan(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDADoubleType::atan_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_atan(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::atan(const Tensor & self) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_atan(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::tanh_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_tanh(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDADoubleType::tanh_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_tanh(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::tanh(const Tensor & self) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_tanh(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::erf_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_erf(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDADoubleType::erf_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_erf(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::erf(const Tensor & self) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_erf(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::erfinv_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_erfinv(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDADoubleType::erfinv_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_erfinv(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::erfinv(const Tensor & self) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_erfinv(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::_sqrt_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_sqrt(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::_sqrt(const Tensor & self) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_sqrt(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::rsqrt_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_rsqrt(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDADoubleType::rsqrt_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_rsqrt(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::rsqrt(const Tensor & self) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_rsqrt(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::_ceil_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_ceil(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::_ceil(const Tensor & self) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_ceil(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::_floor_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_floor(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::_floor(const Tensor & self) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_floor(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::_round_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_round(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::_round(const Tensor & self) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_round(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::_trunc_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_trunc(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::_trunc(const Tensor & self) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_trunc(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::frac_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_frac(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDADoubleType::frac_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_frac(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::frac(const Tensor & self) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_frac(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::mean_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaDoubleTensor_mean(context->thc_state, result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CUDADoubleType::mean(const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaDoubleTensor_mean(context->thc_state, result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CUDADoubleType::mean(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<double>(THCudaDoubleTensor_meanall(context->thc_state, self_->tensor)));
}
Tensor & CUDADoubleType::var_out(Tensor & result, const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaDoubleTensor_var(context->thc_state, result_->tensor, self_->tensor, dim, (unbiased) ? 0 : 1, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CUDADoubleType::var(const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaDoubleTensor_var(context->thc_state, result_->tensor, self_->tensor, dim, (unbiased) ? 0 : 1, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CUDADoubleType::var(const Tensor & self, bool unbiased) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<double>(THCudaDoubleTensor_varall(context->thc_state, self_->tensor, (unbiased) ? 0 : 1)));
}
Tensor & CUDADoubleType::std_out(Tensor & result, const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaDoubleTensor_std(context->thc_state, result_->tensor, self_->tensor, dim, (unbiased) ? 0 : 1, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CUDADoubleType::std(const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaDoubleTensor_std(context->thc_state, result_->tensor, self_->tensor, dim, (unbiased) ? 0 : 1, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CUDADoubleType::std(const Tensor & self, bool unbiased) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<double>(THCudaDoubleTensor_stdall(context->thc_state, self_->tensor, (unbiased) ? 0 : 1)));
}
Tensor & CUDADoubleType::norm_out(Tensor & result, const Tensor & self, Scalar p, int64_t dim, bool keepdim) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto p_ = p.toDouble();
    dim = maybe_wrap_dim(dim, self_);
    THCudaDoubleTensor_norm(context->thc_state, result_->tensor, self_->tensor, p_, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CUDADoubleType::norm(const Tensor & self, Scalar p, int64_t dim, bool keepdim) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto p_ = p.toDouble();
    dim = maybe_wrap_dim(dim, self_);
    THCudaDoubleTensor_norm(context->thc_state, result_->tensor, self_->tensor, p_, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CUDADoubleType::norm(const Tensor & self, Scalar p) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto p_ = p.toDouble();
    // norm(value) for a sparse tensor returns a DENSE 0-dim tensor
    if (self.is_sparse()) {
      auto result = THCudaDoubleTensor_normall(context->thc_state,  self_->tensor, convert<double>(p_));
      return toBackend(toDense(backend())).tensor({}).fill_(result);
    }
    // aten_custom_call is followed by the generated call to normall
    return scalarTensor(convert<double>(THCudaDoubleTensor_normall(context->thc_state, self_->tensor, p_)));
}
Tensor & CUDADoubleType::renorm_out(Tensor & result, const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto p_ = p.toDouble();
    dim = maybe_wrap_dim(dim, self_);
    auto maxnorm_ = maxnorm.toDouble();
    THCudaDoubleTensor_renorm(context->thc_state, result_->tensor, self_->tensor, p_, dim, maxnorm_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::renorm(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto p_ = p.toDouble();
    dim = maybe_wrap_dim(dim, self_);
    auto maxnorm_ = maxnorm.toDouble();
    THCudaDoubleTensor_renorm(context->thc_state, result_->tensor, self_->tensor, p_, dim, maxnorm_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::renorm_(Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto p_ = p.toDouble();
    dim = maybe_wrap_dim(dim, self_);
    auto maxnorm_ = maxnorm.toDouble();
    THCudaDoubleTensor_renorm(context->thc_state, self_->tensor, self_->tensor, p_, dim, maxnorm_);
    return self;
}
Tensor CUDADoubleType::s_dist(const Tensor & self, const Tensor & other, Scalar p) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",2, false);
    auto p_ = p.toDouble();
    return scalarTensor(convert<double>(THCudaDoubleTensor_dist(context->thc_state, self_->tensor, other_->tensor, p_)));
}
Tensor & CUDADoubleType::reciprocal_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_cinv(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::reciprocal(const Tensor & self) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_cinv(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::reciprocal_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_cinv(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDADoubleType::neg_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_neg(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::neg(const Tensor & self) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_neg(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::neg_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_neg(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDADoubleType::s_atan2_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",2, false);
    THCudaDoubleTensor_atan2(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDADoubleType::s_atan2(const Tensor & self, const Tensor & other) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",2, false);
    THCudaDoubleTensor_atan2(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDADoubleType::s_atan2_(Tensor & self, const Tensor & other) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",3, false);
    THCudaDoubleTensor_atan2(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDADoubleType::pow_out(Tensor & result, const Tensor & self, Scalar exponent) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toDouble();
    THCudaDoubleTensor_pow(context->thc_state, result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::pow(const Tensor & self, Scalar exponent) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toDouble();
    THCudaDoubleTensor_pow(context->thc_state, result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::s_pow_out(Tensor & result, const Tensor & self, const Tensor & exponent) const {
    if (exponent.dim() == 0) {
        return static_cast<const Type*>(this)->pow_out(result, self, Scalar(exponent));
    }
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto exponent_ = checked_cast_tensor<CUDADoubleTensor>(exponent.pImpl,"exponent",2, false);
    THCudaDoubleTensor_cpow(context->thc_state, result_->tensor, self_->tensor, exponent_->tensor);
    result_->maybeScalar(self_->isScalar() && exponent_->isScalar());
    return result;
}
Tensor CUDADoubleType::s_pow(const Tensor & self, const Tensor & exponent) const {
    if (exponent.dim() == 0) {
        return static_cast<const Type*>(this)->pow(self, Scalar(exponent));
    }
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto exponent_ = checked_cast_tensor<CUDADoubleTensor>(exponent.pImpl,"exponent",2, false);
    THCudaDoubleTensor_cpow(context->thc_state, result_->tensor, self_->tensor, exponent_->tensor);
    result_->maybeScalar(self_->isScalar() && exponent_->isScalar());
    return result;
}
Tensor & CUDADoubleType::pow_out(Tensor & result, Scalar base, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto base_ = base.toDouble();
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    THCudaDoubleTensor_tpow(context->thc_state, result_->tensor, base_, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::pow(Scalar base, const Tensor & self) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto base_ = base.toDouble();
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    THCudaDoubleTensor_tpow(context->thc_state, result_->tensor, base_, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::pow_(Tensor & self, Scalar exponent) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toDouble();
    THCudaDoubleTensor_pow(context->thc_state, self_->tensor, self_->tensor, exponent_);
    return self;
}
Tensor & CUDADoubleType::s_pow_(Tensor & self, const Tensor & exponent) const {
    if (exponent.dim() == 0) {
        return static_cast<const Type*>(this)->pow_(self, Scalar(exponent));
    }
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto exponent_ = checked_cast_tensor<CUDADoubleTensor>(exponent.pImpl,"exponent",3, false);
    THCudaDoubleTensor_cpow(context->thc_state, self_->tensor, self_->tensor, exponent_->tensor);
    return self;
}
Tensor & CUDADoubleType::s_lerp_out(Tensor & result, const Tensor & self, const Tensor & end, Scalar weight) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto end_ = checked_cast_tensor<CUDADoubleTensor>(end.pImpl,"end",2, false);
    auto weight_ = weight.toDouble();
    THCudaDoubleTensor_lerp(context->thc_state, result_->tensor, self_->tensor, end_->tensor, weight_);
    result_->maybeScalar(self_->isScalar() && end_->isScalar());
    return result;
}
Tensor CUDADoubleType::s_lerp(const Tensor & self, const Tensor & end, Scalar weight) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto end_ = checked_cast_tensor<CUDADoubleTensor>(end.pImpl,"end",2, false);
    auto weight_ = weight.toDouble();
    THCudaDoubleTensor_lerp(context->thc_state, result_->tensor, self_->tensor, end_->tensor, weight_);
    result_->maybeScalar(self_->isScalar() && end_->isScalar());
    return result;
}
Tensor & CUDADoubleType::s_lerp_(Tensor & self, const Tensor & end, Scalar weight) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto end_ = checked_cast_tensor<CUDADoubleTensor>(end.pImpl,"end",3, false);
    auto weight_ = weight.toDouble();
    THCudaDoubleTensor_lerp(context->thc_state, self_->tensor, self_->tensor, end_->tensor, weight_);
    return self;
}
Tensor & CUDADoubleType::zero_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_zero(context->thc_state, self_->tensor);
    return self;
}
Tensor CUDADoubleType::_sumall(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<double>(THCudaDoubleTensor_sumall(context->thc_state, self_->tensor)));
}
Tensor & CUDADoubleType::_sum_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaDoubleTensor_sum(context->thc_state, result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CUDADoubleType::_sum(const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaDoubleTensor_sum(context->thc_state, result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CUDADoubleType::_prodall(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<double>(THCudaDoubleTensor_prodall(context->thc_state, self_->tensor)));
}
Tensor & CUDADoubleType::_prod_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaDoubleTensor_prod(context->thc_state, result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CUDADoubleType::_prod(const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaDoubleTensor_prod(context->thc_state, result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor & CUDADoubleType::cumsum_out(Tensor & result, const Tensor & self, int64_t dim) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaDoubleTensor_cumsum(context->thc_state, result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::cumsum(const Tensor & self, int64_t dim) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaDoubleTensor_cumsum(context->thc_state, result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::cumprod_out(Tensor & result, const Tensor & self, int64_t dim) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaDoubleTensor_cumprod(context->thc_state, result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::cumprod(const Tensor & self, int64_t dim) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaDoubleTensor_cumprod(context->thc_state, result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::sign_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_sign(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::sign(const Tensor & self) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_sign(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::sign_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_sign(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor CUDADoubleType::trace(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<double>(THCudaDoubleTensor_trace(context->thc_state, self_->tensor)));
}
Tensor & CUDADoubleType::add_out(Tensor & result, const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    auto alpha_ = alpha.toDouble();
    THCudaDoubleTensor_add_scaled(context->thc_state, result_->tensor, self_->tensor, other_, alpha_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::add(const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    auto alpha_ = alpha.toDouble();
    THCudaDoubleTensor_add_scaled(context->thc_state, result_->tensor, self_->tensor, other_, alpha_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::s_add_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if(other.type().is_sparse()) {
        return static_cast<const Type*>(this)->add_out(result, self, SparseTensor(other), alpha);
    }
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toDouble();
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",3, false);
    THCudaDoubleTensor_cadd(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDADoubleType::s_add(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if(other.type().is_sparse()) {
        return static_cast<const Type*>(this)->add(self, SparseTensor(other), alpha);
    }
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add(self, Scalar(other), alpha);
    }
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toDouble();
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",3, false);
    THCudaDoubleTensor_cadd(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDADoubleType::add_out(Tensor & result, const Tensor & self, SparseTensor other, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toDouble();
    auto other_ = checked_cast_tensor<SparseCUDADoubleTensor>(other.tref.pImpl,"other",3,false);
    THCSDoubleTensor_spcadd(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::add(const Tensor & self, SparseTensor other, Scalar alpha) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toDouble();
    auto other_ = checked_cast_tensor<SparseCUDADoubleTensor>(other.tref.pImpl,"other",3,false);
    THCSDoubleTensor_spcadd(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::add_(Tensor & self, Scalar other, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    auto alpha_ = alpha.toDouble();
    THCudaDoubleTensor_add_scaled(context->thc_state, self_->tensor, self_->tensor, other_, alpha_);
    return self;
}
Tensor & CUDADoubleType::s_add_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if(other.type().is_sparse()) {
        return static_cast<const Type*>(this)->add_(self, SparseTensor(other), alpha);
    }
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toDouble();
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",4, false);
    THCudaDoubleTensor_cadd(context->thc_state, self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & CUDADoubleType::add_(Tensor & self, SparseTensor other, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toDouble();
    auto other_ = checked_cast_tensor<SparseCUDADoubleTensor>(other.tref.pImpl,"other",4,false);
    THCSDoubleTensor_spcadd(context->thc_state, self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & CUDADoubleType::sub_out(Tensor & result, const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    auto alpha_ = alpha.toDouble();
    THCudaDoubleTensor_sub_scaled(context->thc_state, result_->tensor, self_->tensor, other_, alpha_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::sub(const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    auto alpha_ = alpha.toDouble();
    THCudaDoubleTensor_sub_scaled(context->thc_state, result_->tensor, self_->tensor, other_, alpha_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::s_sub_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toDouble();
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",3, false);
    THCudaDoubleTensor_csub(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDADoubleType::s_sub(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub(self, Scalar(other), alpha);
    }
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toDouble();
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",3, false);
    THCudaDoubleTensor_csub(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDADoubleType::sub_(Tensor & self, Scalar other, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    auto alpha_ = alpha.toDouble();
    THCudaDoubleTensor_sub_scaled(context->thc_state, self_->tensor, self_->tensor, other_, alpha_);
    return self;
}
Tensor & CUDADoubleType::s_sub_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toDouble();
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",4, false);
    THCudaDoubleTensor_csub(context->thc_state, self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & CUDADoubleType::mul_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_mul(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::mul(const Tensor & self, Scalar other) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_mul(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::s_mul_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",2, false);
    THCudaDoubleTensor_cmul(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDADoubleType::s_mul(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul(self, Scalar(other));
    }
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",2, false);
    THCudaDoubleTensor_cmul(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDADoubleType::mul_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_mul(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDADoubleType::s_mul_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",3, false);
    THCudaDoubleTensor_cmul(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDADoubleType::div_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_div(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::div(const Tensor & self, Scalar other) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_div(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::s_div_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->div_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",2, false);
    THCudaDoubleTensor_cdiv(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDADoubleType::s_div(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->div(self, Scalar(other));
    }
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",2, false);
    THCudaDoubleTensor_cdiv(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDADoubleType::div_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_div(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDADoubleType::s_div_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->div_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",3, false);
    THCudaDoubleTensor_cdiv(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDADoubleType::fmod_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_fmod(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::fmod(const Tensor & self, Scalar other) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_fmod(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::s_fmod_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->fmod_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",2, false);
    THCudaDoubleTensor_cfmod(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDADoubleType::s_fmod(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->fmod(self, Scalar(other));
    }
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",2, false);
    THCudaDoubleTensor_cfmod(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDADoubleType::fmod_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_fmod(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDADoubleType::s_fmod_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->fmod_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",3, false);
    THCudaDoubleTensor_cfmod(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDADoubleType::remainder_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_remainder(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::remainder(const Tensor & self, Scalar other) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_remainder(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::s_remainder_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->remainder_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",2, false);
    THCudaDoubleTensor_cremainder(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDADoubleType::s_remainder(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->remainder(self, Scalar(other));
    }
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",2, false);
    THCudaDoubleTensor_cremainder(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDADoubleType::remainder_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THCudaDoubleTensor_remainder(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDADoubleType::s_remainder_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->remainder_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",3, false);
    THCudaDoubleTensor_cremainder(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDADoubleType::clamp_out(Tensor & result, const Tensor & self, Scalar min, Scalar max) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toDouble();
    auto max_ = max.toDouble();
    THCudaDoubleTensor_clamp(context->thc_state, result_->tensor, self_->tensor, min_, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::clamp(const Tensor & self, Scalar min, Scalar max) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toDouble();
    auto max_ = max.toDouble();
    THCudaDoubleTensor_clamp(context->thc_state, result_->tensor, self_->tensor, min_, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::clamp_(Tensor & self, Scalar min, Scalar max) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toDouble();
    auto max_ = max.toDouble();
    THCudaDoubleTensor_clamp(context->thc_state, self_->tensor, self_->tensor, min_, max_);
    return self;
}
Tensor & CUDADoubleType::clamp_min_out(Tensor & result, const Tensor & self, Scalar min) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toDouble();
    THCudaDoubleTensor_cmaxValue(context->thc_state, result_->tensor, self_->tensor, min_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::clamp_min(const Tensor & self, Scalar min) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toDouble();
    THCudaDoubleTensor_cmaxValue(context->thc_state, result_->tensor, self_->tensor, min_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::clamp_min_(Tensor & self, Scalar min) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toDouble();
    THCudaDoubleTensor_cmaxValue(context->thc_state, self_->tensor, self_->tensor, min_);
    return self;
}
Tensor & CUDADoubleType::clamp_max_out(Tensor & result, const Tensor & self, Scalar max) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto max_ = max.toDouble();
    THCudaDoubleTensor_cminValue(context->thc_state, result_->tensor, self_->tensor, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::clamp_max(const Tensor & self, Scalar max) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto max_ = max.toDouble();
    THCudaDoubleTensor_cminValue(context->thc_state, result_->tensor, self_->tensor, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::clamp_max_(Tensor & self, Scalar max) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto max_ = max.toDouble();
    THCudaDoubleTensor_cminValue(context->thc_state, self_->tensor, self_->tensor, max_);
    return self;
}
Tensor CUDADoubleType::_dot(const Tensor & self, const Tensor & tensor) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto tensor_ = checked_cast_tensor<CUDADoubleTensor>(tensor.pImpl,"tensor",2, false);
    return scalarTensor(convert<double>(THCudaDoubleTensor_dot(context->thc_state, self_->tensor, tensor_->tensor)));
}
Tensor & CUDADoubleType::tril_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_tril(context->thc_state, result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::tril(const Tensor & self, int64_t diagonal) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_tril(context->thc_state, result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::tril_(Tensor & self, int64_t diagonal) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_tril(context->thc_state, self_->tensor, self_->tensor, diagonal);
    return self;
}
Tensor & CUDADoubleType::triu_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_triu(context->thc_state, result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::triu(const Tensor & self, int64_t diagonal) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_triu(context->thc_state, result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::triu_(Tensor & self, int64_t diagonal) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_triu(context->thc_state, self_->tensor, self_->tensor, diagonal);
    return self;
}
Tensor & CUDADoubleType::cross_out(Tensor & result, const Tensor & self, const Tensor & other, int64_t dim) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",2, false);
    THCudaDoubleTensor_cross(context->thc_state, result_->tensor, self_->tensor, other_->tensor, dim);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDADoubleType::cross(const Tensor & self, const Tensor & other, int64_t dim) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDADoubleTensor>(other.pImpl,"other",2, false);
    THCudaDoubleTensor_cross(context->thc_state, result_->tensor, self_->tensor, other_->tensor, dim);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDADoubleType::diag_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    if (self_->isScalar()) {
      throw std::runtime_error("Input must be 1-d or 2-d");
    }
    THCudaDoubleTensor_diag(context->thc_state, result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::diag(const Tensor & self, int64_t diagonal) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    if (self_->isScalar()) {
      throw std::runtime_error("Input must be 1-d or 2-d");
    }
    THCudaDoubleTensor_diag(context->thc_state, result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::s_addmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    if(mat1.type().is_sparse()) {
        return static_cast<const Type*>(this)->addmm_out(result, self, SparseTensor(mat1), mat2, beta, alpha);
    }
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toDouble();
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toDouble();
    auto mat1_ = checked_cast_tensor<CUDADoubleTensor>(mat1.pImpl,"mat1",4, false);
    auto mat2_ = checked_cast_tensor<CUDADoubleTensor>(mat2.pImpl,"mat2",5, false);
    THCudaDoubleTensor_addmm(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CUDADoubleType::s_addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    if(mat1.type().is_sparse()) {
        return static_cast<const Type*>(this)->addmm(self, SparseTensor(mat1), mat2, beta, alpha);
    }
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toDouble();
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toDouble();
    auto mat1_ = checked_cast_tensor<CUDADoubleTensor>(mat1.pImpl,"mat1",4, false);
    auto mat2_ = checked_cast_tensor<CUDADoubleTensor>(mat2.pImpl,"mat2",5, false);
    THCudaDoubleTensor_addmm(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CUDADoubleType::addmm_out(Tensor & result, const Tensor & self, SparseTensor mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toDouble();
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toDouble();
    auto mat1_ = checked_cast_tensor<SparseCUDADoubleTensor>(mat1.tref.pImpl,"mat1",4,false);
    auto mat2_ = checked_cast_tensor<CUDADoubleTensor>(mat2.pImpl,"mat2",5, false);
    THCSDoubleTensor_spaddmm(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CUDADoubleType::addmm(const Tensor & self, SparseTensor mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toDouble();
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toDouble();
    auto mat1_ = checked_cast_tensor<SparseCUDADoubleTensor>(mat1.tref.pImpl,"mat1",4,false);
    auto mat2_ = checked_cast_tensor<CUDADoubleTensor>(mat2.pImpl,"mat2",5, false);
    THCSDoubleTensor_spaddmm(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CUDADoubleType::addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    if(mat1.type().is_sparse()) {
        return static_cast<const Type*>(this)->addmm_(self, SparseTensor(mat1), mat2, beta, alpha);
    }
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toDouble();
    auto alpha_ = alpha.toDouble();
    auto mat1_ = checked_cast_tensor<CUDADoubleTensor>(mat1.pImpl,"mat1",5, false);
    auto mat2_ = checked_cast_tensor<CUDADoubleTensor>(mat2.pImpl,"mat2",6, false);
    THCudaDoubleTensor_addmm(context->thc_state, self_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    return self;
}
Tensor & CUDADoubleType::addmm_(Tensor & self, SparseTensor mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toDouble();
    auto alpha_ = alpha.toDouble();
    auto mat1_ = checked_cast_tensor<SparseCUDADoubleTensor>(mat1.tref.pImpl,"mat1",5,false);
    auto mat2_ = checked_cast_tensor<CUDADoubleTensor>(mat2.pImpl,"mat2",6, false);
    THCSDoubleTensor_spaddmm(context->thc_state, self_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    return self;
}
Tensor & CUDADoubleType::s__addmv_out(Tensor & result, const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toDouble();
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toDouble();
    auto mat_ = checked_cast_tensor<CUDADoubleTensor>(mat.pImpl,"mat",4, false);
    auto vec_ = checked_cast_tensor<CUDADoubleTensor>(vec.pImpl,"vec",5, false);
    THCudaDoubleTensor_addmv(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, mat_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && mat_->isScalar() && vec_->isScalar());
    return result;
}
Tensor CUDADoubleType::s__addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toDouble();
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toDouble();
    auto mat_ = checked_cast_tensor<CUDADoubleTensor>(mat.pImpl,"mat",4, false);
    auto vec_ = checked_cast_tensor<CUDADoubleTensor>(vec.pImpl,"vec",5, false);
    THCudaDoubleTensor_addmv(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, mat_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && mat_->isScalar() && vec_->isScalar());
    return result;
}
Tensor & CUDADoubleType::_addmv_(Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toDouble();
    auto alpha_ = alpha.toDouble();
    auto mat_ = checked_cast_tensor<CUDADoubleTensor>(mat.pImpl,"mat",5, false);
    auto vec_ = checked_cast_tensor<CUDADoubleTensor>(vec.pImpl,"vec",6, false);
    THCudaDoubleTensor_addmv(context->thc_state, self_->tensor, beta_, self_->tensor, alpha_, mat_->tensor, vec_->tensor);
    return self;
}
Tensor & CUDADoubleType::s__addr_out(Tensor & result, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toDouble();
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toDouble();
    auto vec1_ = checked_cast_tensor<CUDADoubleTensor>(vec1.pImpl,"vec1",4, false);
    auto vec2_ = checked_cast_tensor<CUDADoubleTensor>(vec2.pImpl,"vec2",5, false);
    THCudaDoubleTensor_addr(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, vec1_->tensor, vec2_->tensor);
    result_->maybeScalar(self_->isScalar() && vec1_->isScalar() && vec2_->isScalar());
    return result;
}
Tensor CUDADoubleType::s__addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toDouble();
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toDouble();
    auto vec1_ = checked_cast_tensor<CUDADoubleTensor>(vec1.pImpl,"vec1",4, false);
    auto vec2_ = checked_cast_tensor<CUDADoubleTensor>(vec2.pImpl,"vec2",5, false);
    THCudaDoubleTensor_addr(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, vec1_->tensor, vec2_->tensor);
    result_->maybeScalar(self_->isScalar() && vec1_->isScalar() && vec2_->isScalar());
    return result;
}
Tensor & CUDADoubleType::_addr_(Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toDouble();
    auto alpha_ = alpha.toDouble();
    auto vec1_ = checked_cast_tensor<CUDADoubleTensor>(vec1.pImpl,"vec1",5, false);
    auto vec2_ = checked_cast_tensor<CUDADoubleTensor>(vec2.pImpl,"vec2",6, false);
    THCudaDoubleTensor_addr(context->thc_state, self_->tensor, beta_, self_->tensor, alpha_, vec1_->tensor, vec2_->tensor);
    return self;
}
Tensor & CUDADoubleType::_ger_out(Tensor & result, const Tensor & self, const Tensor & vec2) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.dim() == 0 ? 1 : self.size(0),vec2.dim() == 0 ? 1 : vec2.size(0) });
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto vec2_ = checked_cast_tensor<CUDADoubleTensor>(vec2.pImpl,"vec2",2, false);
    THCudaDoubleTensor_addr(context->thc_state, result_->tensor, double(0), result_->tensor, double(1), self_->tensor, vec2_->tensor);
    result_->maybeScalar(false);
    return result;
}
Tensor CUDADoubleType::_ger(const Tensor & self, const Tensor & vec2) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.dim() == 0 ? 1 : self.size(0),vec2.dim() == 0 ? 1 : vec2.size(0) });
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto vec2_ = checked_cast_tensor<CUDADoubleTensor>(vec2.pImpl,"vec2",2, false);
    THCudaDoubleTensor_addr(context->thc_state, result_->tensor, double(0), result_->tensor, double(1), self_->tensor, vec2_->tensor);
    result_->maybeScalar(false);
    return result;
}
Tensor & CUDADoubleType::_mv_out(Tensor & result, const Tensor & self, const Tensor & vec) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.size(0) });
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto vec_ = checked_cast_tensor<CUDADoubleTensor>(vec.pImpl,"vec",2, false);
    THCudaDoubleTensor_addmv(context->thc_state, result_->tensor, double(0), result_->tensor, double(1), self_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && vec_->isScalar());
    return result;
}
Tensor CUDADoubleType::_mv(const Tensor & self, const Tensor & vec) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.size(0) });
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto vec_ = checked_cast_tensor<CUDADoubleTensor>(vec.pImpl,"vec",2, false);
    THCudaDoubleTensor_addmv(context->thc_state, result_->tensor, double(0), result_->tensor, double(1), self_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && vec_->isScalar());
    return result;
}
Tensor & CUDADoubleType::_mm_out(Tensor & result, const Tensor & self, const Tensor & mat2) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.size(0),mat2.size(1) });
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CUDADoubleTensor>(mat2.pImpl,"mat2",2, false);
    THCudaDoubleTensor_addmm(context->thc_state, result_->tensor, double(0), result_->tensor, double(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CUDADoubleType::_mm(const Tensor & self, const Tensor & mat2) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.size(0),mat2.size(1) });
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CUDADoubleTensor>(mat2.pImpl,"mat2",2, false);
    THCudaDoubleTensor_addmm(context->thc_state, result_->tensor, double(0), result_->tensor, double(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CUDADoubleType::bmm_out(Tensor & result, const Tensor & self, const Tensor & mat2) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.size(0),self.size(1),mat2.size(2) });
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CUDADoubleTensor>(mat2.pImpl,"mat2",2, false);
    THCudaDoubleTensor_baddbmm(context->thc_state, result_->tensor, double(0), result_->tensor, double(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CUDADoubleType::bmm(const Tensor & self, const Tensor & mat2) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.size(0),self.size(1),mat2.size(2) });
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CUDADoubleTensor>(mat2.pImpl,"mat2",2, false);
    THCudaDoubleTensor_baddbmm(context->thc_state, result_->tensor, double(0), result_->tensor, double(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CUDADoubleType::s_addbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toDouble();
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toDouble();
    auto batch1_ = checked_cast_tensor<CUDADoubleTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CUDADoubleTensor>(batch2.pImpl,"batch2",5, false);
    THCudaDoubleTensor_addbmm(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor CUDADoubleType::s_addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toDouble();
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toDouble();
    auto batch1_ = checked_cast_tensor<CUDADoubleTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CUDADoubleTensor>(batch2.pImpl,"batch2",5, false);
    THCudaDoubleTensor_addbmm(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor & CUDADoubleType::addbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toDouble();
    auto alpha_ = alpha.toDouble();
    auto batch1_ = checked_cast_tensor<CUDADoubleTensor>(batch1.pImpl,"batch1",5, false);
    auto batch2_ = checked_cast_tensor<CUDADoubleTensor>(batch2.pImpl,"batch2",6, false);
    THCudaDoubleTensor_addbmm(context->thc_state, self_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    return self;
}
Tensor & CUDADoubleType::s_baddbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toDouble();
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toDouble();
    auto batch1_ = checked_cast_tensor<CUDADoubleTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CUDADoubleTensor>(batch2.pImpl,"batch2",5, false);
    THCudaDoubleTensor_baddbmm(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor CUDADoubleType::s_baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toDouble();
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toDouble();
    auto batch1_ = checked_cast_tensor<CUDADoubleTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CUDADoubleTensor>(batch2.pImpl,"batch2",5, false);
    THCudaDoubleTensor_baddbmm(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor & CUDADoubleType::baddbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toDouble();
    auto alpha_ = alpha.toDouble();
    auto batch1_ = checked_cast_tensor<CUDADoubleTensor>(batch1.pImpl,"batch1",5, false);
    auto batch2_ = checked_cast_tensor<CUDADoubleTensor>(batch2.pImpl,"batch2",6, false);
    THCudaDoubleTensor_baddbmm(context->thc_state, self_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    return self;
}
Tensor & CUDADoubleType::s_addcmul_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toDouble();
    auto tensor1_ = checked_cast_tensor<CUDADoubleTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CUDADoubleTensor>(tensor2.pImpl,"tensor2",4, false);
    THCudaDoubleTensor_addcmul(context->thc_state, result_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor CUDADoubleType::s_addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toDouble();
    auto tensor1_ = checked_cast_tensor<CUDADoubleTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CUDADoubleTensor>(tensor2.pImpl,"tensor2",4, false);
    THCudaDoubleTensor_addcmul(context->thc_state, result_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor & CUDADoubleType::s_addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toDouble();
    auto tensor1_ = checked_cast_tensor<CUDADoubleTensor>(tensor1.pImpl,"tensor1",4, false);
    auto tensor2_ = checked_cast_tensor<CUDADoubleTensor>(tensor2.pImpl,"tensor2",5, false);
    THCudaDoubleTensor_addcmul(context->thc_state, self_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    return self;
}
Tensor & CUDADoubleType::s_addcdiv_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toDouble();
    auto tensor1_ = checked_cast_tensor<CUDADoubleTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CUDADoubleTensor>(tensor2.pImpl,"tensor2",4, false);
    THCudaDoubleTensor_addcdiv(context->thc_state, result_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor CUDADoubleType::s_addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toDouble();
    auto tensor1_ = checked_cast_tensor<CUDADoubleTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CUDADoubleTensor>(tensor2.pImpl,"tensor2",4, false);
    THCudaDoubleTensor_addcdiv(context->thc_state, result_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor & CUDADoubleType::s_addcdiv_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toDouble();
    auto tensor1_ = checked_cast_tensor<CUDADoubleTensor>(tensor1.pImpl,"tensor1",4, false);
    auto tensor2_ = checked_cast_tensor<CUDADoubleTensor>(tensor2.pImpl,"tensor2",5, false);
    THCudaDoubleTensor_addcdiv(context->thc_state, self_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    return self;
}
std::tuple<Tensor &,Tensor &> CUDADoubleType::gesv_out(Tensor & solution, Tensor & lu, const Tensor & self, const Tensor & A) const {
    auto solution_ = checked_cast_tensor<CUDADoubleTensor>(solution.pImpl,"solution",0, false);
    auto lu_ = checked_cast_tensor<CUDADoubleTensor>(lu.pImpl,"lu",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto A_ = checked_cast_tensor<CUDADoubleTensor>(A.pImpl,"A",2, false);
    THCudaDoubleTensor_gesv(context->thc_state, solution_->tensor, lu_->tensor, self_->tensor, A_->tensor);
    bool maybe_scalar = self_->isScalar() && A_->isScalar();
    solution_->maybeScalar(maybe_scalar);
    lu_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(solution, lu);
}
std::tuple<Tensor,Tensor> CUDADoubleType::gesv(const Tensor & self, const Tensor & A) const {
    auto solution_ = new CUDADoubleTensor(context);
    auto solution = Tensor(solution_, false);
    auto lu_ = new CUDADoubleTensor(context);
    auto lu = Tensor(lu_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto A_ = checked_cast_tensor<CUDADoubleTensor>(A.pImpl,"A",2, false);
    THCudaDoubleTensor_gesv(context->thc_state, solution_->tensor, lu_->tensor, self_->tensor, A_->tensor);
    bool maybe_scalar = self_->isScalar() && A_->isScalar();
    solution_->maybeScalar(maybe_scalar);
    lu_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(solution, lu);
}
std::tuple<Tensor &,Tensor &> CUDADoubleType::gels_out(Tensor & res1, Tensor & res2, const Tensor & self, const Tensor & A) const {
    auto res1_ = checked_cast_tensor<CUDADoubleTensor>(res1.pImpl,"res1",0, false);
    auto res2_ = checked_cast_tensor<CUDADoubleTensor>(res2.pImpl,"res2",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto A_ = checked_cast_tensor<CUDADoubleTensor>(A.pImpl,"A",2, false);
    THCudaDoubleTensor_gels(context->thc_state, res1_->tensor, res2_->tensor, self_->tensor, A_->tensor);
    bool maybe_scalar = self_->isScalar() && A_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(res1, res2);
}
std::tuple<Tensor,Tensor> CUDADoubleType::gels(const Tensor & self, const Tensor & A) const {
    auto res1_ = new CUDADoubleTensor(context);
    auto res1 = Tensor(res1_, false);
    auto res2_ = new CUDADoubleTensor(context);
    auto res2 = Tensor(res2_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto A_ = checked_cast_tensor<CUDADoubleTensor>(A.pImpl,"A",2, false);
    THCudaDoubleTensor_gels(context->thc_state, res1_->tensor, res2_->tensor, self_->tensor, A_->tensor);
    bool maybe_scalar = self_->isScalar() && A_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(res1, res2);
}
std::tuple<Tensor &,Tensor &> CUDADoubleType::symeig_out(Tensor & res1, Tensor & res2, const Tensor & self, bool eigenvectors, bool upper) const {
    auto res1_ = checked_cast_tensor<CUDADoubleTensor>(res1.pImpl,"res1",0, false);
    auto res2_ = checked_cast_tensor<CUDADoubleTensor>(res2.pImpl,"res2",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_syev(context->thc_state, res1_->tensor, res2_->tensor, self_->tensor, (eigenvectors) ? "V" : "N", (upper) ? "U" : "L");
    bool maybe_scalar = self_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(res1, res2);
}
std::tuple<Tensor,Tensor> CUDADoubleType::symeig(const Tensor & self, bool eigenvectors, bool upper) const {
    auto res1_ = new CUDADoubleTensor(context);
    auto res1 = Tensor(res1_, false);
    auto res2_ = new CUDADoubleTensor(context);
    auto res2 = Tensor(res2_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_syev(context->thc_state, res1_->tensor, res2_->tensor, self_->tensor, (eigenvectors) ? "V" : "N", (upper) ? "U" : "L");
    bool maybe_scalar = self_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(res1, res2);
}
std::tuple<Tensor &,Tensor &> CUDADoubleType::eig_out(Tensor & res1, Tensor & res2, const Tensor & self, bool eigenvectors) const {
    auto res1_ = checked_cast_tensor<CUDADoubleTensor>(res1.pImpl,"res1",0, false);
    auto res2_ = checked_cast_tensor<CUDADoubleTensor>(res2.pImpl,"res2",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_geev(context->thc_state, res1_->tensor, res2_->tensor, self_->tensor, (eigenvectors) ? "V" : "N");
    bool maybe_scalar = self_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(res1, res2);
}
std::tuple<Tensor,Tensor> CUDADoubleType::eig(const Tensor & self, bool eigenvectors) const {
    auto res1_ = new CUDADoubleTensor(context);
    auto res1 = Tensor(res1_, false);
    auto res2_ = new CUDADoubleTensor(context);
    auto res2 = Tensor(res2_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_geev(context->thc_state, res1_->tensor, res2_->tensor, self_->tensor, (eigenvectors) ? "V" : "N");
    bool maybe_scalar = self_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(res1, res2);
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDADoubleType::svd_out(Tensor & res1, Tensor & res2, Tensor & res3, const Tensor & self, bool some) const {
    auto res1_ = checked_cast_tensor<CUDADoubleTensor>(res1.pImpl,"res1",0, false);
    auto res2_ = checked_cast_tensor<CUDADoubleTensor>(res2.pImpl,"res2",0, false);
    auto res3_ = checked_cast_tensor<CUDADoubleTensor>(res3.pImpl,"res3",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_gesvd(context->thc_state, res1_->tensor, res2_->tensor, res3_->tensor, self_->tensor, (some) ? "S" : "A");
    bool maybe_scalar = self_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    res3_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(res1, res2, res3);
}
std::tuple<Tensor,Tensor,Tensor> CUDADoubleType::svd(const Tensor & self, bool some) const {
    auto res1_ = new CUDADoubleTensor(context);
    auto res1 = Tensor(res1_, false);
    auto res2_ = new CUDADoubleTensor(context);
    auto res2 = Tensor(res2_, false);
    auto res3_ = new CUDADoubleTensor(context);
    auto res3 = Tensor(res3_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_gesvd(context->thc_state, res1_->tensor, res2_->tensor, res3_->tensor, self_->tensor, (some) ? "S" : "A");
    bool maybe_scalar = self_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    res3_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(res1, res2, res3);
}
Tensor & CUDADoubleType::inverse_out(Tensor & output, const Tensor & self) const {
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_getri(context->thc_state, output_->tensor, self_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDADoubleType::inverse(const Tensor & self) const {
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_getri(context->thc_state, output_->tensor, self_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDADoubleType::potrf_out(Tensor & output, const Tensor & self, bool upper) const {
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_potrf(context->thc_state, output_->tensor, self_->tensor, (upper) ? "U" : "L");
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDADoubleType::potrf(const Tensor & self, bool upper) const {
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_potrf(context->thc_state, output_->tensor, self_->tensor, (upper) ? "U" : "L");
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDADoubleType::potrs_out(Tensor & result, const Tensor & self, const Tensor & input2, bool upper) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto input2_ = checked_cast_tensor<CUDADoubleTensor>(input2.pImpl,"input2",2, false);
    THCudaDoubleTensor_potrs(context->thc_state, result_->tensor, self_->tensor, input2_->tensor, (upper) ? "U" : "L");
    result_->maybeScalar(self_->isScalar() && input2_->isScalar());
    return result;
}
Tensor CUDADoubleType::potrs(const Tensor & self, const Tensor & input2, bool upper) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto input2_ = checked_cast_tensor<CUDADoubleTensor>(input2.pImpl,"input2",2, false);
    THCudaDoubleTensor_potrs(context->thc_state, result_->tensor, self_->tensor, input2_->tensor, (upper) ? "U" : "L");
    result_->maybeScalar(self_->isScalar() && input2_->isScalar());
    return result;
}
Tensor & CUDADoubleType::potri_out(Tensor & output, const Tensor & self, bool upper) const {
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_potri(context->thc_state, output_->tensor, self_->tensor, (upper) ? "U" : "L");
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDADoubleType::potri(const Tensor & self, bool upper) const {
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_potri(context->thc_state, output_->tensor, self_->tensor, (upper) ? "U" : "L");
    output_->maybeScalar(self_->isScalar());
    return output;
}
std::tuple<Tensor &,Tensor &> CUDADoubleType::qr_out(Tensor & res1, Tensor & res2, const Tensor & self) const {
    auto res1_ = checked_cast_tensor<CUDADoubleTensor>(res1.pImpl,"res1",0, false);
    auto res2_ = checked_cast_tensor<CUDADoubleTensor>(res2.pImpl,"res2",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_qr(context->thc_state, res1_->tensor, res2_->tensor, self_->tensor);
    bool maybe_scalar = self_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(res1, res2);
}
std::tuple<Tensor,Tensor> CUDADoubleType::qr(const Tensor & self) const {
    auto res1_ = new CUDADoubleTensor(context);
    auto res1 = Tensor(res1_, false);
    auto res2_ = new CUDADoubleTensor(context);
    auto res2 = Tensor(res2_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_qr(context->thc_state, res1_->tensor, res2_->tensor, self_->tensor);
    bool maybe_scalar = self_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(res1, res2);
}
std::tuple<Tensor &,Tensor &> CUDADoubleType::geqrf_out(Tensor & res1, Tensor & res2, const Tensor & self) const {
    auto res1_ = checked_cast_tensor<CUDADoubleTensor>(res1.pImpl,"res1",0, false);
    auto res2_ = checked_cast_tensor<CUDADoubleTensor>(res2.pImpl,"res2",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_geqrf(context->thc_state, res1_->tensor, res2_->tensor, self_->tensor);
    bool maybe_scalar = self_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(res1, res2);
}
std::tuple<Tensor,Tensor> CUDADoubleType::geqrf(const Tensor & self) const {
    auto res1_ = new CUDADoubleTensor(context);
    auto res1 = Tensor(res1_, false);
    auto res2_ = new CUDADoubleTensor(context);
    auto res2 = Tensor(res2_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    THCudaDoubleTensor_geqrf(context->thc_state, res1_->tensor, res2_->tensor, self_->tensor);
    bool maybe_scalar = self_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(res1, res2);
}
std::tuple<Tensor &,Tensor &> CUDADoubleType::btrifact_out(Tensor & result, Tensor & pivots, const Tensor & self, bool pivot) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto pivots_ = checked_cast_tensor<CUDAIntTensor>(pivots.pImpl,"pivots",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    THCudaDoubleTensor_btrifact(context->thc_state, result_->tensor, pivots_->tensor, NULL, pivot, self_->tensor);
    bool maybe_scalar = self_->isScalar();
    result_->maybeScalar(maybe_scalar);
    pivots_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(result, pivots);
}
std::tuple<Tensor,Tensor> CUDADoubleType::btrifact(const Tensor & self, bool pivot) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto pivots_ = new CUDAIntTensor(context);
    auto pivots = Tensor(pivots_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    THCudaDoubleTensor_btrifact(context->thc_state, result_->tensor, pivots_->tensor, NULL, pivot, self_->tensor);
    bool maybe_scalar = self_->isScalar();
    result_->maybeScalar(maybe_scalar);
    pivots_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(result, pivots);
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDADoubleType::btrifact_with_info_out(Tensor & result, Tensor & pivots, Tensor & info, const Tensor & self, bool pivot) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto pivots_ = checked_cast_tensor<CUDAIntTensor>(pivots.pImpl,"pivots",0, false);
    auto info_ = checked_cast_tensor<CUDAIntTensor>(info.pImpl,"info",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    THCudaDoubleTensor_btrifact(context->thc_state, result_->tensor, pivots_->tensor, info_->tensor, pivot, self_->tensor);
    bool maybe_scalar = self_->isScalar();
    result_->maybeScalar(maybe_scalar);
    pivots_->maybeScalar(maybe_scalar);
    info_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(result, pivots, info);
}
std::tuple<Tensor,Tensor,Tensor> CUDADoubleType::btrifact_with_info(const Tensor & self, bool pivot) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto pivots_ = new CUDAIntTensor(context);
    auto pivots = Tensor(pivots_, false);
    auto info_ = new CUDAIntTensor(context);
    auto info = Tensor(info_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    THCudaDoubleTensor_btrifact(context->thc_state, result_->tensor, pivots_->tensor, info_->tensor, pivot, self_->tensor);
    bool maybe_scalar = self_->isScalar();
    result_->maybeScalar(maybe_scalar);
    pivots_->maybeScalar(maybe_scalar);
    info_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(result, pivots, info);
}
Tensor & CUDADoubleType::btrisolve_out(Tensor & result, const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto LU_data_ = checked_cast_tensor<CUDADoubleTensor>(LU_data.pImpl,"LU_data",2, false);
    auto LU_pivots_ = checked_cast_tensor<CUDAIntTensor>(LU_pivots.pImpl,"LU_pivots",3, false);
    THCudaDoubleTensor_btrisolve(context->thc_state, result_->tensor, self_->tensor, LU_data_->tensor, LU_pivots_->tensor);
    result_->maybeScalar(self_->isScalar() && LU_data_->isScalar() && LU_pivots_->isScalar());
    return result;
}
Tensor CUDADoubleType::btrisolve(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto LU_data_ = checked_cast_tensor<CUDADoubleTensor>(LU_data.pImpl,"LU_data",2, false);
    auto LU_pivots_ = checked_cast_tensor<CUDAIntTensor>(LU_pivots.pImpl,"LU_pivots",3, false);
    THCudaDoubleTensor_btrisolve(context->thc_state, result_->tensor, self_->tensor, LU_data_->tensor, LU_pivots_->tensor);
    result_->maybeScalar(self_->isScalar() && LU_data_->isScalar() && LU_pivots_->isScalar());
    return result;
}
Tensor & CUDADoubleType::random_(Tensor & self, int64_t from, int64_t to, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    THCudaDoubleTensor_clampedRandom(context->thc_state, self_->tensor, from, to);
    return self;
}
Tensor & CUDADoubleType::random_(Tensor & self, int64_t to, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    THCudaDoubleTensor_cappedRandom(context->thc_state, self_->tensor, to);
    return self;
}
Tensor & CUDADoubleType::random_(Tensor & self, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    THCudaDoubleTensor_random(context->thc_state, self_->tensor);
    return self;
}
Tensor & CUDADoubleType::multinomial_out(Tensor & result, const Tensor & self, int64_t num_samples, bool replacement, Generator * generator) const {
    auto result_ = checked_cast_tensor<CUDALongTensor>(result.pImpl,"result",0, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    THCudaDoubleTensor_multinomial(context->thc_state, result_->tensor, self_->tensor, num_samples, replacement);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDADoubleType::multinomial(const Tensor & self, int64_t num_samples, bool replacement, Generator * generator) const {
    auto result_ = new CUDALongTensor(context);
    auto result = Tensor(result_, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    THCudaDoubleTensor_multinomial(context->thc_state, result_->tensor, self_->tensor, num_samples, replacement);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::uniform_(Tensor & self, double from, double to, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    THCudaDoubleTensor_uniform(context->thc_state, self_->tensor, from, to);
    return self;
}
Tensor & CUDADoubleType::normal_out(Tensor & output, const Tensor & mean, double std, Generator * generator) const {
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",0, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    auto mean_ = checked_cast_tensor<CUDADoubleTensor>(mean.pImpl,"mean",2, false);
    THCudaDoubleTensor_normal_means(context->thc_state, output_->tensor, mean_->tensor, std);
    output_->maybeScalar(mean_->isScalar());
    return output;
}
Tensor CUDADoubleType::normal(const Tensor & mean, double std, Generator * generator) const {
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    auto mean_ = checked_cast_tensor<CUDADoubleTensor>(mean.pImpl,"mean",2, false);
    THCudaDoubleTensor_normal_means(context->thc_state, output_->tensor, mean_->tensor, std);
    output_->maybeScalar(mean_->isScalar());
    return output;
}
Tensor & CUDADoubleType::normal_out(Tensor & output, double mean, const Tensor & std, Generator * generator) const {
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",0, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    auto std_ = checked_cast_tensor<CUDADoubleTensor>(std.pImpl,"std",3, false);
    THCudaDoubleTensor_normal_stddevs(context->thc_state, output_->tensor, mean, std_->tensor);
    output_->maybeScalar(std_->isScalar());
    return output;
}
Tensor CUDADoubleType::normal(double mean, const Tensor & std, Generator * generator) const {
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    auto std_ = checked_cast_tensor<CUDADoubleTensor>(std.pImpl,"std",3, false);
    THCudaDoubleTensor_normal_stddevs(context->thc_state, output_->tensor, mean, std_->tensor);
    output_->maybeScalar(std_->isScalar());
    return output;
}
Tensor & CUDADoubleType::normal_out(Tensor & output, const Tensor & mean, const Tensor & std, Generator * generator) const {
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",0, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    auto mean_ = checked_cast_tensor<CUDADoubleTensor>(mean.pImpl,"mean",2, false);
    auto std_ = checked_cast_tensor<CUDADoubleTensor>(std.pImpl,"std",3, false);
    THCudaDoubleTensor_normal_means_stddevs(context->thc_state, output_->tensor, mean_->tensor, std_->tensor);
    output_->maybeScalar(mean_->isScalar() && std_->isScalar());
    return output;
}
Tensor CUDADoubleType::normal(const Tensor & mean, const Tensor & std, Generator * generator) const {
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    auto mean_ = checked_cast_tensor<CUDADoubleTensor>(mean.pImpl,"mean",2, false);
    auto std_ = checked_cast_tensor<CUDADoubleTensor>(std.pImpl,"std",3, false);
    THCudaDoubleTensor_normal_means_stddevs(context->thc_state, output_->tensor, mean_->tensor, std_->tensor);
    output_->maybeScalar(mean_->isScalar() && std_->isScalar());
    return output;
}
Tensor & CUDADoubleType::normal_(Tensor & self, double mean, double std, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    THCudaDoubleTensor_normal(context->thc_state, self_->tensor, mean, std);
    return self;
}
Tensor & CUDADoubleType::cauchy_(Tensor & self, double median, double sigma, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    THCudaDoubleTensor_cauchy(context->thc_state, self_->tensor, median, sigma);
    return self;
}
Tensor & CUDADoubleType::log_normal_(Tensor & self, double mean, double std, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    THCudaDoubleTensor_logNormal(context->thc_state, self_->tensor, mean, std);
    return self;
}
Tensor & CUDADoubleType::exponential_(Tensor & self, double lambd, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    THCudaDoubleTensor_exponential(context->thc_state, self_->tensor, lambd);
    return self;
}
Tensor & CUDADoubleType::geometric_(Tensor & self, double p, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    THCudaDoubleTensor_geometric(context->thc_state, self_->tensor, p);
    return self;
}
Tensor & CUDADoubleType::bernoulli_out(Tensor & output, const Tensor & self, Generator * generator) const {
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",0, false);
    output.resize_(self.sizes());
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    THCudaDoubleTensor_bernoulli_Tensor(context->thc_state, output_->tensor, self_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDADoubleType::bernoulli(const Tensor & self, Generator * generator) const {
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    output.resize_(self.sizes());
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    THCudaDoubleTensor_bernoulli_Tensor(context->thc_state, output_->tensor, self_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDADoubleType::tensor(Storage & storage, int64_t storageOffset, IntList size, IntList stride) const {
    auto storage_ = checked_cast_storage<CUDADoubleStorage>(&storage,"storage",1);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    return Tensor((new CUDADoubleTensor(context, THCudaDoubleTensor_newWithStorage(context->thc_state, storage_->storage, storageOffset, size_, stride_)))->maybeScalar(size.size() == 0),false);
}
Tensor CUDADoubleType::tensor(IntList size) const {
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new CUDADoubleTensor(context, THCudaDoubleTensor_newWithSize(context->thc_state, size_, NULL)))->maybeScalar(size.size() == 0),false);
}
Tensor CUDADoubleType::tensor(IntList size, IntList stride) const {
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    return Tensor((new CUDADoubleTensor(context, THCudaDoubleTensor_newWithSize(context->thc_state, size_, stride_)))->maybeScalar(size.size() == 0),false);
}
Tensor CUDADoubleType::tensor() const {
    return Tensor((new CUDADoubleTensor(context, THCudaDoubleTensor_new(context->thc_state))),false);
}
Tensor CUDADoubleType::alias(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    return Tensor((new CUDADoubleTensor(context, THCudaDoubleTensor_newWithTensor(context->thc_state, self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor & CUDADoubleType::_copy_ignoring_overlaps_(Tensor & self, const Tensor & src) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto src_ = checked_cast_tensor<CUDADoubleTensor>(src.pImpl,"src",2, false);
    THCudaDoubleTensor_copyIgnoringOverlaps(context->thc_state, self_->tensor, src_->tensor);
    return self;
}
Tensor & CUDADoubleType::as_strided_out(Tensor & result, const Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    auto result_ = checked_cast_tensor<CUDADoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    if (storage_offset == -1) {
      storage_offset = self_->tensor->storageOffset;
    }
    THCudaDoubleTensor_setStorage(context->thc_state, result_->tensor, self_->tensor->storage, storage_offset, size_, stride_);
    result_->maybeScalar(size.size() == 0);
    return result;
}
Tensor CUDADoubleType::as_strided(const Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    auto result_ = new CUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    if (storage_offset == -1) {
      storage_offset = self_->tensor->storageOffset;
    }
    THCudaDoubleTensor_setStorage(context->thc_state, result_->tensor, self_->tensor->storage, storage_offset, size_, stride_);
    result_->maybeScalar(size.size() == 0);
    return result;
}
Tensor & CUDADoubleType::as_strided_(Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    if (storage_offset == -1) {
      storage_offset = self_->tensor->storageOffset;
    }
    THCudaDoubleTensor_setStorage(context->thc_state, self_->tensor, self_->tensor->storage, storage_offset, size_, stride_);
    self_->maybeScalar(size.size() == 0);
    return self;
}
Tensor & CUDADoubleType::_cat_out(Tensor & self, TensorList tensors, int64_t dim) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",0, false);
    auto tensors_ = tensor_list_checked_cast<CUDADoubleTensor, Tensor, THCudaDoubleTensor>(tensors,"tensors",1);
    THCudaDoubleTensor_catArray(context->thc_state, self_->tensor, tensors_.data(), tensors_.size(), dim);
    return self;
}
Tensor CUDADoubleType::_cat(TensorList tensors, int64_t dim) const {
    auto self_ = new CUDADoubleTensor(context);
    auto self = Tensor(self_, false);
    auto tensors_ = tensor_list_checked_cast<CUDADoubleTensor, Tensor, THCudaDoubleTensor>(tensors,"tensors",1);
    THCudaDoubleTensor_catArray(context->thc_state, self_->tensor, tensors_.data(), tensors_.size(), dim);
    return self;
}
Tensor CUDADoubleType::_sparse_mask(const Tensor & self, SparseTensor mask) const {
    auto result_ = new SparseCUDADoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<SparseCUDADoubleTensor>(mask.tref.pImpl,"mask",2,false);
    THCudaDoubleTensor_sparseMask(context->thc_state, result_->tensor, self_->tensor, mask_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDADoubleType::binary_cross_entropy_forward_out(Tensor & output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDADoubleTensor>(target.pImpl,"target",2, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",3, true);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",5, false);
    THNN_CudaDoubleBCECriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, weight_ ? weight_->tensor : NULL, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor CUDADoubleType::binary_cross_entropy_forward(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDADoubleTensor>(target.pImpl,"target",2, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",3, true);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaDoubleBCECriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, weight_ ? weight_->tensor : NULL, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor & CUDADoubleType::binary_cross_entropy_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDADoubleTensor>(target.pImpl,"target",3, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",4, true);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",6, false);
    THNN_CudaDoubleBCECriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, weight_ ? weight_->tensor : NULL, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::binary_cross_entropy_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDADoubleTensor>(target.pImpl,"target",3, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",4, true);
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleBCECriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, weight_ ? weight_->tensor : NULL, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDADoubleType::kl_div_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDADoubleTensor>(target.pImpl,"target",2, false);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",4, false);
    THNN_CudaDoubleDistKLDivCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor CUDADoubleType::kl_div_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDADoubleTensor>(target.pImpl,"target",2, false);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaDoubleDistKLDivCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor & CUDADoubleType::kl_div_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDADoubleTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",5, false);
    THNN_CudaDoubleDistKLDivCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::kl_div_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDADoubleTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleDistKLDivCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDADoubleType::l1_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDADoubleTensor>(target.pImpl,"target",2, false);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",4, false);
    THNN_CudaDoubleAbsCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor CUDADoubleType::l1_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDADoubleTensor>(target.pImpl,"target",2, false);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaDoubleAbsCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor & CUDADoubleType::l1_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDADoubleTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",5, false);
    THNN_CudaDoubleAbsCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDADoubleTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleAbsCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDADoubleType::mse_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDADoubleTensor>(target.pImpl,"target",2, false);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",4, false);
    THNN_CudaDoubleMSECriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor CUDADoubleType::mse_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDADoubleTensor>(target.pImpl,"target",2, false);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaDoubleMSECriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor & CUDADoubleType::mse_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDADoubleTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",5, false);
    THNN_CudaDoubleMSECriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::mse_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDADoubleTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleMSECriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDADoubleType::multi_margin_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",2, false);
    auto p_ = p.toDouble();
    auto margin_ = margin.toDouble();
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",5, true);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",7, false);
    THNN_CudaDoubleMultiMarginCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, p_, weight_ ? weight_->tensor : NULL, margin_, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor CUDADoubleType::multi_margin_loss_forward(const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",2, false);
    auto p_ = p.toDouble();
    auto margin_ = margin.toDouble();
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",5, true);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaDoubleMultiMarginCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, p_, weight_ ? weight_->tensor : NULL, margin_, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor & CUDADoubleType::multi_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",3, false);
    auto p_ = p.toDouble();
    auto margin_ = margin.toDouble();
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",6, true);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",8, false);
    THNN_CudaDoubleMultiMarginCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, p_, weight_ ? weight_->tensor : NULL, margin_, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::multi_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",3, false);
    auto p_ = p.toDouble();
    auto margin_ = margin.toDouble();
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",6, true);
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleMultiMarginCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, p_, weight_ ? weight_->tensor : NULL, margin_, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &> CUDADoubleType::multilabel_margin_loss_forward_out(Tensor & output, Tensor & is_target, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",2, false);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",4, false);
    auto is_target_ = checked_cast_tensor<CUDADoubleTensor>(is_target.pImpl,"is_target",4, false);
    THNN_CudaDoubleMultiLabelMarginCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, is_target_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    is_target_->maybeScalar(target_->isScalar());
    return std::tuple<Tensor &, Tensor &>(output, is_target);
}
std::tuple<Tensor,Tensor> CUDADoubleType::multilabel_margin_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",2, false);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    auto is_target_ = new CUDADoubleTensor(context);
    auto is_target = Tensor(is_target_, false);
    THNN_CudaDoubleMultiLabelMarginCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, is_target_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    is_target_->maybeScalar(target_->isScalar());
    return std::tuple<Tensor, Tensor>(output, is_target);
}
Tensor & CUDADoubleType::multilabel_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce, const Tensor & is_target) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",3, false);
    auto is_target_ = checked_cast_tensor<CUDADoubleTensor>(is_target.pImpl,"is_target",6, false);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",6, false);
    THNN_CudaDoubleMultiLabelMarginCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, is_target_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::multilabel_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce, const Tensor & is_target) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",3, false);
    auto is_target_ = checked_cast_tensor<CUDADoubleTensor>(is_target.pImpl,"is_target",6, false);
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleMultiLabelMarginCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, is_target_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &> CUDADoubleType::nll_loss_forward_out(Tensor & output, Tensor & total_weight, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",2, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",3, true);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",6, false);
    auto total_weight_ = checked_cast_tensor<CUDADoubleTensor>(total_weight.pImpl,"total_weight",6, false);
    THNN_CudaDoubleClassNLLCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, weight_ ? weight_->tensor : NULL, total_weight_->tensor, ignore_index, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    total_weight_->maybeScalar(true);
    return std::tuple<Tensor &, Tensor &>(output, total_weight);
}
std::tuple<Tensor,Tensor> CUDADoubleType::nll_loss_forward(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",2, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",3, true);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    auto total_weight_ = new CUDADoubleTensor(context);
    auto total_weight = Tensor(total_weight_, false);
    THNN_CudaDoubleClassNLLCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, weight_ ? weight_->tensor : NULL, total_weight_->tensor, ignore_index, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    total_weight_->maybeScalar(true);
    return std::tuple<Tensor, Tensor>(output, total_weight);
}
Tensor & CUDADoubleType::nll_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",3, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",4, true);
    auto total_weight_ = checked_cast_tensor<CUDADoubleTensor>(total_weight.pImpl,"total_weight",8, false);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",8, false);
    THNN_CudaDoubleClassNLLCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, weight_ ? weight_->tensor : NULL, total_weight_->tensor, ignore_index, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::nll_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",3, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",4, true);
    auto total_weight_ = checked_cast_tensor<CUDADoubleTensor>(total_weight.pImpl,"total_weight",8, false);
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleClassNLLCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, weight_ ? weight_->tensor : NULL, total_weight_->tensor, ignore_index, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &> CUDADoubleType::nll_loss2d_forward_out(Tensor & output, Tensor & total_weight, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",2, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",3, true);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",6, false);
    auto total_weight_ = checked_cast_tensor<CUDADoubleTensor>(total_weight.pImpl,"total_weight",6, false);
    THNN_CudaDoubleSpatialClassNLLCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, weight_ ? weight_->tensor : NULL, total_weight_->tensor, ignore_index, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    total_weight_->maybeScalar(true);
    return std::tuple<Tensor &, Tensor &>(output, total_weight);
}
std::tuple<Tensor,Tensor> CUDADoubleType::nll_loss2d_forward(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",2, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",3, true);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    auto total_weight_ = new CUDADoubleTensor(context);
    auto total_weight = Tensor(total_weight_, false);
    THNN_CudaDoubleSpatialClassNLLCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, weight_ ? weight_->tensor : NULL, total_weight_->tensor, ignore_index, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    total_weight_->maybeScalar(true);
    return std::tuple<Tensor, Tensor>(output, total_weight);
}
Tensor & CUDADoubleType::nll_loss2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",3, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",4, true);
    auto total_weight_ = checked_cast_tensor<CUDADoubleTensor>(total_weight.pImpl,"total_weight",8, false);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",8, false);
    THNN_CudaDoubleSpatialClassNLLCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, weight_ ? weight_->tensor : NULL, total_weight_->tensor, ignore_index, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::nll_loss2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDALongTensor>(target.pImpl,"target",3, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",4, true);
    auto total_weight_ = checked_cast_tensor<CUDADoubleTensor>(total_weight.pImpl,"total_weight",8, false);
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleSpatialClassNLLCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, weight_ ? weight_->tensor : NULL, total_weight_->tensor, ignore_index, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDADoubleType::smooth_l1_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDADoubleTensor>(target.pImpl,"target",2, false);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",4, false);
    THNN_CudaDoubleSmoothL1Criterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor CUDADoubleType::smooth_l1_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDADoubleTensor>(target.pImpl,"target",2, false);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaDoubleSmoothL1Criterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor & CUDADoubleType::smooth_l1_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDADoubleTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",5, false);
    THNN_CudaDoubleSmoothL1Criterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::smooth_l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDADoubleTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleSmoothL1Criterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDADoubleType::soft_margin_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDADoubleTensor>(target.pImpl,"target",2, false);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",4, false);
    THNN_CudaDoubleSoftMarginCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor CUDADoubleType::soft_margin_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CUDADoubleTensor>(target.pImpl,"target",2, false);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaDoubleSoftMarginCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor & CUDADoubleType::soft_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDADoubleTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",5, false);
    THNN_CudaDoubleSoftMarginCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::soft_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CUDADoubleTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleSoftMarginCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDADoubleType::elu_forward_out(Tensor & output, const Tensor & self, Scalar alpha, Scalar scale) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toDouble();
    auto scale_ = scale.toDouble();
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",3, false);
    THNN_CudaDoubleELU_updateOutput(context->thc_state, self_->tensor, output_->tensor, alpha_, scale_, false);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDADoubleType::elu_forward(const Tensor & self, Scalar alpha, Scalar scale) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toDouble();
    auto scale_ = scale.toDouble();
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaDoubleELU_updateOutput(context->thc_state, self_->tensor, output_->tensor, alpha_, scale_, false);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDADoubleType::elu_backward_out(Tensor & grad_input, const Tensor & grad_output, Scalar alpha, Scalar scale, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto alpha_ = alpha.toDouble();
    auto scale_ = scale.toDouble();
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",4, false);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_CudaDoubleELU_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, output_->tensor, alpha_, scale_);
    grad_input_->maybeScalar(output_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::elu_backward(const Tensor & grad_output, Scalar alpha, Scalar scale, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto alpha_ = alpha.toDouble();
    auto scale_ = scale.toDouble();
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",4, false);
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleELU_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, output_->tensor, alpha_, scale_);
    grad_input_->maybeScalar(output_->isScalar());
    return grad_input;
}
Tensor & CUDADoubleType::elu_forward_(Tensor & self, Scalar alpha, Scalar scale) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toDouble();
    auto scale_ = scale.toDouble();
    THNN_CudaDoubleELU_updateOutput(context->thc_state, self_->tensor, self_->tensor, alpha_, scale_, true);
    return self;
}
Tensor & CUDADoubleType::glu_forward_out(Tensor & output, const Tensor & self, int64_t dim) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",2, false);
    THNN_CudaDoubleGatedLinear_updateOutput(context->thc_state, self_->tensor, output_->tensor, dim);
    output_->maybeScalar(false);
    return output;
}
Tensor CUDADoubleType::glu_forward(const Tensor & self, int64_t dim) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaDoubleGatedLinear_updateOutput(context->thc_state, self_->tensor, output_->tensor, dim);
    output_->maybeScalar(false);
    return output;
}
Tensor & CUDADoubleType::glu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    dim = maybe_wrap_dim(dim, self_);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaDoubleGatedLinear_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, dim);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::glu_backward(const Tensor & grad_output, const Tensor & self, int64_t dim) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    dim = maybe_wrap_dim(dim, self_);
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleGatedLinear_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, dim);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDADoubleType::hardtanh_forward_out(Tensor & output, const Tensor & self, Scalar min_val, Scalar max_val) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto min_val_ = min_val.toDouble();
    auto max_val_ = max_val.toDouble();
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",3, false);
    THNN_CudaDoubleHardTanh_updateOutput(context->thc_state, self_->tensor, output_->tensor, min_val_, max_val_, false);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDADoubleType::hardtanh_forward(const Tensor & self, Scalar min_val, Scalar max_val) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto min_val_ = min_val.toDouble();
    auto max_val_ = max_val.toDouble();
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaDoubleHardTanh_updateOutput(context->thc_state, self_->tensor, output_->tensor, min_val_, max_val_, false);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDADoubleType::hardtanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto min_val_ = min_val.toDouble();
    auto max_val_ = max_val.toDouble();
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_CudaDoubleHardTanh_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, min_val_, max_val_, false);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::hardtanh_backward(const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto min_val_ = min_val.toDouble();
    auto max_val_ = max_val.toDouble();
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleHardTanh_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, min_val_, max_val_, false);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDADoubleType::hardtanh_forward_(Tensor & self, Scalar min_val, Scalar max_val) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto min_val_ = min_val.toDouble();
    auto max_val_ = max_val.toDouble();
    THNN_CudaDoubleHardTanh_updateOutput(context->thc_state, self_->tensor, self_->tensor, min_val_, max_val_, true);
    return self;
}
Tensor & CUDADoubleType::leaky_relu_forward_out(Tensor & output, const Tensor & self, Scalar negative_slope) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto negative_slope_ = negative_slope.toDouble();
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",2, false);
    THNN_CudaDoubleLeakyReLU_updateOutput(context->thc_state, self_->tensor, output_->tensor, negative_slope_, false);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDADoubleType::leaky_relu_forward(const Tensor & self, Scalar negative_slope) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto negative_slope_ = negative_slope.toDouble();
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaDoubleLeakyReLU_updateOutput(context->thc_state, self_->tensor, output_->tensor, negative_slope_, false);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDADoubleType::leaky_relu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar negative_slope) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto negative_slope_ = negative_slope.toDouble();
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaDoubleLeakyReLU_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, negative_slope_, false);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::leaky_relu_backward(const Tensor & grad_output, const Tensor & self, Scalar negative_slope) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto negative_slope_ = negative_slope.toDouble();
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleLeakyReLU_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, negative_slope_, false);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDADoubleType::leaky_relu_forward_(Tensor & self, Scalar negative_slope) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto negative_slope_ = negative_slope.toDouble();
    THNN_CudaDoubleLeakyReLU_updateOutput(context->thc_state, self_->tensor, self_->tensor, negative_slope_, true);
    return self;
}
std::tuple<Tensor &,Tensor &> CUDADoubleType::log_sigmoid_forward_out(Tensor & output, Tensor & buffer, const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",1, false);
    auto buffer_ = checked_cast_tensor<CUDADoubleTensor>(buffer.pImpl,"buffer",1, false);
    THNN_CudaDoubleLogSigmoid_updateOutput(context->thc_state, self_->tensor, output_->tensor, buffer_->tensor);
    output_->maybeScalar(self_->isScalar());
    buffer_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &>(output, buffer);
}
std::tuple<Tensor,Tensor> CUDADoubleType::log_sigmoid_forward(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    auto buffer_ = new CUDADoubleTensor(context);
    auto buffer = Tensor(buffer_, false);
    THNN_CudaDoubleLogSigmoid_updateOutput(context->thc_state, self_->tensor, output_->tensor, buffer_->tensor);
    output_->maybeScalar(self_->isScalar());
    buffer_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor>(output, buffer);
}
Tensor & CUDADoubleType::log_sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & buffer) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto buffer_ = checked_cast_tensor<CUDADoubleTensor>(buffer.pImpl,"buffer",3, false);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaDoubleLogSigmoid_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, buffer_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::log_sigmoid_backward(const Tensor & grad_output, const Tensor & self, const Tensor & buffer) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto buffer_ = checked_cast_tensor<CUDADoubleTensor>(buffer.pImpl,"buffer",3, false);
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleLogSigmoid_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, buffer_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDADoubleType::log_softmax_forward_out(Tensor & output, const Tensor & self, int64_t dim) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",2, false);
    THNN_CudaDoubleLogSoftMax_updateOutput(context->thc_state, self_->tensor, output_->tensor, dim);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDADoubleType::log_softmax_forward(const Tensor & self, int64_t dim) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaDoubleLogSoftMax_updateOutput(context->thc_state, self_->tensor, output_->tensor, dim);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDADoubleType::log_softmax_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",4, false);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_CudaDoubleLogSoftMax_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, output_->tensor, dim);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::log_softmax_backward(const Tensor & grad_output, const Tensor & self, int64_t dim, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",4, false);
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleLogSoftMax_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, output_->tensor, dim);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDADoubleType::prelu_forward_out(Tensor & output, const Tensor & self, const Tensor & weight) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",2, false);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",2, false);
    THNN_CudaDoublePReLU_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDADoubleType::prelu_forward(const Tensor & self, const Tensor & weight) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",2, false);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaDoublePReLU_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
std::tuple<Tensor &,Tensor &> CUDADoubleType::prelu_backward_out(Tensor & grad_input, Tensor & grad_weight, const Tensor & grad_output, const Tensor & self, const Tensor & weight) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",3, false);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",3, true);
    auto grad_weight_ = checked_cast_tensor<CUDADoubleTensor>(grad_weight.pImpl,"grad_weight",3, true);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    if (grad_input_) THNN_CudaDoublePReLU_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor);
    if (grad_weight_) THNN_CudaDoublePReLU_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &>(grad_input, grad_weight);
}
std::tuple<Tensor,Tensor> CUDADoubleType::prelu_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, std::array<bool,2> output_mask) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",3, false);
    auto grad_input_ = output_mask[0] ? new CUDADoubleTensor(context) : nullptr;
    auto grad_input = Tensor(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_input_, false);
    auto grad_weight_ = output_mask[1] ? new CUDADoubleTensor(context) : nullptr;
    auto grad_weight = Tensor(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_weight_, false);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    if (grad_input_) THNN_CudaDoublePReLU_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor);
    if (grad_weight_) THNN_CudaDoublePReLU_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor>(grad_input, grad_weight);
}
Tensor & CUDADoubleType::rrelu_with_noise_forward_out(Tensor & output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto noise_ = checked_cast_tensor<CUDADoubleTensor>(noise.pImpl,"noise",2, false);
    auto lower_ = lower.toDouble();
    auto upper_ = upper.toDouble();
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",6, false);
    THNN_CudaDoubleRReLU_updateOutput(context->thc_state, self_->tensor, output_->tensor, noise_->tensor, lower_, upper_, training, false, NULL);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDADoubleType::rrelu_with_noise_forward(const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto noise_ = checked_cast_tensor<CUDADoubleTensor>(noise.pImpl,"noise",2, false);
    auto lower_ = lower.toDouble();
    auto upper_ = upper.toDouble();
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaDoubleRReLU_updateOutput(context->thc_state, self_->tensor, output_->tensor, noise_->tensor, lower_, upper_, training, false, NULL);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDADoubleType::rrelu_with_noise_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto noise_ = checked_cast_tensor<CUDADoubleTensor>(noise.pImpl,"noise",3, false);
    auto lower_ = lower.toDouble();
    auto upper_ = upper.toDouble();
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",6, false);
    THNN_CudaDoubleRReLU_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, noise_->tensor, lower_, upper_, training, false);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::rrelu_with_noise_backward(const Tensor & grad_output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto noise_ = checked_cast_tensor<CUDADoubleTensor>(noise.pImpl,"noise",3, false);
    auto lower_ = lower.toDouble();
    auto upper_ = upper.toDouble();
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleRReLU_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, noise_->tensor, lower_, upper_, training, false);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDADoubleType::rrelu_with_noise_forward_(Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto noise_ = checked_cast_tensor<CUDADoubleTensor>(noise.pImpl,"noise",2, false);
    auto lower_ = lower.toDouble();
    auto upper_ = upper.toDouble();
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    THNN_CudaDoubleRReLU_updateOutput(context->thc_state, self_->tensor, self_->tensor, noise_->tensor, lower_, upper_, training, true, NULL);
    return self;
}
Tensor & CUDADoubleType::softmax_forward_out(Tensor & output, const Tensor & self, int64_t dim) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",2, false);
    THNN_CudaDoubleSoftMax_updateOutput(context->thc_state, self_->tensor, output_->tensor, dim);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDADoubleType::softmax_forward(const Tensor & self, int64_t dim) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaDoubleSoftMax_updateOutput(context->thc_state, self_->tensor, output_->tensor, dim);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDADoubleType::softmax_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",4, false);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_CudaDoubleSoftMax_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, output_->tensor, dim);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::softmax_backward(const Tensor & grad_output, const Tensor & self, int64_t dim, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",4, false);
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleSoftMax_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, output_->tensor, dim);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDADoubleType::softplus_forward_out(Tensor & output, const Tensor & self, Scalar beta, Scalar threshold) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toDouble();
    auto threshold_ = threshold.toDouble();
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",3, false);
    THNN_CudaDoubleSoftPlus_updateOutput(context->thc_state, self_->tensor, output_->tensor, beta_, threshold_);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDADoubleType::softplus_forward(const Tensor & self, Scalar beta, Scalar threshold) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toDouble();
    auto threshold_ = threshold.toDouble();
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaDoubleSoftPlus_updateOutput(context->thc_state, self_->tensor, output_->tensor, beta_, threshold_);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDADoubleType::softplus_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto beta_ = beta.toDouble();
    auto threshold_ = threshold.toDouble();
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",5, false);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",5, false);
    THNN_CudaDoubleSoftPlus_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, output_->tensor, beta_, threshold_);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::softplus_backward(const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto beta_ = beta.toDouble();
    auto threshold_ = threshold.toDouble();
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",5, false);
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleSoftPlus_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, output_->tensor, beta_, threshold_);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDADoubleType::softshrink_forward_out(Tensor & output, const Tensor & self, Scalar lambd) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto lambd_ = lambd.toDouble();
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",2, false);
    THNN_CudaDoubleSoftShrink_updateOutput(context->thc_state, self_->tensor, output_->tensor, lambd_);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDADoubleType::softshrink_forward(const Tensor & self, Scalar lambd) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto lambd_ = lambd.toDouble();
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaDoubleSoftShrink_updateOutput(context->thc_state, self_->tensor, output_->tensor, lambd_);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDADoubleType::softshrink_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar lambd) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto lambd_ = lambd.toDouble();
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaDoubleSoftShrink_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, lambd_);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::softshrink_backward(const Tensor & grad_output, const Tensor & self, Scalar lambd) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto lambd_ = lambd.toDouble();
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleSoftShrink_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, lambd_);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDADoubleType::threshold_forward_out(Tensor & output, const Tensor & self, Scalar threshold, Scalar value) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto threshold_ = threshold.toDouble();
    auto value_ = value.toDouble();
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",3, false);
    THNN_CudaDoubleThreshold_updateOutput(context->thc_state, self_->tensor, output_->tensor, threshold_, value_, false);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDADoubleType::threshold_forward(const Tensor & self, Scalar threshold, Scalar value) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto threshold_ = threshold.toDouble();
    auto value_ = value.toDouble();
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaDoubleThreshold_updateOutput(context->thc_state, self_->tensor, output_->tensor, threshold_, value_, false);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDADoubleType::threshold_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar threshold, Scalar value) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto threshold_ = threshold.toDouble();
    auto value_ = value.toDouble();
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_CudaDoubleThreshold_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, threshold_, value_, false);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::threshold_backward(const Tensor & grad_output, const Tensor & self, Scalar threshold, Scalar value) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto threshold_ = threshold.toDouble();
    auto value_ = value.toDouble();
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleThreshold_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, threshold_, value_, false);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDADoubleType::threshold_forward_(Tensor & self, Scalar threshold, Scalar value) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto threshold_ = threshold.toDouble();
    auto value_ = value.toDouble();
    THNN_CudaDoubleThreshold_updateOutput(context->thc_state, self_->tensor, self_->tensor, threshold_, value_, true);
    return self;
}
Tensor & CUDADoubleType::adaptive_avg_pool2d_forward_out(Tensor & output, const Tensor & self, IntList output_size) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 2);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",2, false);
    THNN_CudaDoubleSpatialAdaptiveAveragePooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[1], output_size_[0]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDADoubleType::adaptive_avg_pool2d_forward(const Tensor & self, IntList output_size) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 2);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaDoubleSpatialAdaptiveAveragePooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[1], output_size_[0]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDADoubleType::adaptive_avg_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",2, false);
    THNN_CudaDoubleSpatialAdaptiveAveragePooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::adaptive_avg_pool2d_backward(const Tensor & grad_output, const Tensor & self) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleSpatialAdaptiveAveragePooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDADoubleType::adaptive_avg_pool3d_forward_out(Tensor & output, const Tensor & self, IntList output_size) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 2);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",2, false);
    THNN_CudaDoubleVolumetricAdaptiveAveragePooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[0], output_size_[2], output_size_[1]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDADoubleType::adaptive_avg_pool3d_forward(const Tensor & self, IntList output_size) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 2);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaDoubleVolumetricAdaptiveAveragePooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[0], output_size_[2], output_size_[1]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDADoubleType::adaptive_avg_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",2, false);
    THNN_CudaDoubleVolumetricAdaptiveAveragePooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::adaptive_avg_pool3d_backward(const Tensor & grad_output, const Tensor & self) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleVolumetricAdaptiveAveragePooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &> CUDADoubleType::adaptive_max_pool2d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList output_size) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 2);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",2, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",2, false);
    THNN_CudaDoubleSpatialAdaptiveMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, output_size_[1], output_size_[0]);
    bool maybe_scalar = self_->isScalar();
    output_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(output, indices);
}
std::tuple<Tensor,Tensor> CUDADoubleType::adaptive_max_pool2d_forward(const Tensor & self, IntList output_size) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 2);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    auto indices_ = new CUDALongTensor(context);
    auto indices = Tensor(indices_, false);
    THNN_CudaDoubleSpatialAdaptiveMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, output_size_[1], output_size_[0]);
    bool maybe_scalar = self_->isScalar();
    output_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(output, indices);
}
Tensor & CUDADoubleType::adaptive_max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",3, false);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaDoubleSpatialAdaptiveMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::adaptive_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",3, false);
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleSpatialAdaptiveMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &> CUDADoubleType::adaptive_max_pool3d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList output_size) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 2);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",2, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",2, false);
    THNN_CudaDoubleVolumetricAdaptiveMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, output_size_[0], output_size_[2], output_size_[1]);
    bool maybe_scalar = self_->isScalar();
    output_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(output, indices);
}
std::tuple<Tensor,Tensor> CUDADoubleType::adaptive_max_pool3d_forward(const Tensor & self, IntList output_size) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 2);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    auto indices_ = new CUDALongTensor(context);
    auto indices = Tensor(indices_, false);
    THNN_CudaDoubleVolumetricAdaptiveMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, output_size_[0], output_size_[2], output_size_[1]);
    bool maybe_scalar = self_->isScalar();
    output_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(output, indices);
}
Tensor & CUDADoubleType::adaptive_max_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",3, false);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaDoubleVolumetricAdaptiveMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::adaptive_max_pool3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",3, false);
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleVolumetricAdaptiveMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDADoubleType::avg_pool2d_forward_out(Tensor & output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 2);
    auto stride_ = check_intlist<2>(stride, "stride", 3, kernel_size);
    auto padding_ = check_intlist<2>(padding, "padding", 4);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",6, false);
    THNN_CudaDoubleSpatialAveragePooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], ceil_mode, count_include_pad);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDADoubleType::avg_pool2d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 2);
    auto stride_ = check_intlist<2>(stride, "stride", 3, kernel_size);
    auto padding_ = check_intlist<2>(padding, "padding", 4);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaDoubleSpatialAveragePooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], ceil_mode, count_include_pad);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDADoubleType::avg_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto stride_ = check_intlist<2>(stride, "stride", 4, kernel_size);
    auto padding_ = check_intlist<2>(padding, "padding", 5);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",7, false);
    THNN_CudaDoubleSpatialAveragePooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], ceil_mode, count_include_pad);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::avg_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto stride_ = check_intlist<2>(stride, "stride", 4, kernel_size);
    auto padding_ = check_intlist<2>(padding, "padding", 5);
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleSpatialAveragePooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], ceil_mode, count_include_pad);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDADoubleType::avg_pool3d_forward_out(Tensor & output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 2);
    auto stride_ = check_intlist<3>(stride, "stride", 3, kernel_size);
    auto padding_ = check_intlist<3>(padding, "padding", 4);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",6, false);
    THNN_CudaDoubleVolumetricAveragePooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], ceil_mode, count_include_pad);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDADoubleType::avg_pool3d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 2);
    auto stride_ = check_intlist<3>(stride, "stride", 3, kernel_size);
    auto padding_ = check_intlist<3>(padding, "padding", 4);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaDoubleVolumetricAveragePooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], ceil_mode, count_include_pad);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDADoubleType::avg_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto stride_ = check_intlist<3>(stride, "stride", 4, kernel_size);
    auto padding_ = check_intlist<3>(padding, "padding", 5);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",7, false);
    THNN_CudaDoubleVolumetricAveragePooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], ceil_mode, count_include_pad);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::avg_pool3d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto stride_ = check_intlist<3>(stride, "stride", 4, kernel_size);
    auto padding_ = check_intlist<3>(padding, "padding", 5);
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleVolumetricAveragePooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], ceil_mode, count_include_pad);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &> CUDADoubleType::fractional_max_pool2d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & random_samples) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 2);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 3);
    auto random_samples_ = checked_cast_tensor<CUDADoubleTensor>(random_samples.pImpl,"random_samples",4, false);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",4, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",4, false);
    THNN_CudaDoubleSpatialFractionalMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[1], output_size_[0], kernel_size_[1], kernel_size_[0], indices_->tensor, random_samples_->tensor);
    output_->maybeScalar(false);
    return std::tuple<Tensor &, Tensor &>(output, indices);
}
std::tuple<Tensor,Tensor> CUDADoubleType::fractional_max_pool2d_forward(const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & random_samples) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 2);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 3);
    auto random_samples_ = checked_cast_tensor<CUDADoubleTensor>(random_samples.pImpl,"random_samples",4, false);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    auto indices_ = new CUDALongTensor(context);
    auto indices = Tensor(indices_, false);
    THNN_CudaDoubleSpatialFractionalMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[1], output_size_[0], kernel_size_[1], kernel_size_[0], indices_->tensor, random_samples_->tensor);
    output_->maybeScalar(false);
    return std::tuple<Tensor, Tensor>(output, indices);
}
Tensor & CUDADoubleType::fractional_max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 4);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",5, false);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",5, false);
    THNN_CudaDoubleSpatialFractionalMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, output_size_[1], output_size_[0], kernel_size_[1], kernel_size_[0], indices_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::fractional_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 4);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",5, false);
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleSpatialFractionalMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, output_size_[1], output_size_[0], kernel_size_[1], kernel_size_[0], indices_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &> CUDADoubleType::max_pool2d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 2);
    auto stride_ = check_intlist<2>(stride, "stride", 3, kernel_size);
    auto padding_ = check_intlist<2>(padding, "padding", 4);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 5);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",6, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",6, false);
    THNN_CudaDoubleSpatialDilatedMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], ceil_mode);
    bool maybe_scalar = self_->isScalar();
    output_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(output, indices);
}
std::tuple<Tensor,Tensor> CUDADoubleType::max_pool2d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 2);
    auto stride_ = check_intlist<2>(stride, "stride", 3, kernel_size);
    auto padding_ = check_intlist<2>(padding, "padding", 4);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 5);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    auto indices_ = new CUDALongTensor(context);
    auto indices = Tensor(indices_, false);
    THNN_CudaDoubleSpatialDilatedMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], ceil_mode);
    bool maybe_scalar = self_->isScalar();
    output_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(output, indices);
}
Tensor & CUDADoubleType::max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto stride_ = check_intlist<2>(stride, "stride", 4, kernel_size);
    auto padding_ = check_intlist<2>(padding, "padding", 5);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 6);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",8, false);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",8, false);
    THNN_CudaDoubleSpatialDilatedMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], ceil_mode);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::max_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto stride_ = check_intlist<2>(stride, "stride", 4, kernel_size);
    auto padding_ = check_intlist<2>(padding, "padding", 5);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 6);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",8, false);
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleSpatialDilatedMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], ceil_mode);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &> CUDADoubleType::max_pool3d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 2);
    auto stride_ = check_intlist<3>(stride, "stride", 3, kernel_size);
    auto padding_ = check_intlist<3>(padding, "padding", 4);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 5);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",6, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",6, false);
    THNN_CudaDoubleVolumetricDilatedMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], ceil_mode);
    bool maybe_scalar = self_->isScalar();
    output_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(output, indices);
}
std::tuple<Tensor,Tensor> CUDADoubleType::max_pool3d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 2);
    auto stride_ = check_intlist<3>(stride, "stride", 3, kernel_size);
    auto padding_ = check_intlist<3>(padding, "padding", 4);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 5);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    auto indices_ = new CUDALongTensor(context);
    auto indices = Tensor(indices_, false);
    THNN_CudaDoubleVolumetricDilatedMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], ceil_mode);
    bool maybe_scalar = self_->isScalar();
    output_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(output, indices);
}
Tensor & CUDADoubleType::max_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto stride_ = check_intlist<3>(stride, "stride", 4, kernel_size);
    auto padding_ = check_intlist<3>(padding, "padding", 5);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 6);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",8, false);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",8, false);
    THNN_CudaDoubleVolumetricDilatedMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], ceil_mode);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::max_pool3d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto stride_ = check_intlist<3>(stride, "stride", 4, kernel_size);
    auto padding_ = check_intlist<3>(padding, "padding", 5);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 6);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",8, false);
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleVolumetricDilatedMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], ceil_mode);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDADoubleType::max_unpool2d_forward_out(Tensor & output, const Tensor & self, const Tensor & indices, IntList output_size) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",2, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 3);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",3, false);
    THNN_CudaDoubleSpatialMaxUnpooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, output_size_[1], output_size_[0]);
    output_->maybeScalar(self_->isScalar() && indices_->isScalar());
    return output;
}
Tensor CUDADoubleType::max_unpool2d_forward(const Tensor & self, const Tensor & indices, IntList output_size) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",2, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 3);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaDoubleSpatialMaxUnpooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, output_size_[1], output_size_[0]);
    output_->maybeScalar(self_->isScalar() && indices_->isScalar());
    return output;
}
Tensor & CUDADoubleType::max_unpool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",3, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 4);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_CudaDoubleSpatialMaxUnpooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor, output_size_[1], output_size_[0]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::max_unpool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",3, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 4);
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleSpatialMaxUnpooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor, output_size_[1], output_size_[0]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDADoubleType::max_unpool3d_forward_out(Tensor & output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",2, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 3);
    auto stride_ = check_intlist<3>(stride, "stride", 4);
    auto padding_ = check_intlist<3>(padding, "padding", 5);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",5, false);
    THNN_CudaDoubleVolumetricMaxUnpooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, output_size_[0], output_size_[2], output_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1]);
    output_->maybeScalar(self_->isScalar() && indices_->isScalar());
    return output;
}
Tensor CUDADoubleType::max_unpool3d_forward(const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",2, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 3);
    auto stride_ = check_intlist<3>(stride, "stride", 4);
    auto padding_ = check_intlist<3>(padding, "padding", 5);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaDoubleVolumetricMaxUnpooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, output_size_[0], output_size_[2], output_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1]);
    output_->maybeScalar(self_->isScalar() && indices_->isScalar());
    return output;
}
Tensor & CUDADoubleType::max_unpool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",3, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 4);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",6, false);
    THNN_CudaDoubleVolumetricMaxUnpooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor, output_size_[0], output_size_[2], output_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::max_unpool3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",3, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 4);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleVolumetricMaxUnpooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor, output_size_[0], output_size_[2], output_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDADoubleType::reflection_pad1d_forward_out(Tensor & output, const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<2>(padding, "padding", 2);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",2, false);
    THNN_CudaDoubleTemporalReflectionPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDADoubleType::reflection_pad1d_forward(const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<2>(padding, "padding", 2);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaDoubleTemporalReflectionPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDADoubleType::reflection_pad1d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<2>(padding, "padding", 3);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaDoubleTemporalReflectionPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::reflection_pad1d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<2>(padding, "padding", 3);
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleTemporalReflectionPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDADoubleType::reflection_pad2d_forward_out(Tensor & output, const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<4>(padding, "padding", 2);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",2, false);
    THNN_CudaDoubleSpatialReflectionPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1], padding_[2], padding_[3]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDADoubleType::reflection_pad2d_forward(const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<4>(padding, "padding", 2);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaDoubleSpatialReflectionPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1], padding_[2], padding_[3]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDADoubleType::reflection_pad2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<4>(padding, "padding", 3);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaDoubleSpatialReflectionPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1], padding_[2], padding_[3]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::reflection_pad2d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<4>(padding, "padding", 3);
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleSpatialReflectionPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1], padding_[2], padding_[3]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDADoubleType::replication_pad1d_forward_out(Tensor & output, const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<2>(padding, "padding", 2);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",2, false);
    THNN_CudaDoubleTemporalReplicationPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDADoubleType::replication_pad1d_forward(const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<2>(padding, "padding", 2);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaDoubleTemporalReplicationPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDADoubleType::replication_pad1d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<2>(padding, "padding", 3);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaDoubleTemporalReplicationPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::replication_pad1d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<2>(padding, "padding", 3);
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleTemporalReplicationPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDADoubleType::replication_pad2d_forward_out(Tensor & output, const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<4>(padding, "padding", 2);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",2, false);
    THNN_CudaDoubleSpatialReplicationPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1], padding_[2], padding_[3]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDADoubleType::replication_pad2d_forward(const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<4>(padding, "padding", 2);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaDoubleSpatialReplicationPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1], padding_[2], padding_[3]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDADoubleType::replication_pad2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<4>(padding, "padding", 3);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaDoubleSpatialReplicationPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1], padding_[2], padding_[3]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::replication_pad2d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<4>(padding, "padding", 3);
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleSpatialReplicationPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1], padding_[2], padding_[3]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDADoubleType::replication_pad3d_forward_out(Tensor & output, const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<6>(padding, "padding", 2);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",2, false);
    THNN_CudaDoubleVolumetricReplicationPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1], padding_[2], padding_[3], padding_[4], padding_[5]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDADoubleType::replication_pad3d_forward(const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<6>(padding, "padding", 2);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaDoubleVolumetricReplicationPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1], padding_[2], padding_[3], padding_[4], padding_[5]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDADoubleType::replication_pad3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<6>(padding, "padding", 3);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaDoubleVolumetricReplicationPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1], padding_[2], padding_[3], padding_[4], padding_[5]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::replication_pad3d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<6>(padding, "padding", 3);
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleVolumetricReplicationPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1], padding_[2], padding_[3], padding_[4], padding_[5]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDADoubleType::upsample_linear1d_forward_out(Tensor & output, const Tensor & self, IntList output_size, bool align_corners) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<1>(output_size, "output_size", 2);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",3, false);
    THNN_CudaDoubleTemporalUpSamplingLinear_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[0], align_corners);
    return output;
}
Tensor CUDADoubleType::upsample_linear1d_forward(const Tensor & self, IntList output_size, bool align_corners) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<1>(output_size, "output_size", 2);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaDoubleTemporalUpSamplingLinear_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[0], align_corners);
    return output;
}
Tensor & CUDADoubleType::upsample_linear1d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_size_ = check_intlist<1>(output_size, "output_size", 2);
    auto input_size_ = check_intlist<3>(input_size, "input_size", 3);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_CudaDoubleTemporalUpSamplingLinear_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, input_size_[0], input_size_[1], input_size_[2], output_size_[0], align_corners);
    grad_input_->maybeScalar(false);
    return grad_input;
}
Tensor CUDADoubleType::upsample_linear1d_backward(const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_size_ = check_intlist<1>(output_size, "output_size", 2);
    auto input_size_ = check_intlist<3>(input_size, "input_size", 3);
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleTemporalUpSamplingLinear_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, input_size_[0], input_size_[1], input_size_[2], output_size_[0], align_corners);
    grad_input_->maybeScalar(false);
    return grad_input;
}
Tensor & CUDADoubleType::upsample_bilinear2d_forward_out(Tensor & output, const Tensor & self, IntList output_size, bool align_corners) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 2);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",3, false);
    THNN_CudaDoubleSpatialUpSamplingBilinear_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[0], output_size_[1], align_corners);
    return output;
}
Tensor CUDADoubleType::upsample_bilinear2d_forward(const Tensor & self, IntList output_size, bool align_corners) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 2);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaDoubleSpatialUpSamplingBilinear_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[0], output_size_[1], align_corners);
    return output;
}
Tensor & CUDADoubleType::upsample_bilinear2d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 2);
    auto input_size_ = check_intlist<4>(input_size, "input_size", 3);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_CudaDoubleSpatialUpSamplingBilinear_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, input_size_[0], input_size_[1], input_size_[2], input_size_[3], output_size_[0], output_size_[1], align_corners);
    grad_input_->maybeScalar(false);
    return grad_input;
}
Tensor CUDADoubleType::upsample_bilinear2d_backward(const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 2);
    auto input_size_ = check_intlist<4>(input_size, "input_size", 3);
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleSpatialUpSamplingBilinear_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, input_size_[0], input_size_[1], input_size_[2], input_size_[3], output_size_[0], output_size_[1], align_corners);
    grad_input_->maybeScalar(false);
    return grad_input;
}
Tensor & CUDADoubleType::upsample_trilinear3d_forward_out(Tensor & output, const Tensor & self, IntList output_size, bool align_corners) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 2);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",3, false);
    THNN_CudaDoubleVolumetricUpSamplingTrilinear_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[0], output_size_[1], output_size_[2], align_corners);
    return output;
}
Tensor CUDADoubleType::upsample_trilinear3d_forward(const Tensor & self, IntList output_size, bool align_corners) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 2);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaDoubleVolumetricUpSamplingTrilinear_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[0], output_size_[1], output_size_[2], align_corners);
    return output;
}
Tensor & CUDADoubleType::upsample_trilinear3d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 2);
    auto input_size_ = check_intlist<5>(input_size, "input_size", 3);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_CudaDoubleVolumetricUpSamplingTrilinear_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, input_size_[0], input_size_[1], input_size_[2], input_size_[3], input_size_[4], output_size_[0], output_size_[1], output_size_[2], align_corners);
    grad_input_->maybeScalar(false);
    return grad_input;
}
Tensor CUDADoubleType::upsample_trilinear3d_backward(const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 2);
    auto input_size_ = check_intlist<5>(input_size, "input_size", 3);
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleVolumetricUpSamplingTrilinear_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, input_size_[0], input_size_[1], input_size_[2], input_size_[3], input_size_[4], output_size_[0], output_size_[1], output_size_[2], align_corners);
    grad_input_->maybeScalar(false);
    return grad_input;
}
Tensor & CUDADoubleType::upsample_nearest1d_forward_out(Tensor & output, const Tensor & self, int64_t scale_factor) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",2, false);
    THNN_CudaDoubleTemporalUpSamplingNearest_updateOutput(context->thc_state, self_->tensor, output_->tensor, scale_factor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDADoubleType::upsample_nearest1d_forward(const Tensor & self, int64_t scale_factor) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaDoubleTemporalUpSamplingNearest_updateOutput(context->thc_state, self_->tensor, output_->tensor, scale_factor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDADoubleType::upsample_nearest1d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaDoubleTemporalUpSamplingNearest_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, scale_factor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::upsample_nearest1d_backward(const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleTemporalUpSamplingNearest_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, scale_factor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDADoubleType::upsample_nearest2d_forward_out(Tensor & output, const Tensor & self, int64_t scale_factor) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",2, false);
    THNN_CudaDoubleSpatialUpSamplingNearest_updateOutput(context->thc_state, self_->tensor, output_->tensor, scale_factor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDADoubleType::upsample_nearest2d_forward(const Tensor & self, int64_t scale_factor) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaDoubleSpatialUpSamplingNearest_updateOutput(context->thc_state, self_->tensor, output_->tensor, scale_factor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDADoubleType::upsample_nearest2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaDoubleSpatialUpSamplingNearest_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, scale_factor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::upsample_nearest2d_backward(const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleSpatialUpSamplingNearest_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, scale_factor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDADoubleType::upsample_nearest3d_forward_out(Tensor & output, const Tensor & self, int64_t scale_factor) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",2, false);
    THNN_CudaDoubleVolumetricUpSamplingNearest_updateOutput(context->thc_state, self_->tensor, output_->tensor, scale_factor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDADoubleType::upsample_nearest3d_forward(const Tensor & self, int64_t scale_factor) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaDoubleVolumetricUpSamplingNearest_updateOutput(context->thc_state, self_->tensor, output_->tensor, scale_factor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDADoubleType::upsample_nearest3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_CudaDoubleVolumetricUpSamplingNearest_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, scale_factor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::upsample_nearest3d_backward(const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleVolumetricUpSamplingNearest_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, scale_factor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CUDADoubleType::_sigmoid_forward_out(Tensor & output, const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",1, false);
    THNN_CudaDoubleSigmoid_updateOutput(context->thc_state, self_->tensor, output_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDADoubleType::_sigmoid_forward(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaDoubleSigmoid_updateOutput(context->thc_state, self_->tensor, output_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDADoubleType::_sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",2, false);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",2, false);
    THNN_CudaDoubleSigmoid_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, output_->tensor);
    grad_input_->maybeScalar(output_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::_sigmoid_backward(const Tensor & grad_output, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",2, false);
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleSigmoid_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, output_->tensor);
    grad_input_->maybeScalar(output_->isScalar());
    return grad_input;
}
Tensor & CUDADoubleType::_tanh_forward_out(Tensor & output, const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",1, false);
    THNN_CudaDoubleTanh_updateOutput(context->thc_state, self_->tensor, output_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CUDADoubleType::_tanh_forward(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaDoubleTanh_updateOutput(context->thc_state, self_->tensor, output_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CUDADoubleType::_tanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",2, false);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",2, false);
    THNN_CudaDoubleTanh_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, output_->tensor);
    grad_input_->maybeScalar(output_->isScalar());
    return grad_input;
}
Tensor CUDADoubleType::_tanh_backward(const Tensor & grad_output, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",2, false);
    auto grad_input_ = new CUDADoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_CudaDoubleTanh_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, output_->tensor);
    grad_input_->maybeScalar(output_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDADoubleType::thnn_batch_norm_forward_out(Tensor & output, Tensor & save_mean, Tensor & save_std, const Tensor & self, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",2, true);
    auto bias_ = checked_cast_tensor<CUDADoubleTensor>(bias.pImpl,"bias",3, true);
    auto running_mean_ = checked_cast_tensor<CUDADoubleTensor>(running_mean.pImpl,"running_mean",4, true);
    auto running_var_ = checked_cast_tensor<CUDADoubleTensor>(running_var.pImpl,"running_var",5, true);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",8, false);
    auto save_mean_ = checked_cast_tensor<CUDADoubleTensor>(save_mean.pImpl,"save_mean",8, false);
    auto save_std_ = checked_cast_tensor<CUDADoubleTensor>(save_std.pImpl,"save_std",8, false);
    THNN_CudaDoubleBatchNormalization_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_ ? weight_->tensor : NULL, bias_ ? bias_->tensor : NULL, running_mean_ ? running_mean_->tensor : NULL, running_var_ ? running_var_->tensor : NULL, save_mean_->tensor, save_std_->tensor, training, momentum, eps);
    bool maybe_scalar = self_->isScalar() && (!weight_ || weight_->isScalar()) && (!bias_ || bias_->isScalar()) && (!running_mean_ || running_mean_->isScalar()) && (!running_var_ || running_var_->isScalar());
    output_->maybeScalar(maybe_scalar);
    save_mean_->maybeScalar(maybe_scalar);
    save_std_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(output, save_mean, save_std);
}
std::tuple<Tensor,Tensor,Tensor> CUDADoubleType::thnn_batch_norm_forward(const Tensor & self, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",2, true);
    auto bias_ = checked_cast_tensor<CUDADoubleTensor>(bias.pImpl,"bias",3, true);
    auto running_mean_ = checked_cast_tensor<CUDADoubleTensor>(running_mean.pImpl,"running_mean",4, true);
    auto running_var_ = checked_cast_tensor<CUDADoubleTensor>(running_var.pImpl,"running_var",5, true);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    auto save_mean_ = new CUDADoubleTensor(context);
    auto save_mean = Tensor(save_mean_, false);
    auto save_std_ = new CUDADoubleTensor(context);
    auto save_std = Tensor(save_std_, false);
    THNN_CudaDoubleBatchNormalization_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_ ? weight_->tensor : NULL, bias_ ? bias_->tensor : NULL, running_mean_ ? running_mean_->tensor : NULL, running_var_ ? running_var_->tensor : NULL, save_mean_->tensor, save_std_->tensor, training, momentum, eps);
    bool maybe_scalar = self_->isScalar() && (!weight_ || weight_->isScalar()) && (!bias_ || bias_->isScalar()) && (!running_mean_ || running_mean_->isScalar()) && (!running_var_ || running_var_->isScalar());
    output_->maybeScalar(maybe_scalar);
    save_mean_->maybeScalar(maybe_scalar);
    save_std_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(output, save_mean, save_std);
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDADoubleType::thnn_batch_norm_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, bool training, double eps, const Tensor & save_mean, const Tensor & save_std) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",3, true);
    auto running_mean_ = checked_cast_tensor<CUDADoubleTensor>(running_mean.pImpl,"running_mean",4, true);
    auto running_var_ = checked_cast_tensor<CUDADoubleTensor>(running_var.pImpl,"running_var",5, true);
    auto save_mean_ = checked_cast_tensor<CUDADoubleTensor>(save_mean.pImpl,"save_mean",8, true);
    auto save_std_ = checked_cast_tensor<CUDADoubleTensor>(save_std.pImpl,"save_std",9, true);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",9, true);
    auto grad_weight_ = checked_cast_tensor<CUDADoubleTensor>(grad_weight.pImpl,"grad_weight",9, true);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = checked_cast_tensor<CUDADoubleTensor>(grad_bias.pImpl,"grad_bias",9, true);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    THNN_CudaDoubleBatchNormalization_backward(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, weight_ ? weight_->tensor : NULL, running_mean_ ? running_mean_->tensor : NULL, running_var_ ? running_var_->tensor : NULL, save_mean_ ? save_mean_->tensor : NULL, save_std_ ? save_std_->tensor : NULL, training, 1, eps);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &, Tensor &>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> CUDADoubleType::thnn_batch_norm_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, bool training, double eps, const Tensor & save_mean, const Tensor & save_std, std::array<bool,3> output_mask) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",3, true);
    auto running_mean_ = checked_cast_tensor<CUDADoubleTensor>(running_mean.pImpl,"running_mean",4, true);
    auto running_var_ = checked_cast_tensor<CUDADoubleTensor>(running_var.pImpl,"running_var",5, true);
    auto save_mean_ = checked_cast_tensor<CUDADoubleTensor>(save_mean.pImpl,"save_mean",8, true);
    auto save_std_ = checked_cast_tensor<CUDADoubleTensor>(save_std.pImpl,"save_std",9, true);
    auto grad_input_ = output_mask[0] ? new CUDADoubleTensor(context) : nullptr;
    auto grad_input = Tensor(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_input_, false);
    auto grad_weight_ = output_mask[1] ? new CUDADoubleTensor(context) : nullptr;
    auto grad_weight = Tensor(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_weight_, false);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = output_mask[2] ? new CUDADoubleTensor(context) : nullptr;
    auto grad_bias = Tensor(grad_bias_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_bias_, false);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    THNN_CudaDoubleBatchNormalization_backward(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, weight_ ? weight_->tensor : NULL, running_mean_ ? running_mean_->tensor : NULL, running_var_ ? running_var_->tensor : NULL, save_mean_ ? save_mean_->tensor : NULL, save_std_ ? save_std_->tensor : NULL, training, 1, eps);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDADoubleType::thnn_conv_transpose2d_forward_out(Tensor & output, Tensor & columns, Tensor & ones, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CUDADoubleTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto output_padding_ = check_intlist<2>(output_padding, "output_padding", 7);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 8);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",8, false);
    auto columns_ = checked_cast_tensor<CUDADoubleTensor>(columns.pImpl,"columns",8, false);
    auto ones_ = checked_cast_tensor<CUDADoubleTensor>(ones.pImpl,"ones",8, false);
    THNN_CudaDoubleSpatialFullDilatedConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], output_padding_[1], output_padding_[0]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    columns_->maybeScalar(maybe_scalar);
    ones_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(output, columns, ones);
}
std::tuple<Tensor,Tensor,Tensor> CUDADoubleType::thnn_conv_transpose2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CUDADoubleTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto output_padding_ = check_intlist<2>(output_padding, "output_padding", 7);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 8);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    auto columns_ = new CUDADoubleTensor(context);
    auto columns = Tensor(columns_, false);
    auto ones_ = new CUDADoubleTensor(context);
    auto ones = Tensor(ones_, false);
    THNN_CudaDoubleSpatialFullDilatedConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], output_padding_[1], output_padding_[0]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    columns_->maybeScalar(maybe_scalar);
    ones_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(output, columns, ones);
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDADoubleType::thnn_conv_transpose2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & columns, const Tensor & ones) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto output_padding_ = check_intlist<2>(output_padding, "output_padding", 7);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 8);
    auto columns_ = checked_cast_tensor<CUDADoubleTensor>(columns.pImpl,"columns",9, false);
    auto ones_ = checked_cast_tensor<CUDADoubleTensor>(ones.pImpl,"ones",10, false);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",10, true);
    auto grad_weight_ = checked_cast_tensor<CUDADoubleTensor>(grad_weight.pImpl,"grad_weight",10, true);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = checked_cast_tensor<CUDADoubleTensor>(grad_bias.pImpl,"grad_bias",10, true);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(1) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_CudaDoubleSpatialFullDilatedConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, columns_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], output_padding_[1], output_padding_[0]);
    if (grad_weight_ || grad_bias_) THNN_CudaDoubleSpatialFullDilatedConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], output_padding_[1], output_padding_[0], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &, Tensor &>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> CUDADoubleType::thnn_conv_transpose2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto output_padding_ = check_intlist<2>(output_padding, "output_padding", 7);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 8);
    auto columns_ = checked_cast_tensor<CUDADoubleTensor>(columns.pImpl,"columns",9, false);
    auto ones_ = checked_cast_tensor<CUDADoubleTensor>(ones.pImpl,"ones",10, false);
    auto grad_input_ = output_mask[0] ? new CUDADoubleTensor(context) : nullptr;
    auto grad_input = Tensor(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_input_, false);
    auto grad_weight_ = output_mask[1] ? new CUDADoubleTensor(context) : nullptr;
    auto grad_weight = Tensor(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_weight_, false);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = output_mask[2] ? new CUDADoubleTensor(context) : nullptr;
    auto grad_bias = Tensor(grad_bias_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_bias_, false);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(1) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_CudaDoubleSpatialFullDilatedConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, columns_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], output_padding_[1], output_padding_[0]);
    if (grad_weight_ || grad_bias_) THNN_CudaDoubleSpatialFullDilatedConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], output_padding_[1], output_padding_[0], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDADoubleType::thnn_conv_transpose3d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CUDADoubleTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto output_padding_ = check_intlist<3>(output_padding, "output_padding", 7);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 8);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",8, false);
    auto finput_ = checked_cast_tensor<CUDADoubleTensor>(finput.pImpl,"finput",8, false);
    auto fgrad_input_ = checked_cast_tensor<CUDADoubleTensor>(fgrad_input.pImpl,"fgrad_input",8, false);
    THNN_CudaDoubleVolumetricFullDilatedConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], output_padding_[0], output_padding_[2], output_padding_[1]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    finput_->maybeScalar(maybe_scalar);
    fgrad_input_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(output, finput, fgrad_input);
}
std::tuple<Tensor,Tensor,Tensor> CUDADoubleType::thnn_conv_transpose3d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CUDADoubleTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto output_padding_ = check_intlist<3>(output_padding, "output_padding", 7);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 8);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    auto finput_ = new CUDADoubleTensor(context);
    auto finput = Tensor(finput_, false);
    auto fgrad_input_ = new CUDADoubleTensor(context);
    auto fgrad_input = Tensor(fgrad_input_, false);
    THNN_CudaDoubleVolumetricFullDilatedConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], output_padding_[0], output_padding_[2], output_padding_[1]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    finput_->maybeScalar(maybe_scalar);
    fgrad_input_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(output, finput, fgrad_input);
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDADoubleType::thnn_conv_transpose3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & finput, const Tensor & fgrad_input) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto output_padding_ = check_intlist<3>(output_padding, "output_padding", 7);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 8);
    auto finput_ = checked_cast_tensor<CUDADoubleTensor>(finput.pImpl,"finput",9, false);
    auto fgrad_input_ = checked_cast_tensor<CUDADoubleTensor>(fgrad_input.pImpl,"fgrad_input",10, false);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",10, true);
    auto grad_weight_ = checked_cast_tensor<CUDADoubleTensor>(grad_weight.pImpl,"grad_weight",10, true);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = checked_cast_tensor<CUDADoubleTensor>(grad_bias.pImpl,"grad_bias",10, true);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(1) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_CudaDoubleVolumetricFullDilatedConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], output_padding_[0], output_padding_[2], output_padding_[1]);
    if (grad_weight_ || grad_bias_) THNN_CudaDoubleVolumetricFullDilatedConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], output_padding_[0], output_padding_[2], output_padding_[1], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &, Tensor &>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> CUDADoubleType::thnn_conv_transpose3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto output_padding_ = check_intlist<3>(output_padding, "output_padding", 7);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 8);
    auto finput_ = checked_cast_tensor<CUDADoubleTensor>(finput.pImpl,"finput",9, false);
    auto fgrad_input_ = checked_cast_tensor<CUDADoubleTensor>(fgrad_input.pImpl,"fgrad_input",10, false);
    auto grad_input_ = output_mask[0] ? new CUDADoubleTensor(context) : nullptr;
    auto grad_input = Tensor(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_input_, false);
    auto grad_weight_ = output_mask[1] ? new CUDADoubleTensor(context) : nullptr;
    auto grad_weight = Tensor(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_weight_, false);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = output_mask[2] ? new CUDADoubleTensor(context) : nullptr;
    auto grad_bias = Tensor(grad_bias_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_bias_, false);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(1) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_CudaDoubleVolumetricFullDilatedConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], output_padding_[0], output_padding_[2], output_padding_[1]);
    if (grad_weight_ || grad_bias_) THNN_CudaDoubleVolumetricFullDilatedConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], output_padding_[0], output_padding_[2], output_padding_[1], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDADoubleType::thnn_conv2d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CUDADoubleTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",6, false);
    auto finput_ = checked_cast_tensor<CUDADoubleTensor>(finput.pImpl,"finput",6, false);
    auto fgrad_input_ = checked_cast_tensor<CUDADoubleTensor>(fgrad_input.pImpl,"fgrad_input",6, false);
    THNN_CudaDoubleSpatialConvolutionMM_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    finput_->maybeScalar(maybe_scalar);
    fgrad_input_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(output, finput, fgrad_input);
}
std::tuple<Tensor,Tensor,Tensor> CUDADoubleType::thnn_conv2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CUDADoubleTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    auto finput_ = new CUDADoubleTensor(context);
    auto finput = Tensor(finput_, false);
    auto fgrad_input_ = new CUDADoubleTensor(context);
    auto fgrad_input = Tensor(fgrad_input_, false);
    THNN_CudaDoubleSpatialConvolutionMM_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    finput_->maybeScalar(maybe_scalar);
    fgrad_input_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(output, finput, fgrad_input);
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDADoubleType::thnn_conv2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto finput_ = checked_cast_tensor<CUDADoubleTensor>(finput.pImpl,"finput",7, false);
    auto fgrad_input_ = checked_cast_tensor<CUDADoubleTensor>(fgrad_input.pImpl,"fgrad_input",8, false);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",8, true);
    auto grad_weight_ = checked_cast_tensor<CUDADoubleTensor>(grad_weight.pImpl,"grad_weight",8, true);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = checked_cast_tensor<CUDADoubleTensor>(grad_bias.pImpl,"grad_bias",8, true);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_CudaDoubleSpatialConvolutionMM_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, finput_->tensor, fgrad_input_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
    if (grad_weight_ || grad_bias_) THNN_CudaDoubleSpatialConvolutionMM_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &, Tensor &>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> CUDADoubleType::thnn_conv2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto finput_ = checked_cast_tensor<CUDADoubleTensor>(finput.pImpl,"finput",7, false);
    auto fgrad_input_ = checked_cast_tensor<CUDADoubleTensor>(fgrad_input.pImpl,"fgrad_input",8, false);
    auto grad_input_ = output_mask[0] ? new CUDADoubleTensor(context) : nullptr;
    auto grad_input = Tensor(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_input_, false);
    auto grad_weight_ = output_mask[1] ? new CUDADoubleTensor(context) : nullptr;
    auto grad_weight = Tensor(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_weight_, false);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = output_mask[2] ? new CUDADoubleTensor(context) : nullptr;
    auto grad_bias = Tensor(grad_bias_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_bias_, false);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_CudaDoubleSpatialConvolutionMM_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, finput_->tensor, fgrad_input_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
    if (grad_weight_ || grad_bias_) THNN_CudaDoubleSpatialConvolutionMM_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}
Tensor & CUDADoubleType::thnn_conv_depthwise2d_forward_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CUDADoubleTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",7, false);
    THNN_CudaDoubleSpatialDepthwiseConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
    output_->maybeScalar(self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar()));
    return output;
}
Tensor CUDADoubleType::thnn_conv_depthwise2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CUDADoubleTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_CudaDoubleSpatialDepthwiseConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
    output_->maybeScalar(self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar()));
    return output;
}
std::tuple<Tensor &,Tensor &> CUDADoubleType::thnn_conv_depthwise2d_backward_out(Tensor & grad_input, Tensor & grad_weight, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",7, true);
    auto grad_weight_ = checked_cast_tensor<CUDADoubleTensor>(grad_weight.pImpl,"grad_weight",7, true);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    if (grad_input_) THNN_CudaDoubleSpatialDepthwiseConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
    if (grad_weight_) THNN_CudaDoubleSpatialDepthwiseConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &>(grad_input, grad_weight);
}
std::tuple<Tensor,Tensor> CUDADoubleType::thnn_conv_depthwise2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, std::array<bool,2> output_mask) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
    auto grad_input_ = output_mask[0] ? new CUDADoubleTensor(context) : nullptr;
    auto grad_input = Tensor(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_input_, false);
    auto grad_weight_ = output_mask[1] ? new CUDADoubleTensor(context) : nullptr;
    auto grad_weight = Tensor(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_weight_, false);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    if (grad_input_) THNN_CudaDoubleSpatialDepthwiseConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
    if (grad_weight_) THNN_CudaDoubleSpatialDepthwiseConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor>(grad_input, grad_weight);
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDADoubleType::thnn_conv_dilated2d_forward_out(Tensor & output, Tensor & columns, Tensor & ones, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CUDADoubleTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",7, false);
    auto columns_ = checked_cast_tensor<CUDADoubleTensor>(columns.pImpl,"columns",7, false);
    auto ones_ = checked_cast_tensor<CUDADoubleTensor>(ones.pImpl,"ones",7, false);
    THNN_CudaDoubleSpatialDilatedConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    columns_->maybeScalar(maybe_scalar);
    ones_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(output, columns, ones);
}
std::tuple<Tensor,Tensor,Tensor> CUDADoubleType::thnn_conv_dilated2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CUDADoubleTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    auto columns_ = new CUDADoubleTensor(context);
    auto columns = Tensor(columns_, false);
    auto ones_ = new CUDADoubleTensor(context);
    auto ones = Tensor(ones_, false);
    THNN_CudaDoubleSpatialDilatedConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    columns_->maybeScalar(maybe_scalar);
    ones_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(output, columns, ones);
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDADoubleType::thnn_conv_dilated2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
    auto columns_ = checked_cast_tensor<CUDADoubleTensor>(columns.pImpl,"columns",8, false);
    auto ones_ = checked_cast_tensor<CUDADoubleTensor>(ones.pImpl,"ones",9, false);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",9, true);
    auto grad_weight_ = checked_cast_tensor<CUDADoubleTensor>(grad_weight.pImpl,"grad_weight",9, true);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = checked_cast_tensor<CUDADoubleTensor>(grad_bias.pImpl,"grad_bias",9, true);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_CudaDoubleSpatialDilatedConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, columns_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
    if (grad_weight_ || grad_bias_) THNN_CudaDoubleSpatialDilatedConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &, Tensor &>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> CUDADoubleType::thnn_conv_dilated2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
    auto columns_ = checked_cast_tensor<CUDADoubleTensor>(columns.pImpl,"columns",8, false);
    auto ones_ = checked_cast_tensor<CUDADoubleTensor>(ones.pImpl,"ones",9, false);
    auto grad_input_ = output_mask[0] ? new CUDADoubleTensor(context) : nullptr;
    auto grad_input = Tensor(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_input_, false);
    auto grad_weight_ = output_mask[1] ? new CUDADoubleTensor(context) : nullptr;
    auto grad_weight = Tensor(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_weight_, false);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = output_mask[2] ? new CUDADoubleTensor(context) : nullptr;
    auto grad_bias = Tensor(grad_bias_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_bias_, false);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_CudaDoubleSpatialDilatedConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, columns_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
    if (grad_weight_ || grad_bias_) THNN_CudaDoubleSpatialDilatedConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDADoubleType::thnn_conv_dilated3d_forward_out(Tensor & output, Tensor & columns, Tensor & ones, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CUDADoubleTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 7);
    auto output_ = checked_cast_tensor<CUDADoubleTensor>(output.pImpl,"output",7, false);
    auto columns_ = checked_cast_tensor<CUDADoubleTensor>(columns.pImpl,"columns",7, false);
    auto ones_ = checked_cast_tensor<CUDADoubleTensor>(ones.pImpl,"ones",7, false);
    THNN_CudaDoubleVolumetricDilatedConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    columns_->maybeScalar(maybe_scalar);
    ones_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(output, columns, ones);
}
std::tuple<Tensor,Tensor,Tensor> CUDADoubleType::thnn_conv_dilated3d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CUDADoubleTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 7);
    auto output_ = new CUDADoubleTensor(context);
    auto output = Tensor(output_, false);
    auto columns_ = new CUDADoubleTensor(context);
    auto columns = Tensor(columns_, false);
    auto ones_ = new CUDADoubleTensor(context);
    auto ones = Tensor(ones_, false);
    THNN_CudaDoubleVolumetricDilatedConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    columns_->maybeScalar(maybe_scalar);
    ones_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(output, columns, ones);
}
std::tuple<Tensor &,Tensor &,Tensor &> CUDADoubleType::thnn_conv_dilated3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 7);
    auto columns_ = checked_cast_tensor<CUDADoubleTensor>(columns.pImpl,"columns",8, false);
    auto ones_ = checked_cast_tensor<CUDADoubleTensor>(ones.pImpl,"ones",9, false);
    auto grad_input_ = checked_cast_tensor<CUDADoubleTensor>(grad_input.pImpl,"grad_input",9, true);
    auto grad_weight_ = checked_cast_tensor<CUDADoubleTensor>(grad_weight.pImpl,"grad_weight",9, true);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = checked_cast_tensor<CUDADoubleTensor>(grad_bias.pImpl,"grad_bias",9, true);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_CudaDoubleVolumetricDilatedConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, columns_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1]);
    if (grad_weight_ || grad_bias_) THNN_CudaDoubleVolumetricDilatedConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &, Tensor &>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> CUDADoubleType::thnn_conv_dilated3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) const {
    auto grad_output_ = checked_cast_tensor<CUDADoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CUDADoubleTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CUDADoubleTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 7);
    auto columns_ = checked_cast_tensor<CUDADoubleTensor>(columns.pImpl,"columns",8, false);
    auto ones_ = checked_cast_tensor<CUDADoubleTensor>(ones.pImpl,"ones",9, false);
    auto grad_input_ = output_mask[0] ? new CUDADoubleTensor(context) : nullptr;
    auto grad_input = Tensor(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_input_, false);
    auto grad_weight_ = output_mask[1] ? new CUDADoubleTensor(context) : nullptr;
    auto grad_weight = Tensor(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_weight_, false);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = output_mask[2] ? new CUDADoubleTensor(context) : nullptr;
    auto grad_bias = Tensor(grad_bias_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_bias_, false);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_CudaDoubleVolumetricDilatedConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, columns_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1]);
    if (grad_weight_ || grad_bias_) THNN_CudaDoubleVolumetricDilatedConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}
Tensor & CUDADoubleType::abs_out(Tensor & result, const Tensor & self) const {
    return  at::native::_abs_out_cuda(result, self);
}
Tensor & CUDADoubleType::ceil_out(Tensor & result, const Tensor & self) const {
    return  at::native::_ceil_out_cuda(result, self);
}
Tensor & CUDADoubleType::cos_out(Tensor & result, const Tensor & self) const {
    return  at::native::_cos_out_cuda(result, self);
}
Tensor CUDADoubleType::embedding_dense_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) const {
    return  at::native::embedding_backward_cuda(grad, indices, num_weights, padding_idx, scale_grad_by_freq);
}
Tensor & CUDADoubleType::embedding_renorm_(Tensor & self, const Tensor & indices, double max_norm, double norm_type) const {
    return  at::native::embedding_renorm_cuda_(self, indices, max_norm, norm_type);
}
std::tuple<Tensor,Tensor,Tensor> CUDADoubleType::embedding_bag(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse) const {
    return  at::native::embedding_bag_cuda(weight, indices, offsets, scale_grad_by_freq, mode, sparse);
}
Tensor CUDADoubleType::embedding_bag_dense_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode) const {
    return  at::native::embedding_bag_backward_cuda(grad, indices, offsets, offset2bag, bag_size, num_weights, scale_grad_by_freq, mode);
}
Tensor & CUDADoubleType::exp_out(Tensor & result, const Tensor & self) const {
    return  at::native::_exp_out_cuda(result, self);
}
Tensor & CUDADoubleType::eye_out(Tensor & result, int64_t n, int64_t m) const {
    return  at::native::eye_out_cuda(result, n, m);
}
Tensor & CUDADoubleType::floor_out(Tensor & result, const Tensor & self) const {
    return  at::native::_floor_out_cuda(result, self);
}
Tensor CUDADoubleType::_fft_with_size(const Tensor & self, int64_t signal_ndim, bool complex_input, bool complex_output, bool inverse, IntList checked_signal_sizes, bool normalized, bool onesided, IntList output_sizes) const {
    return  at::native::_fft_cufft(self, signal_ndim, complex_input, complex_output, inverse, checked_signal_sizes, normalized, onesided, output_sizes);
}
Tensor & CUDADoubleType::log_out(Tensor & result, const Tensor & self) const {
    return  at::native::_log_out_cuda(result, self);
}
std::tuple<Tensor,Tensor> CUDADoubleType::RoiPooling2d_forward(const Tensor & input, const Tensor & rois, int64_t pooledHeight, int64_t pooledWidth, double spatialScale) const {
    return  at::native::RoiPooling2d_forward_cuda(input, rois, pooledHeight, pooledWidth, spatialScale);
}
Tensor CUDADoubleType::RoiPooling2d_backward(const Tensor & input, const Tensor & rois, int64_t pooledHeight, int64_t pooledWidth, double spatialScale, const Tensor & gradOutput, const Tensor & argmaxes) const {
    return  at::native::RoiPooling2d_backward_cuda(input, rois, pooledHeight, pooledWidth, spatialScale, gradOutput, argmaxes);
}
Tensor & CUDADoubleType::round_out(Tensor & result, const Tensor & self) const {
    return  at::native::_round_out_cuda(result, self);
}
Tensor & CUDADoubleType::sin_out(Tensor & result, const Tensor & self) const {
    return  at::native::_sin_out_cuda(result, self);
}
Tensor & CUDADoubleType::sspaddmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return  at::native::_sspaddmm_out_only_sparse(result, self, mat1, mat2, beta, alpha);
}
Tensor CUDADoubleType::sum(const Tensor & self) const {
    return  at::native::_sum_cuda(self);
}
Tensor & CUDADoubleType::sum_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    return  at::native::_sum_out_cuda(result, self, dim, keepdim);
}
Tensor & CUDADoubleType::sqrt_out(Tensor & result, const Tensor & self) const {
    return  at::native::_sqrt_out_cuda(result, self);
}
Tensor CUDADoubleType::prod(const Tensor & self) const {
    return  at::native::_prod_cuda(self);
}
Tensor & CUDADoubleType::prod_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    return  at::native::_prod_out_cuda(result, self, dim, keepdim);
}
Tensor & CUDADoubleType::trunc_out(Tensor & result, const Tensor & self) const {
    return  at::native::_trunc_out_cuda(result, self);
}
std::tuple<Tensor,Tensor> CUDADoubleType::_unique(const Tensor & self, bool sorted, bool return_inverse) const {
    return  at::native::_unique_cuda(self, sorted, return_inverse);
}
Tensor CUDADoubleType::_s_where(const Tensor & condition, const Tensor & self, const Tensor & other) const {
    return  at::native::_s_where_cuda(condition, self, other);
}
Tensor CUDADoubleType::_standard_gamma_grad(const Tensor & self, const Tensor & output) const {
    return  at::native::_standard_gamma_grad_cuda(self, output);
}
Tensor CUDADoubleType::poisson(const Tensor & self, Generator * generator) const {
    return  at::native::_s_poisson_cuda(self, generator);
}

}
