// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/CUDAIntType.h"
#include "ATen/CUDAIntStorage.h"
#include "ATen/CUDAIntTensor.h"
#include "ATen/CUDAGenerator.h"
#include "ATen/CUDAByteTensor.h"
#include "ATen/CUDAIntTensor.h"
#include "ATen/CUDALongTensor.h"
#include "ATen/SparseCUDAIntTensor.h"
#include "ATen/CUDAIntTensor.h"
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

CUDAIntType::CUDAIntType(Context* context)
: Type(context, /*is_variable_or_undefined=*/false) {}
ScalarType CUDAIntType::scalarType() const {
  return ScalarType::Int;
}
Backend CUDAIntType::backend() const {
  return Backend::CUDA;
}
bool CUDAIntType::is_cuda() const { return backend() == kCUDA || backend() == kSparseCUDA; }
bool CUDAIntType::is_sparse() const { return backend() == kSparseCPU || backend() == kSparseCUDA; }
bool CUDAIntType::is_distributed() const { return false; }

std::unique_ptr<Storage> CUDAIntType::storage() const {
  return std::unique_ptr<Storage>(new CUDAIntStorage(context));
}
std::unique_ptr<Storage> CUDAIntType::storage(size_t size) const {
  return std::unique_ptr<Storage>(new CUDAIntStorage(context,size));
}
std::unique_ptr<Storage> CUDAIntType::storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const {
    return std::unique_ptr<Storage>(
      new CUDAIntStorage(context,data,size,deleter));
}
std::unique_ptr<Storage> CUDAIntType::storageWithAllocator(int64_t size, std::unique_ptr<Allocator> allocator) const {
    return std::unique_ptr<Storage>(
        new CUDAIntStorage(context, size, std::move(allocator)));
}
Tensor CUDAIntType::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THCudaIntTensor_retain(context->thc_state,  (THCudaIntTensor*) th_pointer);
  return Tensor(new CUDAIntTensor(context,(THCudaIntTensor*)(th_pointer)), false);
}
std::unique_ptr<Storage> CUDAIntType::unsafeStorageFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THCudaIntStorage_retain(context->thc_state,  (THCudaIntStorage*) th_pointer);
  return std::unique_ptr<Storage>(new CUDAIntStorage(context, (THCudaIntStorage*) th_pointer));
}
std::unique_ptr<Generator> CUDAIntType::generator() const {
  return std::unique_ptr<Generator>(new CUDAGenerator(context));
}

const char * CUDAIntType::toString() const {
  return CUDAIntType::typeString();
}
TypeID CUDAIntType::ID() const {
  return TypeID::CUDAInt;
}

std::size_t CUDAIntType::elementSizeInBytes() const {
  return sizeof(int);
}

const char * CUDAIntType::typeString() {
  return "CUDAIntType";
}

/* example
Tensor * CUDAIntType::add(Tensor & a, Tensor & b) {
  std::cout << "add CUDAIntTensor\n";
  return &a;
}
*/

int64_t CUDAIntType::storage_offset(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    return static_cast<int64_t>(THCudaIntTensor_storageOffset(context->thc_state, self_->tensor));
}
Tensor & CUDAIntType::resize_(Tensor & self, IntList size) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    THCudaIntTensor_resize(context->thc_state, self_->tensor, size_, NULL);
    self_->maybeScalar(size.size() == 0);
    return self;
}
int64_t CUDAIntType::numel(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    return static_cast<int64_t>(THCudaIntTensor_nElement(context->thc_state, self_->tensor));
}
Tensor & CUDAIntType::set_(Tensor & self, Storage & storage) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto storage_ = checked_cast_storage<CUDAIntStorage>(&storage,"storage",2);
    THCudaIntTensor_setStorage(context->thc_state, self_->tensor, storage_->storage, 0, THLongStorageView::makeFromLength(static_cast<int64_t>(storage.size())), NULL);
    self_->maybeScalar(false);
    return self;
}
Tensor & CUDAIntType::set_(Tensor & self, Storage & sourceStorage, int64_t storage_offset, IntList size, IntList stride) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto sourceStorage_ = checked_cast_storage<CUDAIntStorage>(&sourceStorage,"sourceStorage",2);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    THCudaIntTensor_setStorage(context->thc_state, self_->tensor, sourceStorage_->storage, storage_offset, size_, stride_);
    self_->maybeScalar(size.size() == 0);
    return self;
}
Tensor & CUDAIntType::set_(Tensor & self, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto source_ = checked_cast_tensor<CUDAIntTensor>(source.pImpl,"source",2, false);
    THCudaIntTensor_set(context->thc_state, self_->tensor, source_->tensor);
    self_->maybeScalar(source_->isScalar());
    return self;
}
Tensor & CUDAIntType::set_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    THCudaIntTensor_setStorage(context->thc_state, self_->tensor, NULL, 0, NULL, NULL);
    self_->maybeScalar(false);
    return self;
}
Tensor & CUDAIntType::fill_(Tensor & self, Scalar value) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toInt();
    THCudaIntTensor_fill(context->thc_state, self_->tensor, value_);
    return self;
}
Tensor & CUDAIntType::fill_(Tensor & self, const Tensor & value) const {
    if (value.dim() == 0) {
        return static_cast<const Type*>(this)->fill_(self, Scalar(value));
    }
    AT_ERROR("fill_ only supports a 0-dimensional value tensor, but got tensor "
        "with %" PRId64 " dimension(s)", value.dim());
}
bool CUDAIntType::is_contiguous(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    return THCudaIntTensor_isContiguous(context->thc_state, self_->tensor);
}
bool CUDAIntType::is_set_to(const Tensor & self, const Tensor & tensor) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto tensor_ = checked_cast_tensor<CUDAIntTensor>(tensor.pImpl,"tensor",2, false);
    return THCudaIntTensor_isSetTo(context->thc_state, self_->tensor, tensor_->tensor);
}
Tensor & CUDAIntType::s_masked_fill_(Tensor & self, const Tensor & mask, Scalar value) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CUDAByteTensor>(mask.pImpl,"mask",2, false);
    auto value_ = value.toInt();
    THCudaIntTensor_maskedFill(context->thc_state, self_->tensor, mask_->tensor, value_);
    return self;
}
Tensor & CUDAIntType::s_masked_fill_(Tensor & self, const Tensor & mask, const Tensor & value) const {
    if (value.dim() == 0) {
        return static_cast<const Type*>(this)->masked_fill_(self, mask, Scalar(value));
    }
    AT_ERROR("masked_fill_ only supports a 0-dimensional value tensor, but got tensor "
        "with %" PRId64 " dimension(s)", value.dim());
}
Tensor & CUDAIntType::s_masked_scatter_(Tensor & self, const Tensor & mask, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CUDAByteTensor>(mask.pImpl,"mask",2, false);
    auto source_ = checked_cast_tensor<CUDAIntTensor>(source.pImpl,"source",3, false);
    THCudaIntTensor_maskedCopy(context->thc_state, self_->tensor, mask_->tensor, source_->tensor);
    return self;
}
Tensor & CUDAIntType::s_masked_select_out(Tensor & result, const Tensor & self, const Tensor & mask) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CUDAByteTensor>(mask.pImpl,"mask",2, false);
    THCudaIntTensor_maskedSelect(context->thc_state, result_->tensor, self_->tensor, mask_->tensor);
    result_->maybeScalar(self_->isScalar() && mask_->isScalar());
    return result;
}
Tensor CUDAIntType::s_masked_select(const Tensor & self, const Tensor & mask) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CUDAByteTensor>(mask.pImpl,"mask",2, false);
    THCudaIntTensor_maskedSelect(context->thc_state, result_->tensor, self_->tensor, mask_->tensor);
    result_->maybeScalar(self_->isScalar() && mask_->isScalar());
    return result;
}
Tensor CUDAIntType::transpose(const Tensor & self, int64_t dim0, int64_t dim1) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    dim0 = maybe_wrap_dim(dim0, self_);
    dim1 = maybe_wrap_dim(dim1, self_);
    return Tensor((new CUDAIntTensor(context, THCudaIntTensor_newTranspose(context->thc_state, self_->tensor, dim0, dim1)))->maybeScalar(self_->isScalar()),false);
}
Tensor CUDAIntType::t(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    return Tensor((new CUDAIntTensor(context, THCudaIntTensor_newTranspose(context->thc_state, self_->tensor, 0, 1)))->maybeScalar(self_->isScalar()),false);
}
Tensor & CUDAIntType::nonzero_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDALongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    THCudaIntTensor_nonzero(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAIntType::nonzero(const Tensor & self) const {
    auto result_ = new CUDALongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    THCudaIntTensor_nonzero(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAIntType::contiguous(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    return Tensor((new CUDAIntTensor(context, THCudaIntTensor_newContiguous(context->thc_state, self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor CUDAIntType::clone(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    return Tensor((new CUDAIntTensor(context, THCudaIntTensor_newClone(context->thc_state, self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor CUDAIntType::view(const Tensor & self, IntList size) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new CUDAIntTensor(context, THCudaIntTensor_newView(context->thc_state, self_->tensor, size_)))->maybeScalar(size.size() == 0),false);
}
Tensor & CUDAIntType::resize_as_(Tensor & self, const Tensor & the_template) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto the_template_ = checked_cast_tensor<CUDAIntTensor>(the_template.pImpl,"the_template",2, false);
    THCudaIntTensor_resizeAs(context->thc_state, self_->tensor, the_template_->tensor);
    self_->maybeScalar(the_template_->isScalar());
    return self;
}
Tensor & CUDAIntType::index_select_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    THCudaIntTensor_indexSelect(context->thc_state, result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
Tensor CUDAIntType::index_select(const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    THCudaIntTensor_indexSelect(context->thc_state, result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
Tensor & CUDAIntType::_indexCopy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    auto source_ = checked_cast_tensor<CUDAIntTensor>(source.pImpl,"source",4, false);
    THCudaIntTensor_indexCopy(context->thc_state, self_->tensor, dim, index_->tensor, source_->tensor);
    return self;
}
Tensor & CUDAIntType::take_out(Tensor & result, const Tensor & self, const Tensor & index) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",2, false);
    THCudaIntTensor_take(context->thc_state, result_->tensor, self_->tensor, index_->tensor);
    result_->maybeScalar(index_->isScalar());
    return result;
}
Tensor CUDAIntType::take(const Tensor & self, const Tensor & index) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",2, false);
    THCudaIntTensor_take(context->thc_state, result_->tensor, self_->tensor, index_->tensor);
    result_->maybeScalar(index_->isScalar());
    return result;
}
Tensor & CUDAIntType::put_(Tensor & self, const Tensor & index, const Tensor & source, bool accumulate) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",2, false);
    auto source_ = checked_cast_tensor<CUDAIntTensor>(source.pImpl,"source",3, false);
    THCudaIntTensor_put(context->thc_state, self_->tensor, index_->tensor, source_->tensor, accumulate);
    return self;
}
Tensor & CUDAIntType::index_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    auto source_ = checked_cast_tensor<CUDAIntTensor>(source.pImpl,"source",4, false);
    THCudaIntTensor_indexAdd(context->thc_state, self_->tensor, dim, index_->tensor, source_->tensor);
    return self;
}
Tensor & CUDAIntType::index_fill_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    auto value_ = value.toInt();
    THCudaIntTensor_indexFill(context->thc_state, self_->tensor, dim, index_->tensor, value_);
    return self;
}
Tensor & CUDAIntType::index_fill_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & value) const {
    if (value.dim() == 0) {
        return static_cast<const Type*>(this)->index_fill_(self, dim, index, Scalar(value));
    }
    AT_ERROR("index_fill_ only supports a 0-dimensional value tensor, but got tensor "
        "with %" PRId64 " dimension(s)", value.dim());
}
Tensor CUDAIntType::unfold(const Tensor & self, int64_t dimension, int64_t size, int64_t step) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    dimension = maybe_wrap_dim(dimension, self_);
    THCudaIntTensor_unfold(context->thc_state, result_->tensor, self_->tensor, dimension, size, step);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAIntType::_range_out(Tensor & result, Scalar start, Scalar end, Scalar step) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto start_ = start.toLong();
    auto end_ = end.toLong();
    auto step_ = step.toLong();
    THCudaIntTensor_range(context->thc_state, result_->tensor, start_, end_, step_);
    return result;
}
Tensor CUDAIntType::_range(Scalar start, Scalar end, Scalar step) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto start_ = start.toLong();
    auto end_ = end.toLong();
    auto step_ = step.toLong();
    THCudaIntTensor_range(context->thc_state, result_->tensor, start_, end_, step_);
    return result;
}
Tensor & CUDAIntType::_arange_out(Tensor & result, Scalar start, Scalar end, Scalar step) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto start_ = start.toLong();
    auto end_ = end.toLong();
    auto step_ = step.toLong();
    THCudaIntTensor_arange(context->thc_state, result_->tensor, start_, end_, step_);
    return result;
}
Tensor CUDAIntType::_arange(Scalar start, Scalar end, Scalar step) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto start_ = start.toLong();
    auto end_ = end.toLong();
    auto step_ = step.toLong();
    THCudaIntTensor_arange(context->thc_state, result_->tensor, start_, end_, step_);
    return result;
}
Tensor & CUDAIntType::_arange_out(Tensor & result, Scalar end) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto end_ = end.toLong();
    THCudaIntTensor_arange(context->thc_state, result_->tensor, 0, end_, 1);
    return result;
}
Tensor CUDAIntType::_arange(Scalar end) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto end_ = end.toLong();
    THCudaIntTensor_arange(context->thc_state, result_->tensor, 0, end_, 1);
    return result;
}
Tensor & CUDAIntType::scatter_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
    if (src.dim() == 0) {
        return static_cast<const Type*>(this)->scatter_(self, dim, index, Scalar(src));
    }
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    auto src_ = checked_cast_tensor<CUDAIntTensor>(src.pImpl,"src",4, false);
    THCudaIntTensor_scatter(context->thc_state, self_->tensor, dim, index_->tensor, src_->tensor);
    return self;
}
Tensor & CUDAIntType::scatter_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    auto value_ = value.toInt();
    THCudaIntTensor_scatterFill(context->thc_state, self_->tensor, dim, index_->tensor, value_);
    return self;
}
Tensor & CUDAIntType::scatter_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    auto src_ = checked_cast_tensor<CUDAIntTensor>(src.pImpl,"src",4, false);
    THCudaIntTensor_scatterAdd(context->thc_state, self_->tensor, dim, index_->tensor, src_->tensor);
    return self;
}
Tensor & CUDAIntType::gather_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    result.resize_(index.sizes());
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    THCudaIntTensor_gather(context->thc_state, result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
Tensor CUDAIntType::gather(const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    result.resize_(index.sizes());
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CUDALongTensor>(index.pImpl,"index",3, false);
    THCudaIntTensor_gather(context->thc_state, result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
void* CUDAIntType::data_ptr(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    return THCudaIntTensor_data(context->thc_state, self_->tensor);
}
bool CUDAIntType::equal(const Tensor & self, const Tensor & other) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",2, false);
    return THCudaIntTensor_equal(context->thc_state, self_->tensor, other_->tensor);
}
Tensor & CUDAIntType::__and___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_bitand(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAIntType::__and__(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_bitand(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAIntType::s___and___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__and___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",2, false);
    THCudaIntTensor_cbitand(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAIntType::s___and__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__and__(self, Scalar(other));
    }
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",2, false);
    THCudaIntTensor_cbitand(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAIntType::__iand__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_bitand(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAIntType::s___iand__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__iand__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",3, false);
    THCudaIntTensor_cbitand(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAIntType::__or___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_bitor(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAIntType::__or__(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_bitor(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAIntType::s___or___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__or___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",2, false);
    THCudaIntTensor_cbitor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAIntType::s___or__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__or__(self, Scalar(other));
    }
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",2, false);
    THCudaIntTensor_cbitor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAIntType::__ior__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_bitor(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAIntType::s___ior__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__ior__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",3, false);
    THCudaIntTensor_cbitor(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAIntType::__xor___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_bitxor(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAIntType::__xor__(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_bitxor(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAIntType::s___xor___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__xor___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",2, false);
    THCudaIntTensor_cbitxor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAIntType::s___xor__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__xor__(self, Scalar(other));
    }
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",2, false);
    THCudaIntTensor_cbitxor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAIntType::__ixor__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_bitxor(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAIntType::s___ixor__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__ixor__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",3, false);
    THCudaIntTensor_cbitxor(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAIntType::__lshift___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_lshift(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAIntType::__lshift__(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_lshift(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAIntType::s___lshift___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__lshift___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",2, false);
    THCudaIntTensor_clshift(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAIntType::s___lshift__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__lshift__(self, Scalar(other));
    }
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",2, false);
    THCudaIntTensor_clshift(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAIntType::__ilshift__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_lshift(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAIntType::s___ilshift__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__ilshift__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",3, false);
    THCudaIntTensor_clshift(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAIntType::__rshift___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_rshift(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAIntType::__rshift__(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_rshift(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAIntType::s___rshift___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__rshift___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",2, false);
    THCudaIntTensor_crshift(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAIntType::s___rshift__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__rshift__(self, Scalar(other));
    }
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",2, false);
    THCudaIntTensor_crshift(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAIntType::__irshift__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_rshift(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAIntType::s___irshift__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__irshift__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",3, false);
    THCudaIntTensor_crshift(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAIntType::lt_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_ltValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAIntType::lt(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_ltValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAIntType::s_lt_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->lt_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",2, false);
    THCudaIntTensor_ltTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAIntType::s_lt(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->lt(self, Scalar(other));
    }
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",2, false);
    THCudaIntTensor_ltTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAIntType::lt_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_ltValueT(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAIntType::s_lt_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->lt_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",3, false);
    THCudaIntTensor_ltTensorT(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAIntType::gt_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_gtValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAIntType::gt(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_gtValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAIntType::s_gt_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->gt_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",2, false);
    THCudaIntTensor_gtTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAIntType::s_gt(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->gt(self, Scalar(other));
    }
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",2, false);
    THCudaIntTensor_gtTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAIntType::gt_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_gtValueT(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAIntType::s_gt_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->gt_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",3, false);
    THCudaIntTensor_gtTensorT(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAIntType::le_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_leValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAIntType::le(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_leValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAIntType::s_le_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->le_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",2, false);
    THCudaIntTensor_leTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAIntType::s_le(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->le(self, Scalar(other));
    }
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",2, false);
    THCudaIntTensor_leTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAIntType::le_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_leValueT(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAIntType::s_le_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->le_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",3, false);
    THCudaIntTensor_leTensorT(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAIntType::ge_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_geValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAIntType::ge(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_geValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAIntType::s_ge_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ge_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",2, false);
    THCudaIntTensor_geTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAIntType::s_ge(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ge(self, Scalar(other));
    }
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",2, false);
    THCudaIntTensor_geTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAIntType::ge_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_geValueT(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAIntType::s_ge_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ge_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",3, false);
    THCudaIntTensor_geTensorT(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAIntType::eq_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_eqValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAIntType::eq(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_eqValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAIntType::s_eq_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->eq_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",2, false);
    THCudaIntTensor_eqTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAIntType::s_eq(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->eq(self, Scalar(other));
    }
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",2, false);
    THCudaIntTensor_eqTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAIntType::eq_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_eqValueT(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAIntType::s_eq_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->eq_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",3, false);
    THCudaIntTensor_eqTensorT(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAIntType::ne_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_neValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAIntType::ne(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_neValue(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAIntType::s_ne_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ne_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",2, false);
    THCudaIntTensor_neTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAIntType::s_ne(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ne(self, Scalar(other));
    }
    auto result_ = new CUDAByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",2, false);
    THCudaIntTensor_neTensor(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAIntType::ne_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_neValueT(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAIntType::s_ne_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ne_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",3, false);
    THCudaIntTensor_neTensorT(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
std::tuple<Tensor &,Tensor &> CUDAIntType::min_out(Tensor & min, Tensor & min_indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto min_ = checked_cast_tensor<CUDAIntTensor>(min.pImpl,"min",0, false);
    auto min_indices_ = checked_cast_tensor<CUDALongTensor>(min_indices.pImpl,"min_indices",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaIntTensor_min(context->thc_state, min_->tensor, min_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    min_->maybeScalar(maybe_scalar);
    min_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(min, min_indices);
}
std::tuple<Tensor,Tensor> CUDAIntType::min(const Tensor & self, int64_t dim, bool keepdim) const {
    auto min_ = new CUDAIntTensor(context);
    auto min = Tensor(min_, false);
    auto min_indices_ = new CUDALongTensor(context);
    auto min_indices = Tensor(min_indices_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaIntTensor_min(context->thc_state, min_->tensor, min_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    min_->maybeScalar(maybe_scalar);
    min_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(min, min_indices);
}
Tensor & CUDAIntType::s_min_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",2, false);
    THCudaIntTensor_cmin(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAIntType::s_min(const Tensor & self, const Tensor & other) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",2, false);
    THCudaIntTensor_cmin(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAIntType::min(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<int>(THCudaIntTensor_minall(context->thc_state, self_->tensor)));
}
std::tuple<Tensor &,Tensor &> CUDAIntType::max_out(Tensor & max, Tensor & max_indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto max_ = checked_cast_tensor<CUDAIntTensor>(max.pImpl,"max",0, false);
    auto max_indices_ = checked_cast_tensor<CUDALongTensor>(max_indices.pImpl,"max_indices",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaIntTensor_max(context->thc_state, max_->tensor, max_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    max_->maybeScalar(maybe_scalar);
    max_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(max, max_indices);
}
std::tuple<Tensor,Tensor> CUDAIntType::max(const Tensor & self, int64_t dim, bool keepdim) const {
    auto max_ = new CUDAIntTensor(context);
    auto max = Tensor(max_, false);
    auto max_indices_ = new CUDALongTensor(context);
    auto max_indices = Tensor(max_indices_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaIntTensor_max(context->thc_state, max_->tensor, max_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    max_->maybeScalar(maybe_scalar);
    max_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(max, max_indices);
}
Tensor & CUDAIntType::s_max_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",2, false);
    THCudaIntTensor_cmax(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAIntType::s_max(const Tensor & self, const Tensor & other) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",2, false);
    THCudaIntTensor_cmax(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAIntType::max(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<int>(THCudaIntTensor_maxall(context->thc_state, self_->tensor)));
}
std::tuple<Tensor &,Tensor &> CUDAIntType::mode_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = checked_cast_tensor<CUDAIntTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaIntTensor_mode(context->thc_state, values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CUDAIntType::mode(const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = new CUDAIntTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CUDALongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaIntTensor_mode(context->thc_state, values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
std::tuple<Tensor &,Tensor &> CUDAIntType::median_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = checked_cast_tensor<CUDAIntTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaIntTensor_median(context->thc_state, values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CUDAIntType::median(const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = new CUDAIntTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CUDALongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaIntTensor_median(context->thc_state, values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
Tensor CUDAIntType::median(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<int>(THCudaIntTensor_medianall(context->thc_state, self_->tensor)));
}
std::tuple<Tensor &,Tensor &> CUDAIntType::sort_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool descending) const {
    auto values_ = checked_cast_tensor<CUDAIntTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaIntTensor_sort(context->thc_state, values_->tensor, indices_->tensor, self_->tensor, dim, descending);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CUDAIntType::sort(const Tensor & self, int64_t dim, bool descending) const {
    auto values_ = new CUDAIntTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CUDALongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaIntTensor_sort(context->thc_state, values_->tensor, indices_->tensor, self_->tensor, dim, descending);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
std::tuple<Tensor &,Tensor &> CUDAIntType::topk_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) const {
    auto values_ = checked_cast_tensor<CUDAIntTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CUDALongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaIntTensor_topk(context->thc_state, values_->tensor, indices_->tensor, self_->tensor, k, dim, largest, sorted);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CUDAIntType::topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) const {
    auto values_ = new CUDAIntTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CUDALongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaIntTensor_topk(context->thc_state, values_->tensor, indices_->tensor, self_->tensor, k, dim, largest, sorted);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
int64_t CUDAIntType::get_device(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    return static_cast<int64_t>(THCudaIntTensor_getDevice(context->thc_state, self_->tensor));
}
Tensor & CUDAIntType::_abs_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    THCudaIntTensor_abs(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAIntType::_abs(const Tensor & self) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    THCudaIntTensor_abs(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAIntType::neg_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    THCudaIntTensor_neg(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAIntType::neg(const Tensor & self) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    THCudaIntTensor_neg(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAIntType::neg_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    THCudaIntTensor_neg(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor & CUDAIntType::pow_out(Tensor & result, const Tensor & self, Scalar exponent) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toInt();
    THCudaIntTensor_pow(context->thc_state, result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAIntType::pow(const Tensor & self, Scalar exponent) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toInt();
    THCudaIntTensor_pow(context->thc_state, result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAIntType::s_pow_out(Tensor & result, const Tensor & self, const Tensor & exponent) const {
    if (exponent.dim() == 0) {
        return static_cast<const Type*>(this)->pow_out(result, self, Scalar(exponent));
    }
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto exponent_ = checked_cast_tensor<CUDAIntTensor>(exponent.pImpl,"exponent",2, false);
    THCudaIntTensor_cpow(context->thc_state, result_->tensor, self_->tensor, exponent_->tensor);
    result_->maybeScalar(self_->isScalar() && exponent_->isScalar());
    return result;
}
Tensor CUDAIntType::s_pow(const Tensor & self, const Tensor & exponent) const {
    if (exponent.dim() == 0) {
        return static_cast<const Type*>(this)->pow(self, Scalar(exponent));
    }
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto exponent_ = checked_cast_tensor<CUDAIntTensor>(exponent.pImpl,"exponent",2, false);
    THCudaIntTensor_cpow(context->thc_state, result_->tensor, self_->tensor, exponent_->tensor);
    result_->maybeScalar(self_->isScalar() && exponent_->isScalar());
    return result;
}
Tensor & CUDAIntType::pow_out(Tensor & result, Scalar base, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto base_ = base.toInt();
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",2, false);
    THCudaIntTensor_tpow(context->thc_state, result_->tensor, base_, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAIntType::pow(Scalar base, const Tensor & self) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto base_ = base.toInt();
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",2, false);
    THCudaIntTensor_tpow(context->thc_state, result_->tensor, base_, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAIntType::pow_(Tensor & self, Scalar exponent) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toInt();
    THCudaIntTensor_pow(context->thc_state, self_->tensor, self_->tensor, exponent_);
    return self;
}
Tensor & CUDAIntType::s_pow_(Tensor & self, const Tensor & exponent) const {
    if (exponent.dim() == 0) {
        return static_cast<const Type*>(this)->pow_(self, Scalar(exponent));
    }
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto exponent_ = checked_cast_tensor<CUDAIntTensor>(exponent.pImpl,"exponent",3, false);
    THCudaIntTensor_cpow(context->thc_state, self_->tensor, self_->tensor, exponent_->tensor);
    return self;
}
Tensor & CUDAIntType::zero_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    THCudaIntTensor_zero(context->thc_state, self_->tensor);
    return self;
}
Tensor CUDAIntType::_sumall(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<int>(THCudaIntTensor_sumall(context->thc_state, self_->tensor)));
}
Tensor & CUDAIntType::_sum_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaIntTensor_sum(context->thc_state, result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CUDAIntType::_sum(const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaIntTensor_sum(context->thc_state, result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CUDAIntType::_prodall(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<int>(THCudaIntTensor_prodall(context->thc_state, self_->tensor)));
}
Tensor & CUDAIntType::_prod_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaIntTensor_prod(context->thc_state, result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CUDAIntType::_prod(const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaIntTensor_prod(context->thc_state, result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor & CUDAIntType::cumsum_out(Tensor & result, const Tensor & self, int64_t dim) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaIntTensor_cumsum(context->thc_state, result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAIntType::cumsum(const Tensor & self, int64_t dim) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaIntTensor_cumsum(context->thc_state, result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAIntType::cumprod_out(Tensor & result, const Tensor & self, int64_t dim) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaIntTensor_cumprod(context->thc_state, result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAIntType::cumprod(const Tensor & self, int64_t dim) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCudaIntTensor_cumprod(context->thc_state, result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAIntType::sign_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    THCudaIntTensor_sign(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAIntType::sign(const Tensor & self) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    THCudaIntTensor_sign(context->thc_state, result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAIntType::sign_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    THCudaIntTensor_sign(context->thc_state, self_->tensor, self_->tensor);
    return self;
}
Tensor CUDAIntType::trace(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<int>(THCudaIntTensor_trace(context->thc_state, self_->tensor)));
}
Tensor & CUDAIntType::add_out(Tensor & result, const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    auto alpha_ = alpha.toInt();
    THCudaIntTensor_add_scaled(context->thc_state, result_->tensor, self_->tensor, other_, alpha_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAIntType::add(const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    auto alpha_ = alpha.toInt();
    THCudaIntTensor_add_scaled(context->thc_state, result_->tensor, self_->tensor, other_, alpha_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAIntType::s_add_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if(other.type().is_sparse()) {
        return static_cast<const Type*>(this)->add_out(result, self, SparseTensor(other), alpha);
    }
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toInt();
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",3, false);
    THCudaIntTensor_cadd(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAIntType::s_add(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if(other.type().is_sparse()) {
        return static_cast<const Type*>(this)->add(self, SparseTensor(other), alpha);
    }
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add(self, Scalar(other), alpha);
    }
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toInt();
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",3, false);
    THCudaIntTensor_cadd(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAIntType::add_out(Tensor & result, const Tensor & self, SparseTensor other, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toInt();
    auto other_ = checked_cast_tensor<SparseCUDAIntTensor>(other.tref.pImpl,"other",3,false);
    THCSIntTensor_spcadd(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAIntType::add(const Tensor & self, SparseTensor other, Scalar alpha) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toInt();
    auto other_ = checked_cast_tensor<SparseCUDAIntTensor>(other.tref.pImpl,"other",3,false);
    THCSIntTensor_spcadd(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAIntType::add_(Tensor & self, Scalar other, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    auto alpha_ = alpha.toInt();
    THCudaIntTensor_add_scaled(context->thc_state, self_->tensor, self_->tensor, other_, alpha_);
    return self;
}
Tensor & CUDAIntType::s_add_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if(other.type().is_sparse()) {
        return static_cast<const Type*>(this)->add_(self, SparseTensor(other), alpha);
    }
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toInt();
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",4, false);
    THCudaIntTensor_cadd(context->thc_state, self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & CUDAIntType::add_(Tensor & self, SparseTensor other, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toInt();
    auto other_ = checked_cast_tensor<SparseCUDAIntTensor>(other.tref.pImpl,"other",4,false);
    THCSIntTensor_spcadd(context->thc_state, self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & CUDAIntType::sub_out(Tensor & result, const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    auto alpha_ = alpha.toInt();
    THCudaIntTensor_sub_scaled(context->thc_state, result_->tensor, self_->tensor, other_, alpha_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAIntType::sub(const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    auto alpha_ = alpha.toInt();
    THCudaIntTensor_sub_scaled(context->thc_state, result_->tensor, self_->tensor, other_, alpha_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAIntType::s_sub_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toInt();
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",3, false);
    THCudaIntTensor_csub(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAIntType::s_sub(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub(self, Scalar(other), alpha);
    }
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toInt();
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",3, false);
    THCudaIntTensor_csub(context->thc_state, result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAIntType::sub_(Tensor & self, Scalar other, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    auto alpha_ = alpha.toInt();
    THCudaIntTensor_sub_scaled(context->thc_state, self_->tensor, self_->tensor, other_, alpha_);
    return self;
}
Tensor & CUDAIntType::s_sub_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toInt();
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",4, false);
    THCudaIntTensor_csub(context->thc_state, self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & CUDAIntType::mul_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_mul(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAIntType::mul(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_mul(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAIntType::s_mul_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",2, false);
    THCudaIntTensor_cmul(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAIntType::s_mul(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul(self, Scalar(other));
    }
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",2, false);
    THCudaIntTensor_cmul(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAIntType::mul_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_mul(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAIntType::s_mul_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",3, false);
    THCudaIntTensor_cmul(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAIntType::div_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_div(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAIntType::div(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_div(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAIntType::s_div_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->div_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",2, false);
    THCudaIntTensor_cdiv(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAIntType::s_div(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->div(self, Scalar(other));
    }
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",2, false);
    THCudaIntTensor_cdiv(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAIntType::div_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_div(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAIntType::s_div_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->div_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",3, false);
    THCudaIntTensor_cdiv(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAIntType::fmod_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_fmod(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAIntType::fmod(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_fmod(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAIntType::s_fmod_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->fmod_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",2, false);
    THCudaIntTensor_cfmod(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAIntType::s_fmod(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->fmod(self, Scalar(other));
    }
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",2, false);
    THCudaIntTensor_cfmod(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAIntType::fmod_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_fmod(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAIntType::s_fmod_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->fmod_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",3, false);
    THCudaIntTensor_cfmod(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAIntType::remainder_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_remainder(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAIntType::remainder(const Tensor & self, Scalar other) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_remainder(context->thc_state, result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAIntType::s_remainder_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->remainder_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",2, false);
    THCudaIntTensor_cremainder(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAIntType::s_remainder(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->remainder(self, Scalar(other));
    }
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",2, false);
    THCudaIntTensor_cremainder(context->thc_state, result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAIntType::remainder_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THCudaIntTensor_remainder(context->thc_state, self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CUDAIntType::s_remainder_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->remainder_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",3, false);
    THCudaIntTensor_cremainder(context->thc_state, self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CUDAIntType::clamp_out(Tensor & result, const Tensor & self, Scalar min, Scalar max) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toInt();
    auto max_ = max.toInt();
    THCudaIntTensor_clamp(context->thc_state, result_->tensor, self_->tensor, min_, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAIntType::clamp(const Tensor & self, Scalar min, Scalar max) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toInt();
    auto max_ = max.toInt();
    THCudaIntTensor_clamp(context->thc_state, result_->tensor, self_->tensor, min_, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAIntType::clamp_(Tensor & self, Scalar min, Scalar max) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toInt();
    auto max_ = max.toInt();
    THCudaIntTensor_clamp(context->thc_state, self_->tensor, self_->tensor, min_, max_);
    return self;
}
Tensor & CUDAIntType::clamp_min_out(Tensor & result, const Tensor & self, Scalar min) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toInt();
    THCudaIntTensor_cmaxValue(context->thc_state, result_->tensor, self_->tensor, min_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAIntType::clamp_min(const Tensor & self, Scalar min) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toInt();
    THCudaIntTensor_cmaxValue(context->thc_state, result_->tensor, self_->tensor, min_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAIntType::clamp_min_(Tensor & self, Scalar min) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toInt();
    THCudaIntTensor_cmaxValue(context->thc_state, self_->tensor, self_->tensor, min_);
    return self;
}
Tensor & CUDAIntType::clamp_max_out(Tensor & result, const Tensor & self, Scalar max) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto max_ = max.toInt();
    THCudaIntTensor_cminValue(context->thc_state, result_->tensor, self_->tensor, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAIntType::clamp_max(const Tensor & self, Scalar max) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto max_ = max.toInt();
    THCudaIntTensor_cminValue(context->thc_state, result_->tensor, self_->tensor, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAIntType::clamp_max_(Tensor & self, Scalar max) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto max_ = max.toInt();
    THCudaIntTensor_cminValue(context->thc_state, self_->tensor, self_->tensor, max_);
    return self;
}
Tensor & CUDAIntType::tril_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    THCudaIntTensor_tril(context->thc_state, result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAIntType::tril(const Tensor & self, int64_t diagonal) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    THCudaIntTensor_tril(context->thc_state, result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAIntType::tril_(Tensor & self, int64_t diagonal) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    THCudaIntTensor_tril(context->thc_state, self_->tensor, self_->tensor, diagonal);
    return self;
}
Tensor & CUDAIntType::triu_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    THCudaIntTensor_triu(context->thc_state, result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAIntType::triu(const Tensor & self, int64_t diagonal) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    THCudaIntTensor_triu(context->thc_state, result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAIntType::triu_(Tensor & self, int64_t diagonal) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    THCudaIntTensor_triu(context->thc_state, self_->tensor, self_->tensor, diagonal);
    return self;
}
Tensor & CUDAIntType::cross_out(Tensor & result, const Tensor & self, const Tensor & other, int64_t dim) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",2, false);
    THCudaIntTensor_cross(context->thc_state, result_->tensor, self_->tensor, other_->tensor, dim);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CUDAIntType::cross(const Tensor & self, const Tensor & other, int64_t dim) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CUDAIntTensor>(other.pImpl,"other",2, false);
    THCudaIntTensor_cross(context->thc_state, result_->tensor, self_->tensor, other_->tensor, dim);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CUDAIntType::diag_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    if (self_->isScalar()) {
      throw std::runtime_error("Input must be 1-d or 2-d");
    }
    THCudaIntTensor_diag(context->thc_state, result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CUDAIntType::diag(const Tensor & self, int64_t diagonal) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    if (self_->isScalar()) {
      throw std::runtime_error("Input must be 1-d or 2-d");
    }
    THCudaIntTensor_diag(context->thc_state, result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAIntType::s_addmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    if(mat1.type().is_sparse()) {
        return static_cast<const Type*>(this)->addmm_out(result, self, SparseTensor(mat1), mat2, beta, alpha);
    }
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toInt();
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toInt();
    auto mat1_ = checked_cast_tensor<CUDAIntTensor>(mat1.pImpl,"mat1",4, false);
    auto mat2_ = checked_cast_tensor<CUDAIntTensor>(mat2.pImpl,"mat2",5, false);
    THCudaIntTensor_addmm(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CUDAIntType::s_addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    if(mat1.type().is_sparse()) {
        return static_cast<const Type*>(this)->addmm(self, SparseTensor(mat1), mat2, beta, alpha);
    }
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toInt();
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toInt();
    auto mat1_ = checked_cast_tensor<CUDAIntTensor>(mat1.pImpl,"mat1",4, false);
    auto mat2_ = checked_cast_tensor<CUDAIntTensor>(mat2.pImpl,"mat2",5, false);
    THCudaIntTensor_addmm(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CUDAIntType::addmm_out(Tensor & result, const Tensor & self, SparseTensor mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toInt();
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toInt();
    auto mat1_ = checked_cast_tensor<SparseCUDAIntTensor>(mat1.tref.pImpl,"mat1",4,false);
    auto mat2_ = checked_cast_tensor<CUDAIntTensor>(mat2.pImpl,"mat2",5, false);
    THCSIntTensor_spaddmm(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CUDAIntType::addmm(const Tensor & self, SparseTensor mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toInt();
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toInt();
    auto mat1_ = checked_cast_tensor<SparseCUDAIntTensor>(mat1.tref.pImpl,"mat1",4,false);
    auto mat2_ = checked_cast_tensor<CUDAIntTensor>(mat2.pImpl,"mat2",5, false);
    THCSIntTensor_spaddmm(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CUDAIntType::addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    if(mat1.type().is_sparse()) {
        return static_cast<const Type*>(this)->addmm_(self, SparseTensor(mat1), mat2, beta, alpha);
    }
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toInt();
    auto alpha_ = alpha.toInt();
    auto mat1_ = checked_cast_tensor<CUDAIntTensor>(mat1.pImpl,"mat1",5, false);
    auto mat2_ = checked_cast_tensor<CUDAIntTensor>(mat2.pImpl,"mat2",6, false);
    THCudaIntTensor_addmm(context->thc_state, self_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    return self;
}
Tensor & CUDAIntType::addmm_(Tensor & self, SparseTensor mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toInt();
    auto alpha_ = alpha.toInt();
    auto mat1_ = checked_cast_tensor<SparseCUDAIntTensor>(mat1.tref.pImpl,"mat1",5,false);
    auto mat2_ = checked_cast_tensor<CUDAIntTensor>(mat2.pImpl,"mat2",6, false);
    THCSIntTensor_spaddmm(context->thc_state, self_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    return self;
}
Tensor & CUDAIntType::s__addmv_out(Tensor & result, const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toInt();
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toInt();
    auto mat_ = checked_cast_tensor<CUDAIntTensor>(mat.pImpl,"mat",4, false);
    auto vec_ = checked_cast_tensor<CUDAIntTensor>(vec.pImpl,"vec",5, false);
    THCudaIntTensor_addmv(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, mat_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && mat_->isScalar() && vec_->isScalar());
    return result;
}
Tensor CUDAIntType::s__addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toInt();
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toInt();
    auto mat_ = checked_cast_tensor<CUDAIntTensor>(mat.pImpl,"mat",4, false);
    auto vec_ = checked_cast_tensor<CUDAIntTensor>(vec.pImpl,"vec",5, false);
    THCudaIntTensor_addmv(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, mat_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && mat_->isScalar() && vec_->isScalar());
    return result;
}
Tensor & CUDAIntType::_addmv_(Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toInt();
    auto alpha_ = alpha.toInt();
    auto mat_ = checked_cast_tensor<CUDAIntTensor>(mat.pImpl,"mat",5, false);
    auto vec_ = checked_cast_tensor<CUDAIntTensor>(vec.pImpl,"vec",6, false);
    THCudaIntTensor_addmv(context->thc_state, self_->tensor, beta_, self_->tensor, alpha_, mat_->tensor, vec_->tensor);
    return self;
}
Tensor & CUDAIntType::s__addr_out(Tensor & result, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toInt();
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toInt();
    auto vec1_ = checked_cast_tensor<CUDAIntTensor>(vec1.pImpl,"vec1",4, false);
    auto vec2_ = checked_cast_tensor<CUDAIntTensor>(vec2.pImpl,"vec2",5, false);
    THCudaIntTensor_addr(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, vec1_->tensor, vec2_->tensor);
    result_->maybeScalar(self_->isScalar() && vec1_->isScalar() && vec2_->isScalar());
    return result;
}
Tensor CUDAIntType::s__addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toInt();
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toInt();
    auto vec1_ = checked_cast_tensor<CUDAIntTensor>(vec1.pImpl,"vec1",4, false);
    auto vec2_ = checked_cast_tensor<CUDAIntTensor>(vec2.pImpl,"vec2",5, false);
    THCudaIntTensor_addr(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, vec1_->tensor, vec2_->tensor);
    result_->maybeScalar(self_->isScalar() && vec1_->isScalar() && vec2_->isScalar());
    return result;
}
Tensor & CUDAIntType::_addr_(Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toInt();
    auto alpha_ = alpha.toInt();
    auto vec1_ = checked_cast_tensor<CUDAIntTensor>(vec1.pImpl,"vec1",5, false);
    auto vec2_ = checked_cast_tensor<CUDAIntTensor>(vec2.pImpl,"vec2",6, false);
    THCudaIntTensor_addr(context->thc_state, self_->tensor, beta_, self_->tensor, alpha_, vec1_->tensor, vec2_->tensor);
    return self;
}
Tensor & CUDAIntType::_ger_out(Tensor & result, const Tensor & self, const Tensor & vec2) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.dim() == 0 ? 1 : self.size(0),vec2.dim() == 0 ? 1 : vec2.size(0) });
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto vec2_ = checked_cast_tensor<CUDAIntTensor>(vec2.pImpl,"vec2",2, false);
    THCudaIntTensor_addr(context->thc_state, result_->tensor, int(0), result_->tensor, int(1), self_->tensor, vec2_->tensor);
    result_->maybeScalar(false);
    return result;
}
Tensor CUDAIntType::_ger(const Tensor & self, const Tensor & vec2) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.dim() == 0 ? 1 : self.size(0),vec2.dim() == 0 ? 1 : vec2.size(0) });
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto vec2_ = checked_cast_tensor<CUDAIntTensor>(vec2.pImpl,"vec2",2, false);
    THCudaIntTensor_addr(context->thc_state, result_->tensor, int(0), result_->tensor, int(1), self_->tensor, vec2_->tensor);
    result_->maybeScalar(false);
    return result;
}
Tensor & CUDAIntType::_mv_out(Tensor & result, const Tensor & self, const Tensor & vec) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.size(0) });
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto vec_ = checked_cast_tensor<CUDAIntTensor>(vec.pImpl,"vec",2, false);
    THCudaIntTensor_addmv(context->thc_state, result_->tensor, int(0), result_->tensor, int(1), self_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && vec_->isScalar());
    return result;
}
Tensor CUDAIntType::_mv(const Tensor & self, const Tensor & vec) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.size(0) });
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto vec_ = checked_cast_tensor<CUDAIntTensor>(vec.pImpl,"vec",2, false);
    THCudaIntTensor_addmv(context->thc_state, result_->tensor, int(0), result_->tensor, int(1), self_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && vec_->isScalar());
    return result;
}
Tensor & CUDAIntType::_mm_out(Tensor & result, const Tensor & self, const Tensor & mat2) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.size(0),mat2.size(1) });
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CUDAIntTensor>(mat2.pImpl,"mat2",2, false);
    THCudaIntTensor_addmm(context->thc_state, result_->tensor, int(0), result_->tensor, int(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CUDAIntType::_mm(const Tensor & self, const Tensor & mat2) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.size(0),mat2.size(1) });
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CUDAIntTensor>(mat2.pImpl,"mat2",2, false);
    THCudaIntTensor_addmm(context->thc_state, result_->tensor, int(0), result_->tensor, int(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CUDAIntType::bmm_out(Tensor & result, const Tensor & self, const Tensor & mat2) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.size(0),self.size(1),mat2.size(2) });
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CUDAIntTensor>(mat2.pImpl,"mat2",2, false);
    THCudaIntTensor_baddbmm(context->thc_state, result_->tensor, int(0), result_->tensor, int(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CUDAIntType::bmm(const Tensor & self, const Tensor & mat2) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.size(0),self.size(1),mat2.size(2) });
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CUDAIntTensor>(mat2.pImpl,"mat2",2, false);
    THCudaIntTensor_baddbmm(context->thc_state, result_->tensor, int(0), result_->tensor, int(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CUDAIntType::s_addbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toInt();
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toInt();
    auto batch1_ = checked_cast_tensor<CUDAIntTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CUDAIntTensor>(batch2.pImpl,"batch2",5, false);
    THCudaIntTensor_addbmm(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor CUDAIntType::s_addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toInt();
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toInt();
    auto batch1_ = checked_cast_tensor<CUDAIntTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CUDAIntTensor>(batch2.pImpl,"batch2",5, false);
    THCudaIntTensor_addbmm(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor & CUDAIntType::addbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toInt();
    auto alpha_ = alpha.toInt();
    auto batch1_ = checked_cast_tensor<CUDAIntTensor>(batch1.pImpl,"batch1",5, false);
    auto batch2_ = checked_cast_tensor<CUDAIntTensor>(batch2.pImpl,"batch2",6, false);
    THCudaIntTensor_addbmm(context->thc_state, self_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    return self;
}
Tensor & CUDAIntType::s_baddbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toInt();
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toInt();
    auto batch1_ = checked_cast_tensor<CUDAIntTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CUDAIntTensor>(batch2.pImpl,"batch2",5, false);
    THCudaIntTensor_baddbmm(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor CUDAIntType::s_baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toInt();
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toInt();
    auto batch1_ = checked_cast_tensor<CUDAIntTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CUDAIntTensor>(batch2.pImpl,"batch2",5, false);
    THCudaIntTensor_baddbmm(context->thc_state, result_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor & CUDAIntType::baddbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toInt();
    auto alpha_ = alpha.toInt();
    auto batch1_ = checked_cast_tensor<CUDAIntTensor>(batch1.pImpl,"batch1",5, false);
    auto batch2_ = checked_cast_tensor<CUDAIntTensor>(batch2.pImpl,"batch2",6, false);
    THCudaIntTensor_baddbmm(context->thc_state, self_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    return self;
}
Tensor & CUDAIntType::s_addcmul_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toInt();
    auto tensor1_ = checked_cast_tensor<CUDAIntTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CUDAIntTensor>(tensor2.pImpl,"tensor2",4, false);
    THCudaIntTensor_addcmul(context->thc_state, result_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor CUDAIntType::s_addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toInt();
    auto tensor1_ = checked_cast_tensor<CUDAIntTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CUDAIntTensor>(tensor2.pImpl,"tensor2",4, false);
    THCudaIntTensor_addcmul(context->thc_state, result_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor & CUDAIntType::s_addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toInt();
    auto tensor1_ = checked_cast_tensor<CUDAIntTensor>(tensor1.pImpl,"tensor1",4, false);
    auto tensor2_ = checked_cast_tensor<CUDAIntTensor>(tensor2.pImpl,"tensor2",5, false);
    THCudaIntTensor_addcmul(context->thc_state, self_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    return self;
}
Tensor & CUDAIntType::s_addcdiv_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toInt();
    auto tensor1_ = checked_cast_tensor<CUDAIntTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CUDAIntTensor>(tensor2.pImpl,"tensor2",4, false);
    THCudaIntTensor_addcdiv(context->thc_state, result_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor CUDAIntType::s_addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toInt();
    auto tensor1_ = checked_cast_tensor<CUDAIntTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CUDAIntTensor>(tensor2.pImpl,"tensor2",4, false);
    THCudaIntTensor_addcdiv(context->thc_state, result_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor & CUDAIntType::s_addcdiv_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toInt();
    auto tensor1_ = checked_cast_tensor<CUDAIntTensor>(tensor1.pImpl,"tensor1",4, false);
    auto tensor2_ = checked_cast_tensor<CUDAIntTensor>(tensor2.pImpl,"tensor2",5, false);
    THCudaIntTensor_addcdiv(context->thc_state, self_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    return self;
}
Tensor & CUDAIntType::random_(Tensor & self, int64_t from, int64_t to, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    THCudaIntTensor_clampedRandom(context->thc_state, self_->tensor, from, to);
    return self;
}
Tensor & CUDAIntType::random_(Tensor & self, int64_t to, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    THCudaIntTensor_cappedRandom(context->thc_state, self_->tensor, to);
    return self;
}
Tensor & CUDAIntType::random_(Tensor & self, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    THCudaIntTensor_random(context->thc_state, self_->tensor);
    return self;
}
Tensor & CUDAIntType::geometric_(Tensor & self, double p, Generator * generator) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CUDAGenerator>(generator, &context->defaultGenerator(backend()));
    (void) generator_; //silence unused warning
    THCudaIntTensor_geometric(context->thc_state, self_->tensor, p);
    return self;
}
Tensor CUDAIntType::tensor(Storage & storage, int64_t storageOffset, IntList size, IntList stride) const {
    auto storage_ = checked_cast_storage<CUDAIntStorage>(&storage,"storage",1);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    return Tensor((new CUDAIntTensor(context, THCudaIntTensor_newWithStorage(context->thc_state, storage_->storage, storageOffset, size_, stride_)))->maybeScalar(size.size() == 0),false);
}
Tensor CUDAIntType::tensor(IntList size) const {
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new CUDAIntTensor(context, THCudaIntTensor_newWithSize(context->thc_state, size_, NULL)))->maybeScalar(size.size() == 0),false);
}
Tensor CUDAIntType::tensor(IntList size, IntList stride) const {
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    return Tensor((new CUDAIntTensor(context, THCudaIntTensor_newWithSize(context->thc_state, size_, stride_)))->maybeScalar(size.size() == 0),false);
}
Tensor CUDAIntType::tensor() const {
    return Tensor((new CUDAIntTensor(context, THCudaIntTensor_new(context->thc_state))),false);
}
Tensor CUDAIntType::alias(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    return Tensor((new CUDAIntTensor(context, THCudaIntTensor_newWithTensor(context->thc_state, self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor & CUDAIntType::_copy_ignoring_overlaps_(Tensor & self, const Tensor & src) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto src_ = checked_cast_tensor<CUDAIntTensor>(src.pImpl,"src",2, false);
    THCudaIntTensor_copyIgnoringOverlaps(context->thc_state, self_->tensor, src_->tensor);
    return self;
}
Tensor & CUDAIntType::as_strided_out(Tensor & result, const Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    auto result_ = checked_cast_tensor<CUDAIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    if (storage_offset == -1) {
      storage_offset = self_->tensor->storageOffset;
    }
    THCudaIntTensor_setStorage(context->thc_state, result_->tensor, self_->tensor->storage, storage_offset, size_, stride_);
    result_->maybeScalar(size.size() == 0);
    return result;
}
Tensor CUDAIntType::as_strided(const Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    auto result_ = new CUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    if (storage_offset == -1) {
      storage_offset = self_->tensor->storageOffset;
    }
    THCudaIntTensor_setStorage(context->thc_state, result_->tensor, self_->tensor->storage, storage_offset, size_, stride_);
    result_->maybeScalar(size.size() == 0);
    return result;
}
Tensor & CUDAIntType::as_strided_(Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    if (storage_offset == -1) {
      storage_offset = self_->tensor->storageOffset;
    }
    THCudaIntTensor_setStorage(context->thc_state, self_->tensor, self_->tensor->storage, storage_offset, size_, stride_);
    self_->maybeScalar(size.size() == 0);
    return self;
}
Tensor & CUDAIntType::_cat_out(Tensor & self, TensorList tensors, int64_t dim) const {
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",0, false);
    auto tensors_ = tensor_list_checked_cast<CUDAIntTensor, Tensor, THCudaIntTensor>(tensors,"tensors",1);
    THCudaIntTensor_catArray(context->thc_state, self_->tensor, tensors_.data(), tensors_.size(), dim);
    return self;
}
Tensor CUDAIntType::_cat(TensorList tensors, int64_t dim) const {
    auto self_ = new CUDAIntTensor(context);
    auto self = Tensor(self_, false);
    auto tensors_ = tensor_list_checked_cast<CUDAIntTensor, Tensor, THCudaIntTensor>(tensors,"tensors",1);
    THCudaIntTensor_catArray(context->thc_state, self_->tensor, tensors_.data(), tensors_.size(), dim);
    return self;
}
Tensor CUDAIntType::_sparse_mask(const Tensor & self, SparseTensor mask) const {
    auto result_ = new SparseCUDAIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CUDAIntTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<SparseCUDAIntTensor>(mask.tref.pImpl,"mask",2,false);
    THCudaIntTensor_sparseMask(context->thc_state, result_->tensor, self_->tensor, mask_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CUDAIntType::abs_out(Tensor & result, const Tensor & self) const {
    return  at::native::_abs_out_cuda(result, self);
}
Tensor & CUDAIntType::ceil_out(Tensor & result, const Tensor & self) const {
    return  at::native::_ceil_out_cuda(result, self);
}
Tensor & CUDAIntType::cos_out(Tensor & result, const Tensor & self) const {
    return  at::native::_cos_out_cuda(result, self);
}
Tensor CUDAIntType::embedding_dense_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) const {
    return  at::native::embedding_backward_cuda(grad, indices, num_weights, padding_idx, scale_grad_by_freq);
}
Tensor & CUDAIntType::embedding_renorm_(Tensor & self, const Tensor & indices, double max_norm, double norm_type) const {
    return  at::native::embedding_renorm_cuda_(self, indices, max_norm, norm_type);
}
std::tuple<Tensor,Tensor,Tensor> CUDAIntType::embedding_bag(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse) const {
    return  at::native::embedding_bag_cuda(weight, indices, offsets, scale_grad_by_freq, mode, sparse);
}
Tensor CUDAIntType::embedding_bag_dense_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode) const {
    return  at::native::embedding_bag_backward_cuda(grad, indices, offsets, offset2bag, bag_size, num_weights, scale_grad_by_freq, mode);
}
Tensor & CUDAIntType::exp_out(Tensor & result, const Tensor & self) const {
    return  at::native::_exp_out_cuda(result, self);
}
Tensor & CUDAIntType::eye_out(Tensor & result, int64_t n, int64_t m) const {
    return  at::native::eye_out_cuda(result, n, m);
}
Tensor & CUDAIntType::floor_out(Tensor & result, const Tensor & self) const {
    return  at::native::_floor_out_cuda(result, self);
}
Tensor CUDAIntType::_fft_with_size(const Tensor & self, int64_t signal_ndim, bool complex_input, bool complex_output, bool inverse, IntList checked_signal_sizes, bool normalized, bool onesided, IntList output_sizes) const {
    return  at::native::_fft_cufft(self, signal_ndim, complex_input, complex_output, inverse, checked_signal_sizes, normalized, onesided, output_sizes);
}
Tensor & CUDAIntType::log_out(Tensor & result, const Tensor & self) const {
    return  at::native::_log_out_cuda(result, self);
}
std::tuple<Tensor,Tensor> CUDAIntType::RoiPooling2d_forward(const Tensor & input, const Tensor & rois, int64_t pooledHeight, int64_t pooledWidth, double spatialScale) const {
    return  at::native::RoiPooling2d_forward_cuda(input, rois, pooledHeight, pooledWidth, spatialScale);
}
Tensor CUDAIntType::RoiPooling2d_backward(const Tensor & input, const Tensor & rois, int64_t pooledHeight, int64_t pooledWidth, double spatialScale, const Tensor & gradOutput, const Tensor & argmaxes) const {
    return  at::native::RoiPooling2d_backward_cuda(input, rois, pooledHeight, pooledWidth, spatialScale, gradOutput, argmaxes);
}
Tensor & CUDAIntType::round_out(Tensor & result, const Tensor & self) const {
    return  at::native::_round_out_cuda(result, self);
}
Tensor & CUDAIntType::sin_out(Tensor & result, const Tensor & self) const {
    return  at::native::_sin_out_cuda(result, self);
}
Tensor & CUDAIntType::sspaddmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return  at::native::_sspaddmm_out_only_sparse(result, self, mat1, mat2, beta, alpha);
}
Tensor CUDAIntType::sum(const Tensor & self) const {
    return  at::native::_sum_cuda(self);
}
Tensor & CUDAIntType::sum_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    return  at::native::_sum_out_cuda(result, self, dim, keepdim);
}
Tensor & CUDAIntType::sqrt_out(Tensor & result, const Tensor & self) const {
    return  at::native::_sqrt_out_cuda(result, self);
}
Tensor CUDAIntType::prod(const Tensor & self) const {
    return  at::native::_prod_cuda(self);
}
Tensor & CUDAIntType::prod_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    return  at::native::_prod_out_cuda(result, self, dim, keepdim);
}
Tensor & CUDAIntType::trunc_out(Tensor & result, const Tensor & self) const {
    return  at::native::_trunc_out_cuda(result, self);
}
std::tuple<Tensor,Tensor> CUDAIntType::_unique(const Tensor & self, bool sorted, bool return_inverse) const {
    return  at::native::_unique_cuda(self, sorted, return_inverse);
}
Tensor CUDAIntType::_s_where(const Tensor & condition, const Tensor & self, const Tensor & other) const {
    return  at::native::_s_where_cuda(condition, self, other);
}
Tensor CUDAIntType::_standard_gamma_grad(const Tensor & self, const Tensor & output) const {
    return  at::native::_standard_gamma_grad_cuda(self, output);
}
Tensor CUDAIntType::poisson(const Tensor & self, Generator * generator) const {
    return  at::native::_s_poisson_cuda(self, generator);
}

}
