// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/CPUFloatType.h"
#include "ATen/CPUFloatStorage.h"
#include "ATen/CPUFloatTensor.h"
#include "ATen/CPUGenerator.h"
#include "ATen/CPUByteTensor.h"
#include "ATen/CPUIntTensor.h"
#include "ATen/CPULongTensor.h"
#include "ATen/SparseCPUFloatTensor.h"
#include "ATen/CPUFloatTensor.h"
#include "ATen/CPULongTensor.h"
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

#endif

namespace at {

CPUFloatType::CPUFloatType(Context* context)
: Type(context, /*is_variable_or_undefined=*/false) {}
ScalarType CPUFloatType::scalarType() const {
  return ScalarType::Float;
}
Backend CPUFloatType::backend() const {
  return Backend::CPU;
}
bool CPUFloatType::is_cuda() const { return backend() == kCUDA || backend() == kSparseCUDA; }
bool CPUFloatType::is_sparse() const { return backend() == kSparseCPU || backend() == kSparseCUDA; }
bool CPUFloatType::is_distributed() const { return false; }

std::unique_ptr<Storage> CPUFloatType::storage() const {
  return std::unique_ptr<Storage>(new CPUFloatStorage(context));
}
std::unique_ptr<Storage> CPUFloatType::storage(size_t size) const {
  return std::unique_ptr<Storage>(new CPUFloatStorage(context,size));
}
std::unique_ptr<Storage> CPUFloatType::storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const {
    return std::unique_ptr<Storage>(
      new CPUFloatStorage(context,data,size,deleter));
}
std::unique_ptr<Storage> CPUFloatType::storageWithAllocator(int64_t size, std::unique_ptr<Allocator> allocator) const {
    return std::unique_ptr<Storage>(
        new CPUFloatStorage(context, size, std::move(allocator)));
}
Tensor CPUFloatType::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THFloatTensor_retain( (THFloatTensor*) th_pointer);
  return Tensor(new CPUFloatTensor(context,(THFloatTensor*)(th_pointer)), false);
}
std::unique_ptr<Storage> CPUFloatType::unsafeStorageFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THFloatStorage_retain( (THFloatStorage*) th_pointer);
  return std::unique_ptr<Storage>(new CPUFloatStorage(context, (THFloatStorage*) th_pointer));
}
std::unique_ptr<Generator> CPUFloatType::generator() const {
  return std::unique_ptr<Generator>(new CPUGenerator(context));
}

const char * CPUFloatType::toString() const {
  return CPUFloatType::typeString();
}
TypeID CPUFloatType::ID() const {
  return TypeID::CPUFloat;
}

std::size_t CPUFloatType::elementSizeInBytes() const {
  return sizeof(float);
}

const char * CPUFloatType::typeString() {
  return "CPUFloatType";
}

/* example
Tensor * CPUFloatType::add(Tensor & a, Tensor & b) {
  std::cout << "add CPUFloatTensor\n";
  return &a;
}
*/

int64_t CPUFloatType::storage_offset(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    return static_cast<int64_t>(THFloatTensor_storageOffset(self_->tensor));
}
Tensor & CPUFloatType::resize_(Tensor & self, IntList size) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    THFloatTensor_resize(self_->tensor, size_, NULL);
    self_->maybeScalar(size.size() == 0);
    return self;
}
int64_t CPUFloatType::numel(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    return static_cast<int64_t>(THFloatTensor_nElement(self_->tensor));
}
Tensor & CPUFloatType::set_(Tensor & self, Storage & storage) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto storage_ = checked_cast_storage<CPUFloatStorage>(&storage,"storage",2);
    THFloatTensor_setStorage(self_->tensor, storage_->storage, 0, THLongStorageView::makeFromLength(static_cast<int64_t>(storage.size())), NULL);
    self_->maybeScalar(false);
    return self;
}
Tensor & CPUFloatType::set_(Tensor & self, Storage & sourceStorage, int64_t storage_offset, IntList size, IntList stride) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto sourceStorage_ = checked_cast_storage<CPUFloatStorage>(&sourceStorage,"sourceStorage",2);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    THFloatTensor_setStorage(self_->tensor, sourceStorage_->storage, storage_offset, size_, stride_);
    self_->maybeScalar(size.size() == 0);
    return self;
}
Tensor & CPUFloatType::set_(Tensor & self, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto source_ = checked_cast_tensor<CPUFloatTensor>(source.pImpl,"source",2, false);
    THFloatTensor_set(self_->tensor, source_->tensor);
    self_->maybeScalar(source_->isScalar());
    return self;
}
Tensor & CPUFloatType::set_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_setStorage(self_->tensor, NULL, 0, NULL, NULL);
    self_->maybeScalar(false);
    return self;
}
Tensor & CPUFloatType::fill_(Tensor & self, Scalar value) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toFloat();
    THFloatTensor_fill(self_->tensor, value_);
    return self;
}
Tensor & CPUFloatType::fill_(Tensor & self, const Tensor & value) const {
    if (value.dim() == 0) {
        return static_cast<const Type*>(this)->fill_(self, Scalar(value));
    }
    AT_ERROR("fill_ only supports a 0-dimensional value tensor, but got tensor "
        "with %" PRId64 " dimension(s)", value.dim());
}
bool CPUFloatType::is_contiguous(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    return THFloatTensor_isContiguous(self_->tensor);
}
bool CPUFloatType::is_set_to(const Tensor & self, const Tensor & tensor) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto tensor_ = checked_cast_tensor<CPUFloatTensor>(tensor.pImpl,"tensor",2, false);
    return THFloatTensor_isSetTo(self_->tensor, tensor_->tensor);
}
Tensor & CPUFloatType::s_masked_fill_(Tensor & self, const Tensor & mask, Scalar value) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CPUByteTensor>(mask.pImpl,"mask",2, false);
    auto value_ = value.toFloat();
    THFloatTensor_maskedFill(self_->tensor, mask_->tensor, value_);
    return self;
}
Tensor & CPUFloatType::s_masked_fill_(Tensor & self, const Tensor & mask, const Tensor & value) const {
    if (value.dim() == 0) {
        return static_cast<const Type*>(this)->masked_fill_(self, mask, Scalar(value));
    }
    AT_ERROR("masked_fill_ only supports a 0-dimensional value tensor, but got tensor "
        "with %" PRId64 " dimension(s)", value.dim());
}
Tensor & CPUFloatType::s_masked_scatter_(Tensor & self, const Tensor & mask, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CPUByteTensor>(mask.pImpl,"mask",2, false);
    auto source_ = checked_cast_tensor<CPUFloatTensor>(source.pImpl,"source",3, false);
    THFloatTensor_maskedCopy(self_->tensor, mask_->tensor, source_->tensor);
    return self;
}
Tensor & CPUFloatType::s_masked_select_out(Tensor & result, const Tensor & self, const Tensor & mask) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CPUByteTensor>(mask.pImpl,"mask",2, false);
    THFloatTensor_maskedSelect(result_->tensor, self_->tensor, mask_->tensor);
    result_->maybeScalar(self_->isScalar() && mask_->isScalar());
    return result;
}
Tensor CPUFloatType::s_masked_select(const Tensor & self, const Tensor & mask) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CPUByteTensor>(mask.pImpl,"mask",2, false);
    THFloatTensor_maskedSelect(result_->tensor, self_->tensor, mask_->tensor);
    result_->maybeScalar(self_->isScalar() && mask_->isScalar());
    return result;
}
Tensor CPUFloatType::transpose(const Tensor & self, int64_t dim0, int64_t dim1) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim0 = maybe_wrap_dim(dim0, self_);
    dim1 = maybe_wrap_dim(dim1, self_);
    return Tensor((new CPUFloatTensor(context, THFloatTensor_newTranspose(self_->tensor, dim0, dim1)))->maybeScalar(self_->isScalar()),false);
}
Tensor CPUFloatType::t(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPUFloatTensor(context, THFloatTensor_newTranspose(self_->tensor, 0, 1)))->maybeScalar(self_->isScalar()),false);
}
Tensor & CPUFloatType::nonzero_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_nonzero(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::nonzero(const Tensor & self) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_nonzero(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::contiguous(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPUFloatTensor(context, THFloatTensor_newContiguous(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor CPUFloatType::clone(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPUFloatTensor(context, THFloatTensor_newClone(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor CPUFloatType::view(const Tensor & self, IntList size) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new CPUFloatTensor(context, THFloatTensor_newView(self_->tensor, size_)))->maybeScalar(size.size() == 0),false);
}
Tensor & CPUFloatType::resize_as_(Tensor & self, const Tensor & the_template) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto the_template_ = checked_cast_tensor<CPUFloatTensor>(the_template.pImpl,"the_template",2, false);
    THFloatTensor_resizeAs(self_->tensor, the_template_->tensor);
    self_->maybeScalar(the_template_->isScalar());
    return self;
}
Tensor & CPUFloatType::index_select_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    THFloatTensor_indexSelect(result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
Tensor CPUFloatType::index_select(const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    THFloatTensor_indexSelect(result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
Tensor & CPUFloatType::_indexCopy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    auto source_ = checked_cast_tensor<CPUFloatTensor>(source.pImpl,"source",4, false);
    THFloatTensor_indexCopy(self_->tensor, dim, index_->tensor, source_->tensor);
    return self;
}
Tensor & CPUFloatType::take_out(Tensor & result, const Tensor & self, const Tensor & index) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",2, false);
    THFloatTensor_take(result_->tensor, self_->tensor, index_->tensor);
    result_->maybeScalar(index_->isScalar());
    return result;
}
Tensor CPUFloatType::take(const Tensor & self, const Tensor & index) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",2, false);
    THFloatTensor_take(result_->tensor, self_->tensor, index_->tensor);
    result_->maybeScalar(index_->isScalar());
    return result;
}
Tensor & CPUFloatType::put_(Tensor & self, const Tensor & index, const Tensor & source, bool accumulate) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",2, false);
    auto source_ = checked_cast_tensor<CPUFloatTensor>(source.pImpl,"source",3, false);
    THFloatTensor_put(self_->tensor, index_->tensor, source_->tensor, accumulate);
    return self;
}
Tensor & CPUFloatType::index_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    auto source_ = checked_cast_tensor<CPUFloatTensor>(source.pImpl,"source",4, false);
    THFloatTensor_indexAdd(self_->tensor, dim, index_->tensor, source_->tensor);
    return self;
}
Tensor & CPUFloatType::index_fill_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    auto value_ = value.toFloat();
    THFloatTensor_indexFill(self_->tensor, dim, index_->tensor, value_);
    return self;
}
Tensor & CPUFloatType::index_fill_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & value) const {
    if (value.dim() == 0) {
        return static_cast<const Type*>(this)->index_fill_(self, dim, index, Scalar(value));
    }
    AT_ERROR("index_fill_ only supports a 0-dimensional value tensor, but got tensor "
        "with %" PRId64 " dimension(s)", value.dim());
}
Tensor CPUFloatType::unfold(const Tensor & self, int64_t dimension, int64_t size, int64_t step) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dimension = maybe_wrap_dim(dimension, self_);
    THFloatTensor_unfold(result_->tensor, self_->tensor, dimension, size, step);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::_range_out(Tensor & result, Scalar start, Scalar end, Scalar step) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto start_ = start.toDouble();
    auto end_ = end.toDouble();
    auto step_ = step.toDouble();
    THFloatTensor_range(result_->tensor, start_, end_, step_);
    return result;
}
Tensor CPUFloatType::_range(Scalar start, Scalar end, Scalar step) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto start_ = start.toDouble();
    auto end_ = end.toDouble();
    auto step_ = step.toDouble();
    THFloatTensor_range(result_->tensor, start_, end_, step_);
    return result;
}
Tensor & CPUFloatType::_arange_out(Tensor & result, Scalar start, Scalar end, Scalar step) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto start_ = start.toDouble();
    auto end_ = end.toDouble();
    auto step_ = step.toDouble();
    THFloatTensor_arange(result_->tensor, start_, end_, step_);
    return result;
}
Tensor CPUFloatType::_arange(Scalar start, Scalar end, Scalar step) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto start_ = start.toDouble();
    auto end_ = end.toDouble();
    auto step_ = step.toDouble();
    THFloatTensor_arange(result_->tensor, start_, end_, step_);
    return result;
}
Tensor & CPUFloatType::_arange_out(Tensor & result, Scalar end) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto end_ = end.toDouble();
    THFloatTensor_arange(result_->tensor, 0, end_, 1);
    return result;
}
Tensor CPUFloatType::_arange(Scalar end) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto end_ = end.toDouble();
    THFloatTensor_arange(result_->tensor, 0, end_, 1);
    return result;
}
Tensor & CPUFloatType::scatter_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
    if (src.dim() == 0) {
        return static_cast<const Type*>(this)->scatter_(self, dim, index, Scalar(src));
    }
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    auto src_ = checked_cast_tensor<CPUFloatTensor>(src.pImpl,"src",4, false);
    THFloatTensor_scatter(self_->tensor, dim, index_->tensor, src_->tensor);
    return self;
}
Tensor & CPUFloatType::scatter_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    auto value_ = value.toFloat();
    THFloatTensor_scatterFill(self_->tensor, dim, index_->tensor, value_);
    return self;
}
Tensor & CPUFloatType::scatter_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    auto src_ = checked_cast_tensor<CPUFloatTensor>(src.pImpl,"src",4, false);
    THFloatTensor_scatterAdd(self_->tensor, dim, index_->tensor, src_->tensor);
    return self;
}
Tensor & CPUFloatType::gather_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    result.resize_(index.sizes());
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    THFloatTensor_gather(result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
Tensor CPUFloatType::gather(const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    result.resize_(index.sizes());
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    THFloatTensor_gather(result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
void* CPUFloatType::data_ptr(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    return THFloatTensor_data(self_->tensor);
}
bool CPUFloatType::equal(const Tensor & self, const Tensor & other) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",2, false);
    return THFloatTensor_equal(self_->tensor, other_->tensor);
}
Tensor & CPUFloatType::__and___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_bitand(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::__and__(const Tensor & self, Scalar other) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_bitand(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::s___and___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__and___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",2, false);
    THFloatTensor_cbitand(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUFloatType::s___and__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__and__(self, Scalar(other));
    }
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",2, false);
    THFloatTensor_cbitand(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUFloatType::__iand__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_bitand(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUFloatType::s___iand__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__iand__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",3, false);
    THFloatTensor_cbitand(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUFloatType::__or___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_bitor(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::__or__(const Tensor & self, Scalar other) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_bitor(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::s___or___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__or___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",2, false);
    THFloatTensor_cbitor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUFloatType::s___or__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__or__(self, Scalar(other));
    }
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",2, false);
    THFloatTensor_cbitor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUFloatType::__ior__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_bitor(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUFloatType::s___ior__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__ior__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",3, false);
    THFloatTensor_cbitor(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUFloatType::__xor___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_bitxor(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::__xor__(const Tensor & self, Scalar other) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_bitxor(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::s___xor___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__xor___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",2, false);
    THFloatTensor_cbitxor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUFloatType::s___xor__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__xor__(self, Scalar(other));
    }
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",2, false);
    THFloatTensor_cbitxor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUFloatType::__ixor__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_bitxor(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUFloatType::s___ixor__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__ixor__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",3, false);
    THFloatTensor_cbitxor(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUFloatType::__lshift___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_lshift(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::__lshift__(const Tensor & self, Scalar other) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_lshift(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::s___lshift___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__lshift___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",2, false);
    THFloatTensor_clshift(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUFloatType::s___lshift__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__lshift__(self, Scalar(other));
    }
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",2, false);
    THFloatTensor_clshift(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUFloatType::__ilshift__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_lshift(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUFloatType::s___ilshift__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__ilshift__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",3, false);
    THFloatTensor_clshift(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUFloatType::__rshift___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_rshift(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::__rshift__(const Tensor & self, Scalar other) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_rshift(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::s___rshift___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__rshift___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",2, false);
    THFloatTensor_crshift(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUFloatType::s___rshift__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__rshift__(self, Scalar(other));
    }
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",2, false);
    THFloatTensor_crshift(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUFloatType::__irshift__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_rshift(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUFloatType::s___irshift__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__irshift__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",3, false);
    THFloatTensor_crshift(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUFloatType::lt_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_ltValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::lt(const Tensor & self, Scalar other) const {
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_ltValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::s_lt_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->lt_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",2, false);
    THFloatTensor_ltTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUFloatType::s_lt(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->lt(self, Scalar(other));
    }
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",2, false);
    THFloatTensor_ltTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUFloatType::lt_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_ltValueT(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUFloatType::s_lt_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->lt_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",3, false);
    THFloatTensor_ltTensorT(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUFloatType::gt_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_gtValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::gt(const Tensor & self, Scalar other) const {
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_gtValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::s_gt_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->gt_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",2, false);
    THFloatTensor_gtTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUFloatType::s_gt(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->gt(self, Scalar(other));
    }
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",2, false);
    THFloatTensor_gtTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUFloatType::gt_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_gtValueT(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUFloatType::s_gt_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->gt_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",3, false);
    THFloatTensor_gtTensorT(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUFloatType::le_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_leValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::le(const Tensor & self, Scalar other) const {
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_leValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::s_le_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->le_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",2, false);
    THFloatTensor_leTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUFloatType::s_le(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->le(self, Scalar(other));
    }
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",2, false);
    THFloatTensor_leTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUFloatType::le_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_leValueT(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUFloatType::s_le_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->le_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",3, false);
    THFloatTensor_leTensorT(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUFloatType::ge_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_geValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::ge(const Tensor & self, Scalar other) const {
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_geValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::s_ge_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ge_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",2, false);
    THFloatTensor_geTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUFloatType::s_ge(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ge(self, Scalar(other));
    }
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",2, false);
    THFloatTensor_geTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUFloatType::ge_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_geValueT(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUFloatType::s_ge_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ge_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",3, false);
    THFloatTensor_geTensorT(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUFloatType::eq_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_eqValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::eq(const Tensor & self, Scalar other) const {
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_eqValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::s_eq_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->eq_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",2, false);
    THFloatTensor_eqTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUFloatType::s_eq(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->eq(self, Scalar(other));
    }
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",2, false);
    THFloatTensor_eqTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUFloatType::eq_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_eqValueT(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUFloatType::s_eq_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->eq_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",3, false);
    THFloatTensor_eqTensorT(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUFloatType::ne_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_neValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::ne(const Tensor & self, Scalar other) const {
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_neValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::s_ne_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ne_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",2, false);
    THFloatTensor_neTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUFloatType::s_ne(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ne(self, Scalar(other));
    }
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",2, false);
    THFloatTensor_neTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUFloatType::ne_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_neValueT(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUFloatType::s_ne_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ne_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",3, false);
    THFloatTensor_neTensorT(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
std::tuple<Tensor &,Tensor &> CPUFloatType::min_out(Tensor & min, Tensor & min_indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto min_ = checked_cast_tensor<CPUFloatTensor>(min.pImpl,"min",0, false);
    auto min_indices_ = checked_cast_tensor<CPULongTensor>(min_indices.pImpl,"min_indices",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THFloatTensor_min(min_->tensor, min_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    min_->maybeScalar(maybe_scalar);
    min_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(min, min_indices);
}
std::tuple<Tensor,Tensor> CPUFloatType::min(const Tensor & self, int64_t dim, bool keepdim) const {
    auto min_ = new CPUFloatTensor(context);
    auto min = Tensor(min_, false);
    auto min_indices_ = new CPULongTensor(context);
    auto min_indices = Tensor(min_indices_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THFloatTensor_min(min_->tensor, min_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    min_->maybeScalar(maybe_scalar);
    min_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(min, min_indices);
}
Tensor & CPUFloatType::s_min_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",2, false);
    THFloatTensor_cmin(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUFloatType::s_min(const Tensor & self, const Tensor & other) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",2, false);
    THFloatTensor_cmin(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUFloatType::min(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<float>(THFloatTensor_minall(self_->tensor)));
}
std::tuple<Tensor &,Tensor &> CPUFloatType::max_out(Tensor & max, Tensor & max_indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto max_ = checked_cast_tensor<CPUFloatTensor>(max.pImpl,"max",0, false);
    auto max_indices_ = checked_cast_tensor<CPULongTensor>(max_indices.pImpl,"max_indices",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THFloatTensor_max(max_->tensor, max_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    max_->maybeScalar(maybe_scalar);
    max_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(max, max_indices);
}
std::tuple<Tensor,Tensor> CPUFloatType::max(const Tensor & self, int64_t dim, bool keepdim) const {
    auto max_ = new CPUFloatTensor(context);
    auto max = Tensor(max_, false);
    auto max_indices_ = new CPULongTensor(context);
    auto max_indices = Tensor(max_indices_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THFloatTensor_max(max_->tensor, max_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    max_->maybeScalar(maybe_scalar);
    max_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(max, max_indices);
}
Tensor & CPUFloatType::s_max_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",2, false);
    THFloatTensor_cmax(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUFloatType::s_max(const Tensor & self, const Tensor & other) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",2, false);
    THFloatTensor_cmax(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUFloatType::max(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<float>(THFloatTensor_maxall(self_->tensor)));
}
std::tuple<Tensor &,Tensor &> CPUFloatType::kthvalue_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool keepdim) const {
    auto values_ = checked_cast_tensor<CPUFloatTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THFloatTensor_kthvalue(values_->tensor, indices_->tensor, self_->tensor, k, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CPUFloatType::kthvalue(const Tensor & self, int64_t k, int64_t dim, bool keepdim) const {
    auto values_ = new CPUFloatTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CPULongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THFloatTensor_kthvalue(values_->tensor, indices_->tensor, self_->tensor, k, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
std::tuple<Tensor &,Tensor &> CPUFloatType::mode_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = checked_cast_tensor<CPUFloatTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THFloatTensor_mode(values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CPUFloatType::mode(const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = new CPUFloatTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CPULongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THFloatTensor_mode(values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
std::tuple<Tensor &,Tensor &> CPUFloatType::median_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = checked_cast_tensor<CPUFloatTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THFloatTensor_median(values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CPUFloatType::median(const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = new CPUFloatTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CPULongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THFloatTensor_median(values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
Tensor CPUFloatType::median(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<float>(THFloatTensor_medianall(self_->tensor)));
}
std::tuple<Tensor &,Tensor &> CPUFloatType::sort_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool descending) const {
    auto values_ = checked_cast_tensor<CPUFloatTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THFloatTensor_sort(values_->tensor, indices_->tensor, self_->tensor, dim, descending);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CPUFloatType::sort(const Tensor & self, int64_t dim, bool descending) const {
    auto values_ = new CPUFloatTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CPULongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THFloatTensor_sort(values_->tensor, indices_->tensor, self_->tensor, dim, descending);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
std::tuple<Tensor &,Tensor &> CPUFloatType::topk_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) const {
    auto values_ = checked_cast_tensor<CPUFloatTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THFloatTensor_topk(values_->tensor, indices_->tensor, self_->tensor, k, dim, largest, sorted);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CPUFloatType::topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) const {
    auto values_ = new CPUFloatTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CPULongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THFloatTensor_topk(values_->tensor, indices_->tensor, self_->tensor, k, dim, largest, sorted);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
Tensor & CPUFloatType::_abs_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_abs(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::_abs(const Tensor & self) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_abs(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::sigmoid_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_sigmoid(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUFloatType::sigmoid_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_sigmoid(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::sigmoid(const Tensor & self) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_sigmoid(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::_log_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_log(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::_log(const Tensor & self) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_log(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::log10_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_log10(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUFloatType::log10_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_log10(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::log10(const Tensor & self) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_log10(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::log1p_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_log1p(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUFloatType::log1p_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_log1p(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::log1p(const Tensor & self) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_log1p(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::log2_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_log2(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUFloatType::log2_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_log2(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::log2(const Tensor & self) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_log2(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::lgamma_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_lgamma(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::lgamma(const Tensor & self) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_lgamma(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::lgamma_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_lgamma(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUFloatType::digamma_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_digamma(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::digamma(const Tensor & self) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_digamma(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::digamma_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_digamma(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUFloatType::polygamma_out(Tensor & result, int64_t n, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    THFloatTensor_polygamma(result_->tensor, n, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::polygamma(int64_t n, const Tensor & self) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    THFloatTensor_polygamma(result_->tensor, n, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::polygamma_(Tensor & self, int64_t n) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_polygamma(self_->tensor, n, self_->tensor);
    return self;
}
Tensor & CPUFloatType::_exp_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_exp(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::_exp(const Tensor & self) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_exp(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::expm1_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_expm1(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUFloatType::expm1_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_expm1(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::expm1(const Tensor & self) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_expm1(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::_cos_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_cos(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::_cos(const Tensor & self) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_cos(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::acos_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_acos(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUFloatType::acos_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_acos(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::acos(const Tensor & self) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_acos(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::cosh_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_cosh(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUFloatType::cosh_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_cosh(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::cosh(const Tensor & self) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_cosh(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::_sin_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_sin(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::_sin(const Tensor & self) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_sin(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::asin_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_asin(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUFloatType::asin_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_asin(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::asin(const Tensor & self) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_asin(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::sinh_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_sinh(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUFloatType::sinh_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_sinh(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::sinh(const Tensor & self) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_sinh(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::tan_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_tan(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUFloatType::tan_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_tan(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::tan(const Tensor & self) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_tan(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::atan_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_atan(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUFloatType::atan_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_atan(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::atan(const Tensor & self) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_atan(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::tanh_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_tanh(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUFloatType::tanh_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_tanh(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::tanh(const Tensor & self) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_tanh(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::erf_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_erf(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUFloatType::erf_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_erf(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::erf(const Tensor & self) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_erf(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::erfinv_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_erfinv(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUFloatType::erfinv_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_erfinv(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::erfinv(const Tensor & self) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_erfinv(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::_sqrt_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_sqrt(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::_sqrt(const Tensor & self) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_sqrt(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::rsqrt_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_rsqrt(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUFloatType::rsqrt_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_rsqrt(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::rsqrt(const Tensor & self) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_rsqrt(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::_ceil_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_ceil(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::_ceil(const Tensor & self) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_ceil(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::_floor_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_floor(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::_floor(const Tensor & self) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_floor(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::_round_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_round(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::_round(const Tensor & self) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_round(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::_trunc_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_trunc(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::_trunc(const Tensor & self) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_trunc(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::frac_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_frac(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUFloatType::frac_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_frac(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::frac(const Tensor & self) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_frac(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::mean_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THFloatTensor_mean(result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CPUFloatType::mean(const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THFloatTensor_mean(result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CPUFloatType::mean(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<float>(THFloatTensor_meanall(self_->tensor)));
}
Tensor & CPUFloatType::var_out(Tensor & result, const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THFloatTensor_var(result_->tensor, self_->tensor, dim, (unbiased) ? 0 : 1, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CPUFloatType::var(const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THFloatTensor_var(result_->tensor, self_->tensor, dim, (unbiased) ? 0 : 1, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CPUFloatType::var(const Tensor & self, bool unbiased) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<float>(THFloatTensor_varall(self_->tensor, (unbiased) ? 0 : 1)));
}
Tensor & CPUFloatType::std_out(Tensor & result, const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THFloatTensor_std(result_->tensor, self_->tensor, dim, (unbiased) ? 0 : 1, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CPUFloatType::std(const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THFloatTensor_std(result_->tensor, self_->tensor, dim, (unbiased) ? 0 : 1, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CPUFloatType::std(const Tensor & self, bool unbiased) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<float>(THFloatTensor_stdall(self_->tensor, (unbiased) ? 0 : 1)));
}
Tensor & CPUFloatType::norm_out(Tensor & result, const Tensor & self, Scalar p, int64_t dim, bool keepdim) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto p_ = p.toFloat();
    dim = maybe_wrap_dim(dim, self_);
    THFloatTensor_norm(result_->tensor, self_->tensor, p_, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CPUFloatType::norm(const Tensor & self, Scalar p, int64_t dim, bool keepdim) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto p_ = p.toFloat();
    dim = maybe_wrap_dim(dim, self_);
    THFloatTensor_norm(result_->tensor, self_->tensor, p_, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CPUFloatType::norm(const Tensor & self, Scalar p) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto p_ = p.toFloat();
    // norm(value) for a sparse tensor returns a DENSE 0-dim tensor
    if (self.is_sparse()) {
      auto result = THFloatTensor_normall( self_->tensor, convert<float>(p_));
      return toBackend(toDense(backend())).tensor({}).fill_(result);
    }
    // aten_custom_call is followed by the generated call to normall
    return scalarTensor(convert<float>(THFloatTensor_normall(self_->tensor, p_)));
}
Tensor & CPUFloatType::renorm_out(Tensor & result, const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto p_ = p.toFloat();
    dim = maybe_wrap_dim(dim, self_);
    auto maxnorm_ = maxnorm.toFloat();
    THFloatTensor_renorm(result_->tensor, self_->tensor, p_, dim, maxnorm_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::renorm(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto p_ = p.toFloat();
    dim = maybe_wrap_dim(dim, self_);
    auto maxnorm_ = maxnorm.toFloat();
    THFloatTensor_renorm(result_->tensor, self_->tensor, p_, dim, maxnorm_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::renorm_(Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto p_ = p.toFloat();
    dim = maybe_wrap_dim(dim, self_);
    auto maxnorm_ = maxnorm.toFloat();
    THFloatTensor_renorm(self_->tensor, self_->tensor, p_, dim, maxnorm_);
    return self;
}
Tensor CPUFloatType::s_dist(const Tensor & self, const Tensor & other, Scalar p) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",2, false);
    auto p_ = p.toFloat();
    return scalarTensor(convert<float>(THFloatTensor_dist(self_->tensor, other_->tensor, p_)));
}
Tensor & CPUFloatType::reciprocal_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_cinv(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::reciprocal(const Tensor & self) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_cinv(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::reciprocal_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_cinv(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUFloatType::neg_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_neg(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::neg(const Tensor & self) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_neg(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::neg_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_neg(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUFloatType::s_atan2_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",2, false);
    THFloatTensor_atan2(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUFloatType::s_atan2(const Tensor & self, const Tensor & other) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",2, false);
    THFloatTensor_atan2(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUFloatType::s_atan2_(Tensor & self, const Tensor & other) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",3, false);
    THFloatTensor_atan2(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUFloatType::pow_out(Tensor & result, const Tensor & self, Scalar exponent) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toFloat();
    THFloatTensor_pow(result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::pow(const Tensor & self, Scalar exponent) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toFloat();
    THFloatTensor_pow(result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::s_pow_out(Tensor & result, const Tensor & self, const Tensor & exponent) const {
    if (exponent.dim() == 0) {
        return static_cast<const Type*>(this)->pow_out(result, self, Scalar(exponent));
    }
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto exponent_ = checked_cast_tensor<CPUFloatTensor>(exponent.pImpl,"exponent",2, false);
    THFloatTensor_cpow(result_->tensor, self_->tensor, exponent_->tensor);
    result_->maybeScalar(self_->isScalar() && exponent_->isScalar());
    return result;
}
Tensor CPUFloatType::s_pow(const Tensor & self, const Tensor & exponent) const {
    if (exponent.dim() == 0) {
        return static_cast<const Type*>(this)->pow(self, Scalar(exponent));
    }
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto exponent_ = checked_cast_tensor<CPUFloatTensor>(exponent.pImpl,"exponent",2, false);
    THFloatTensor_cpow(result_->tensor, self_->tensor, exponent_->tensor);
    result_->maybeScalar(self_->isScalar() && exponent_->isScalar());
    return result;
}
Tensor & CPUFloatType::pow_out(Tensor & result, Scalar base, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto base_ = base.toFloat();
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    THFloatTensor_tpow(result_->tensor, base_, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::pow(Scalar base, const Tensor & self) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto base_ = base.toFloat();
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    THFloatTensor_tpow(result_->tensor, base_, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::pow_(Tensor & self, Scalar exponent) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toFloat();
    THFloatTensor_pow(self_->tensor, self_->tensor, exponent_);
    return self;
}
Tensor & CPUFloatType::s_pow_(Tensor & self, const Tensor & exponent) const {
    if (exponent.dim() == 0) {
        return static_cast<const Type*>(this)->pow_(self, Scalar(exponent));
    }
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto exponent_ = checked_cast_tensor<CPUFloatTensor>(exponent.pImpl,"exponent",3, false);
    THFloatTensor_cpow(self_->tensor, self_->tensor, exponent_->tensor);
    return self;
}
Tensor & CPUFloatType::s_lerp_out(Tensor & result, const Tensor & self, const Tensor & end, Scalar weight) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto end_ = checked_cast_tensor<CPUFloatTensor>(end.pImpl,"end",2, false);
    auto weight_ = weight.toFloat();
    THFloatTensor_lerp(result_->tensor, self_->tensor, end_->tensor, weight_);
    result_->maybeScalar(self_->isScalar() && end_->isScalar());
    return result;
}
Tensor CPUFloatType::s_lerp(const Tensor & self, const Tensor & end, Scalar weight) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto end_ = checked_cast_tensor<CPUFloatTensor>(end.pImpl,"end",2, false);
    auto weight_ = weight.toFloat();
    THFloatTensor_lerp(result_->tensor, self_->tensor, end_->tensor, weight_);
    result_->maybeScalar(self_->isScalar() && end_->isScalar());
    return result;
}
Tensor & CPUFloatType::s_lerp_(Tensor & self, const Tensor & end, Scalar weight) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto end_ = checked_cast_tensor<CPUFloatTensor>(end.pImpl,"end",3, false);
    auto weight_ = weight.toFloat();
    THFloatTensor_lerp(self_->tensor, self_->tensor, end_->tensor, weight_);
    return self;
}
Tensor & CPUFloatType::_linspace_out(Tensor & result, Scalar start, Scalar end, int64_t steps) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto start_ = start.toFloat();
    auto end_ = end.toFloat();
    THFloatTensor_linspace(result_->tensor, start_, end_, steps);
    return result;
}
Tensor CPUFloatType::_linspace(Scalar start, Scalar end, int64_t steps) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto start_ = start.toFloat();
    auto end_ = end.toFloat();
    THFloatTensor_linspace(result_->tensor, start_, end_, steps);
    return result;
}
Tensor & CPUFloatType::_logspace_out(Tensor & result, Scalar start, Scalar end, int64_t steps) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto start_ = start.toFloat();
    auto end_ = end.toFloat();
    THFloatTensor_logspace(result_->tensor, start_, end_, steps);
    return result;
}
Tensor CPUFloatType::_logspace(Scalar start, Scalar end, int64_t steps) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto start_ = start.toFloat();
    auto end_ = end.toFloat();
    THFloatTensor_logspace(result_->tensor, start_, end_, steps);
    return result;
}
Tensor & CPUFloatType::histc_out(Tensor & result, const Tensor & self, int64_t bins, Scalar min, Scalar max) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toFloat();
    auto max_ = max.toFloat();
    THFloatTensor_histc(result_->tensor, self_->tensor, bins, min_, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::histc(const Tensor & self, int64_t bins, Scalar min, Scalar max) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toFloat();
    auto max_ = max.toFloat();
    THFloatTensor_histc(result_->tensor, self_->tensor, bins, min_, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::zero_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_zero(self_->tensor);
    return self;
}
Tensor CPUFloatType::_sumall(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<float>(THFloatTensor_sumall(self_->tensor)));
}
Tensor & CPUFloatType::_sum_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THFloatTensor_sum(result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CPUFloatType::_sum(const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THFloatTensor_sum(result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CPUFloatType::_prodall(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<float>(THFloatTensor_prodall(self_->tensor)));
}
Tensor & CPUFloatType::_prod_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THFloatTensor_prod(result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CPUFloatType::_prod(const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THFloatTensor_prod(result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor & CPUFloatType::cumsum_out(Tensor & result, const Tensor & self, int64_t dim) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THFloatTensor_cumsum(result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::cumsum(const Tensor & self, int64_t dim) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THFloatTensor_cumsum(result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::cumprod_out(Tensor & result, const Tensor & self, int64_t dim) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THFloatTensor_cumprod(result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::cumprod(const Tensor & self, int64_t dim) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THFloatTensor_cumprod(result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::sign_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_sign(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::sign(const Tensor & self) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_sign(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::sign_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_sign(self_->tensor, self_->tensor);
    return self;
}
Tensor CPUFloatType::trace(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<float>(THFloatTensor_trace(self_->tensor)));
}
Tensor & CPUFloatType::add_out(Tensor & result, const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    auto alpha_ = alpha.toFloat();
    THFloatTensor_add_scaled(result_->tensor, self_->tensor, other_, alpha_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::add(const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    auto alpha_ = alpha.toFloat();
    THFloatTensor_add_scaled(result_->tensor, self_->tensor, other_, alpha_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::s_add_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if(other.type().is_sparse()) {
        return static_cast<const Type*>(this)->add_out(result, self, SparseTensor(other), alpha);
    }
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toFloat();
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",3, false);
    THFloatTensor_cadd(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUFloatType::s_add(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if(other.type().is_sparse()) {
        return static_cast<const Type*>(this)->add(self, SparseTensor(other), alpha);
    }
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add(self, Scalar(other), alpha);
    }
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toFloat();
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",3, false);
    THFloatTensor_cadd(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUFloatType::add_out(Tensor & result, const Tensor & self, SparseTensor other, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toFloat();
    auto other_ = checked_cast_tensor<SparseCPUFloatTensor>(other.tref.pImpl,"other",3,false);
    THSFloatTensor_spcadd(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::add(const Tensor & self, SparseTensor other, Scalar alpha) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toFloat();
    auto other_ = checked_cast_tensor<SparseCPUFloatTensor>(other.tref.pImpl,"other",3,false);
    THSFloatTensor_spcadd(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::add_(Tensor & self, Scalar other, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    auto alpha_ = alpha.toFloat();
    THFloatTensor_add_scaled(self_->tensor, self_->tensor, other_, alpha_);
    return self;
}
Tensor & CPUFloatType::s_add_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if(other.type().is_sparse()) {
        return static_cast<const Type*>(this)->add_(self, SparseTensor(other), alpha);
    }
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toFloat();
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",4, false);
    THFloatTensor_cadd(self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & CPUFloatType::add_(Tensor & self, SparseTensor other, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toFloat();
    auto other_ = checked_cast_tensor<SparseCPUFloatTensor>(other.tref.pImpl,"other",4,false);
    THSFloatTensor_spcadd(self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & CPUFloatType::sub_out(Tensor & result, const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    auto alpha_ = alpha.toFloat();
    THFloatTensor_sub_scaled(result_->tensor, self_->tensor, other_, alpha_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::sub(const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    auto alpha_ = alpha.toFloat();
    THFloatTensor_sub_scaled(result_->tensor, self_->tensor, other_, alpha_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::s_sub_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toFloat();
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",3, false);
    THFloatTensor_csub(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUFloatType::s_sub(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub(self, Scalar(other), alpha);
    }
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toFloat();
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",3, false);
    THFloatTensor_csub(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUFloatType::sub_(Tensor & self, Scalar other, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    auto alpha_ = alpha.toFloat();
    THFloatTensor_sub_scaled(self_->tensor, self_->tensor, other_, alpha_);
    return self;
}
Tensor & CPUFloatType::s_sub_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toFloat();
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",4, false);
    THFloatTensor_csub(self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & CPUFloatType::mul_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_mul(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::mul(const Tensor & self, Scalar other) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_mul(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::s_mul_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",2, false);
    THFloatTensor_cmul(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUFloatType::s_mul(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul(self, Scalar(other));
    }
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",2, false);
    THFloatTensor_cmul(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUFloatType::mul_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_mul(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUFloatType::s_mul_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",3, false);
    THFloatTensor_cmul(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUFloatType::div_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_div(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::div(const Tensor & self, Scalar other) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_div(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::s_div_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->div_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",2, false);
    THFloatTensor_cdiv(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUFloatType::s_div(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->div(self, Scalar(other));
    }
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",2, false);
    THFloatTensor_cdiv(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUFloatType::div_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_div(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUFloatType::s_div_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->div_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",3, false);
    THFloatTensor_cdiv(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUFloatType::fmod_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_fmod(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::fmod(const Tensor & self, Scalar other) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_fmod(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::s_fmod_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->fmod_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",2, false);
    THFloatTensor_cfmod(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUFloatType::s_fmod(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->fmod(self, Scalar(other));
    }
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",2, false);
    THFloatTensor_cfmod(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUFloatType::fmod_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_fmod(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUFloatType::s_fmod_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->fmod_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",3, false);
    THFloatTensor_cfmod(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUFloatType::remainder_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_remainder(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::remainder(const Tensor & self, Scalar other) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_remainder(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::s_remainder_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->remainder_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",2, false);
    THFloatTensor_cremainder(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUFloatType::s_remainder(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->remainder(self, Scalar(other));
    }
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",2, false);
    THFloatTensor_cremainder(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUFloatType::remainder_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toFloat();
    THFloatTensor_remainder(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUFloatType::s_remainder_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->remainder_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",3, false);
    THFloatTensor_cremainder(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUFloatType::clamp_out(Tensor & result, const Tensor & self, Scalar min, Scalar max) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toFloat();
    auto max_ = max.toFloat();
    THFloatTensor_clamp(result_->tensor, self_->tensor, min_, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::clamp(const Tensor & self, Scalar min, Scalar max) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toFloat();
    auto max_ = max.toFloat();
    THFloatTensor_clamp(result_->tensor, self_->tensor, min_, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::clamp_(Tensor & self, Scalar min, Scalar max) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toFloat();
    auto max_ = max.toFloat();
    THFloatTensor_clamp(self_->tensor, self_->tensor, min_, max_);
    return self;
}
Tensor & CPUFloatType::clamp_min_out(Tensor & result, const Tensor & self, Scalar min) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toFloat();
    THFloatTensor_cmaxValue(result_->tensor, self_->tensor, min_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::clamp_min(const Tensor & self, Scalar min) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toFloat();
    THFloatTensor_cmaxValue(result_->tensor, self_->tensor, min_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::clamp_min_(Tensor & self, Scalar min) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toFloat();
    THFloatTensor_cmaxValue(self_->tensor, self_->tensor, min_);
    return self;
}
Tensor & CPUFloatType::clamp_max_out(Tensor & result, const Tensor & self, Scalar max) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto max_ = max.toFloat();
    THFloatTensor_cminValue(result_->tensor, self_->tensor, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::clamp_max(const Tensor & self, Scalar max) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto max_ = max.toFloat();
    THFloatTensor_cminValue(result_->tensor, self_->tensor, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::clamp_max_(Tensor & self, Scalar max) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto max_ = max.toFloat();
    THFloatTensor_cminValue(self_->tensor, self_->tensor, max_);
    return self;
}
Tensor CPUFloatType::_dot(const Tensor & self, const Tensor & tensor) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto tensor_ = checked_cast_tensor<CPUFloatTensor>(tensor.pImpl,"tensor",2, false);
    return scalarTensor(convert<float>(THFloatTensor_dot(self_->tensor, tensor_->tensor)));
}
Tensor & CPUFloatType::tril_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_tril(result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::tril(const Tensor & self, int64_t diagonal) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_tril(result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::tril_(Tensor & self, int64_t diagonal) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_tril(self_->tensor, self_->tensor, diagonal);
    return self;
}
Tensor & CPUFloatType::triu_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_triu(result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::triu(const Tensor & self, int64_t diagonal) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_triu(result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::triu_(Tensor & self, int64_t diagonal) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_triu(self_->tensor, self_->tensor, diagonal);
    return self;
}
Tensor & CPUFloatType::cross_out(Tensor & result, const Tensor & self, const Tensor & other, int64_t dim) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",2, false);
    THFloatTensor_cross(result_->tensor, self_->tensor, other_->tensor, dim);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUFloatType::cross(const Tensor & self, const Tensor & other, int64_t dim) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUFloatTensor>(other.pImpl,"other",2, false);
    THFloatTensor_cross(result_->tensor, self_->tensor, other_->tensor, dim);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUFloatType::diag_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    if (self_->isScalar()) {
      throw std::runtime_error("Input must be 1-d or 2-d");
    }
    THFloatTensor_diag(result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::diag(const Tensor & self, int64_t diagonal) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    if (self_->isScalar()) {
      throw std::runtime_error("Input must be 1-d or 2-d");
    }
    THFloatTensor_diag(result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::s_addmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    if(mat1.type().is_sparse()) {
        return static_cast<const Type*>(this)->addmm_out(result, self, SparseTensor(mat1), mat2, beta, alpha);
    }
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toFloat();
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toFloat();
    auto mat1_ = checked_cast_tensor<CPUFloatTensor>(mat1.pImpl,"mat1",4, false);
    auto mat2_ = checked_cast_tensor<CPUFloatTensor>(mat2.pImpl,"mat2",5, false);
    THFloatTensor_addmm(result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CPUFloatType::s_addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    if(mat1.type().is_sparse()) {
        return static_cast<const Type*>(this)->addmm(self, SparseTensor(mat1), mat2, beta, alpha);
    }
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toFloat();
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toFloat();
    auto mat1_ = checked_cast_tensor<CPUFloatTensor>(mat1.pImpl,"mat1",4, false);
    auto mat2_ = checked_cast_tensor<CPUFloatTensor>(mat2.pImpl,"mat2",5, false);
    THFloatTensor_addmm(result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CPUFloatType::addmm_out(Tensor & result, const Tensor & self, SparseTensor mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toFloat();
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toFloat();
    auto mat1_ = checked_cast_tensor<SparseCPUFloatTensor>(mat1.tref.pImpl,"mat1",4,false);
    auto mat2_ = checked_cast_tensor<CPUFloatTensor>(mat2.pImpl,"mat2",5, false);
    THSFloatTensor_spaddmm(result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CPUFloatType::addmm(const Tensor & self, SparseTensor mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toFloat();
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toFloat();
    auto mat1_ = checked_cast_tensor<SparseCPUFloatTensor>(mat1.tref.pImpl,"mat1",4,false);
    auto mat2_ = checked_cast_tensor<CPUFloatTensor>(mat2.pImpl,"mat2",5, false);
    THSFloatTensor_spaddmm(result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CPUFloatType::addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    if(mat1.type().is_sparse()) {
        return static_cast<const Type*>(this)->addmm_(self, SparseTensor(mat1), mat2, beta, alpha);
    }
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toFloat();
    auto alpha_ = alpha.toFloat();
    auto mat1_ = checked_cast_tensor<CPUFloatTensor>(mat1.pImpl,"mat1",5, false);
    auto mat2_ = checked_cast_tensor<CPUFloatTensor>(mat2.pImpl,"mat2",6, false);
    THFloatTensor_addmm(self_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    return self;
}
Tensor & CPUFloatType::addmm_(Tensor & self, SparseTensor mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toFloat();
    auto alpha_ = alpha.toFloat();
    auto mat1_ = checked_cast_tensor<SparseCPUFloatTensor>(mat1.tref.pImpl,"mat1",5,false);
    auto mat2_ = checked_cast_tensor<CPUFloatTensor>(mat2.pImpl,"mat2",6, false);
    THSFloatTensor_spaddmm(self_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    return self;
}
Tensor & CPUFloatType::s__addmv_out(Tensor & result, const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toFloat();
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toFloat();
    auto mat_ = checked_cast_tensor<CPUFloatTensor>(mat.pImpl,"mat",4, false);
    auto vec_ = checked_cast_tensor<CPUFloatTensor>(vec.pImpl,"vec",5, false);
    THFloatTensor_addmv(result_->tensor, beta_, self_->tensor, alpha_, mat_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && mat_->isScalar() && vec_->isScalar());
    return result;
}
Tensor CPUFloatType::s__addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toFloat();
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toFloat();
    auto mat_ = checked_cast_tensor<CPUFloatTensor>(mat.pImpl,"mat",4, false);
    auto vec_ = checked_cast_tensor<CPUFloatTensor>(vec.pImpl,"vec",5, false);
    THFloatTensor_addmv(result_->tensor, beta_, self_->tensor, alpha_, mat_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && mat_->isScalar() && vec_->isScalar());
    return result;
}
Tensor & CPUFloatType::_addmv_(Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toFloat();
    auto alpha_ = alpha.toFloat();
    auto mat_ = checked_cast_tensor<CPUFloatTensor>(mat.pImpl,"mat",5, false);
    auto vec_ = checked_cast_tensor<CPUFloatTensor>(vec.pImpl,"vec",6, false);
    THFloatTensor_addmv(self_->tensor, beta_, self_->tensor, alpha_, mat_->tensor, vec_->tensor);
    return self;
}
Tensor & CPUFloatType::s__addr_out(Tensor & result, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toFloat();
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toFloat();
    auto vec1_ = checked_cast_tensor<CPUFloatTensor>(vec1.pImpl,"vec1",4, false);
    auto vec2_ = checked_cast_tensor<CPUFloatTensor>(vec2.pImpl,"vec2",5, false);
    THFloatTensor_addr(result_->tensor, beta_, self_->tensor, alpha_, vec1_->tensor, vec2_->tensor);
    result_->maybeScalar(self_->isScalar() && vec1_->isScalar() && vec2_->isScalar());
    return result;
}
Tensor CPUFloatType::s__addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toFloat();
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toFloat();
    auto vec1_ = checked_cast_tensor<CPUFloatTensor>(vec1.pImpl,"vec1",4, false);
    auto vec2_ = checked_cast_tensor<CPUFloatTensor>(vec2.pImpl,"vec2",5, false);
    THFloatTensor_addr(result_->tensor, beta_, self_->tensor, alpha_, vec1_->tensor, vec2_->tensor);
    result_->maybeScalar(self_->isScalar() && vec1_->isScalar() && vec2_->isScalar());
    return result;
}
Tensor & CPUFloatType::_addr_(Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toFloat();
    auto alpha_ = alpha.toFloat();
    auto vec1_ = checked_cast_tensor<CPUFloatTensor>(vec1.pImpl,"vec1",5, false);
    auto vec2_ = checked_cast_tensor<CPUFloatTensor>(vec2.pImpl,"vec2",6, false);
    THFloatTensor_addr(self_->tensor, beta_, self_->tensor, alpha_, vec1_->tensor, vec2_->tensor);
    return self;
}
Tensor & CPUFloatType::_ger_out(Tensor & result, const Tensor & self, const Tensor & vec2) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.dim() == 0 ? 1 : self.size(0),vec2.dim() == 0 ? 1 : vec2.size(0) });
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto vec2_ = checked_cast_tensor<CPUFloatTensor>(vec2.pImpl,"vec2",2, false);
    THFloatTensor_addr(result_->tensor, float(0), result_->tensor, float(1), self_->tensor, vec2_->tensor);
    result_->maybeScalar(false);
    return result;
}
Tensor CPUFloatType::_ger(const Tensor & self, const Tensor & vec2) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.dim() == 0 ? 1 : self.size(0),vec2.dim() == 0 ? 1 : vec2.size(0) });
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto vec2_ = checked_cast_tensor<CPUFloatTensor>(vec2.pImpl,"vec2",2, false);
    THFloatTensor_addr(result_->tensor, float(0), result_->tensor, float(1), self_->tensor, vec2_->tensor);
    result_->maybeScalar(false);
    return result;
}
Tensor & CPUFloatType::_mv_out(Tensor & result, const Tensor & self, const Tensor & vec) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.size(0) });
    result.zero_();
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto vec_ = checked_cast_tensor<CPUFloatTensor>(vec.pImpl,"vec",2, false);
    THFloatTensor_addmv(result_->tensor, float(0), result_->tensor, float(1), self_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && vec_->isScalar());
    return result;
}
Tensor CPUFloatType::_mv(const Tensor & self, const Tensor & vec) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.size(0) });
    result.zero_();
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto vec_ = checked_cast_tensor<CPUFloatTensor>(vec.pImpl,"vec",2, false);
    THFloatTensor_addmv(result_->tensor, float(0), result_->tensor, float(1), self_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && vec_->isScalar());
    return result;
}
Tensor & CPUFloatType::_mm_out(Tensor & result, const Tensor & self, const Tensor & mat2) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.size(0),mat2.size(1) });
    result.zero_();
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CPUFloatTensor>(mat2.pImpl,"mat2",2, false);
    THFloatTensor_addmm(result_->tensor, float(0), result_->tensor, float(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CPUFloatType::_mm(const Tensor & self, const Tensor & mat2) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.size(0),mat2.size(1) });
    result.zero_();
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CPUFloatTensor>(mat2.pImpl,"mat2",2, false);
    THFloatTensor_addmm(result_->tensor, float(0), result_->tensor, float(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CPUFloatType::bmm_out(Tensor & result, const Tensor & self, const Tensor & mat2) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.size(0),self.size(1),mat2.size(2) });
    result.zero_();
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CPUFloatTensor>(mat2.pImpl,"mat2",2, false);
    THFloatTensor_baddbmm(result_->tensor, float(0), result_->tensor, float(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CPUFloatType::bmm(const Tensor & self, const Tensor & mat2) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.size(0),self.size(1),mat2.size(2) });
    result.zero_();
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CPUFloatTensor>(mat2.pImpl,"mat2",2, false);
    THFloatTensor_baddbmm(result_->tensor, float(0), result_->tensor, float(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CPUFloatType::s_addbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toFloat();
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toFloat();
    auto batch1_ = checked_cast_tensor<CPUFloatTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CPUFloatTensor>(batch2.pImpl,"batch2",5, false);
    THFloatTensor_addbmm(result_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor CPUFloatType::s_addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toFloat();
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toFloat();
    auto batch1_ = checked_cast_tensor<CPUFloatTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CPUFloatTensor>(batch2.pImpl,"batch2",5, false);
    THFloatTensor_addbmm(result_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor & CPUFloatType::addbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toFloat();
    auto alpha_ = alpha.toFloat();
    auto batch1_ = checked_cast_tensor<CPUFloatTensor>(batch1.pImpl,"batch1",5, false);
    auto batch2_ = checked_cast_tensor<CPUFloatTensor>(batch2.pImpl,"batch2",6, false);
    THFloatTensor_addbmm(self_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    return self;
}
Tensor & CPUFloatType::s_baddbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toFloat();
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toFloat();
    auto batch1_ = checked_cast_tensor<CPUFloatTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CPUFloatTensor>(batch2.pImpl,"batch2",5, false);
    THFloatTensor_baddbmm(result_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor CPUFloatType::s_baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toFloat();
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toFloat();
    auto batch1_ = checked_cast_tensor<CPUFloatTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CPUFloatTensor>(batch2.pImpl,"batch2",5, false);
    THFloatTensor_baddbmm(result_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor & CPUFloatType::baddbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toFloat();
    auto alpha_ = alpha.toFloat();
    auto batch1_ = checked_cast_tensor<CPUFloatTensor>(batch1.pImpl,"batch1",5, false);
    auto batch2_ = checked_cast_tensor<CPUFloatTensor>(batch2.pImpl,"batch2",6, false);
    THFloatTensor_baddbmm(self_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    return self;
}
Tensor & CPUFloatType::s_addcmul_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toFloat();
    auto tensor1_ = checked_cast_tensor<CPUFloatTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CPUFloatTensor>(tensor2.pImpl,"tensor2",4, false);
    THFloatTensor_addcmul(result_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor CPUFloatType::s_addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toFloat();
    auto tensor1_ = checked_cast_tensor<CPUFloatTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CPUFloatTensor>(tensor2.pImpl,"tensor2",4, false);
    THFloatTensor_addcmul(result_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor & CPUFloatType::s_addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toFloat();
    auto tensor1_ = checked_cast_tensor<CPUFloatTensor>(tensor1.pImpl,"tensor1",4, false);
    auto tensor2_ = checked_cast_tensor<CPUFloatTensor>(tensor2.pImpl,"tensor2",5, false);
    THFloatTensor_addcmul(self_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    return self;
}
Tensor & CPUFloatType::s_addcdiv_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toFloat();
    auto tensor1_ = checked_cast_tensor<CPUFloatTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CPUFloatTensor>(tensor2.pImpl,"tensor2",4, false);
    THFloatTensor_addcdiv(result_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor CPUFloatType::s_addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toFloat();
    auto tensor1_ = checked_cast_tensor<CPUFloatTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CPUFloatTensor>(tensor2.pImpl,"tensor2",4, false);
    THFloatTensor_addcdiv(result_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor & CPUFloatType::s_addcdiv_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toFloat();
    auto tensor1_ = checked_cast_tensor<CPUFloatTensor>(tensor1.pImpl,"tensor1",4, false);
    auto tensor2_ = checked_cast_tensor<CPUFloatTensor>(tensor2.pImpl,"tensor2",5, false);
    THFloatTensor_addcdiv(self_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    return self;
}
std::tuple<Tensor &,Tensor &> CPUFloatType::gesv_out(Tensor & solution, Tensor & lu, const Tensor & self, const Tensor & A) const {
    auto solution_ = checked_cast_tensor<CPUFloatTensor>(solution.pImpl,"solution",0, false);
    auto lu_ = checked_cast_tensor<CPUFloatTensor>(lu.pImpl,"lu",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto A_ = checked_cast_tensor<CPUFloatTensor>(A.pImpl,"A",2, false);
    THFloatTensor_gesv(solution_->tensor, lu_->tensor, self_->tensor, A_->tensor);
    bool maybe_scalar = self_->isScalar() && A_->isScalar();
    solution_->maybeScalar(maybe_scalar);
    lu_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(solution, lu);
}
std::tuple<Tensor,Tensor> CPUFloatType::gesv(const Tensor & self, const Tensor & A) const {
    auto solution_ = new CPUFloatTensor(context);
    auto solution = Tensor(solution_, false);
    auto lu_ = new CPUFloatTensor(context);
    auto lu = Tensor(lu_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto A_ = checked_cast_tensor<CPUFloatTensor>(A.pImpl,"A",2, false);
    THFloatTensor_gesv(solution_->tensor, lu_->tensor, self_->tensor, A_->tensor);
    bool maybe_scalar = self_->isScalar() && A_->isScalar();
    solution_->maybeScalar(maybe_scalar);
    lu_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(solution, lu);
}
std::tuple<Tensor &,Tensor &> CPUFloatType::gels_out(Tensor & res1, Tensor & res2, const Tensor & self, const Tensor & A) const {
    auto res1_ = checked_cast_tensor<CPUFloatTensor>(res1.pImpl,"res1",0, false);
    auto res2_ = checked_cast_tensor<CPUFloatTensor>(res2.pImpl,"res2",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto A_ = checked_cast_tensor<CPUFloatTensor>(A.pImpl,"A",2, false);
    THFloatTensor_gels(res1_->tensor, res2_->tensor, self_->tensor, A_->tensor);
    bool maybe_scalar = self_->isScalar() && A_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(res1, res2);
}
std::tuple<Tensor,Tensor> CPUFloatType::gels(const Tensor & self, const Tensor & A) const {
    auto res1_ = new CPUFloatTensor(context);
    auto res1 = Tensor(res1_, false);
    auto res2_ = new CPUFloatTensor(context);
    auto res2 = Tensor(res2_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto A_ = checked_cast_tensor<CPUFloatTensor>(A.pImpl,"A",2, false);
    THFloatTensor_gels(res1_->tensor, res2_->tensor, self_->tensor, A_->tensor);
    bool maybe_scalar = self_->isScalar() && A_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(res1, res2);
}
std::tuple<Tensor &,Tensor &> CPUFloatType::trtrs_out(Tensor & res1, Tensor & res2, const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) const {
    auto res1_ = checked_cast_tensor<CPUFloatTensor>(res1.pImpl,"res1",0, false);
    auto res2_ = checked_cast_tensor<CPUFloatTensor>(res2.pImpl,"res2",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto A_ = checked_cast_tensor<CPUFloatTensor>(A.pImpl,"A",2, false);
    THFloatTensor_trtrs(res1_->tensor, res2_->tensor, self_->tensor, A_->tensor, (upper) ? "U" : "L", (transpose) ? "T" : "N", (unitriangular) ? "U" : "N");
    bool maybe_scalar = self_->isScalar() && A_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(res1, res2);
}
std::tuple<Tensor,Tensor> CPUFloatType::trtrs(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) const {
    auto res1_ = new CPUFloatTensor(context);
    auto res1 = Tensor(res1_, false);
    auto res2_ = new CPUFloatTensor(context);
    auto res2 = Tensor(res2_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto A_ = checked_cast_tensor<CPUFloatTensor>(A.pImpl,"A",2, false);
    THFloatTensor_trtrs(res1_->tensor, res2_->tensor, self_->tensor, A_->tensor, (upper) ? "U" : "L", (transpose) ? "T" : "N", (unitriangular) ? "U" : "N");
    bool maybe_scalar = self_->isScalar() && A_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(res1, res2);
}
std::tuple<Tensor &,Tensor &> CPUFloatType::symeig_out(Tensor & res1, Tensor & res2, const Tensor & self, bool eigenvectors, bool upper) const {
    auto res1_ = checked_cast_tensor<CPUFloatTensor>(res1.pImpl,"res1",0, false);
    auto res2_ = checked_cast_tensor<CPUFloatTensor>(res2.pImpl,"res2",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_syev(res1_->tensor, res2_->tensor, self_->tensor, (eigenvectors) ? "V" : "N", (upper) ? "U" : "L");
    bool maybe_scalar = self_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(res1, res2);
}
std::tuple<Tensor,Tensor> CPUFloatType::symeig(const Tensor & self, bool eigenvectors, bool upper) const {
    auto res1_ = new CPUFloatTensor(context);
    auto res1 = Tensor(res1_, false);
    auto res2_ = new CPUFloatTensor(context);
    auto res2 = Tensor(res2_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_syev(res1_->tensor, res2_->tensor, self_->tensor, (eigenvectors) ? "V" : "N", (upper) ? "U" : "L");
    bool maybe_scalar = self_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(res1, res2);
}
std::tuple<Tensor &,Tensor &> CPUFloatType::eig_out(Tensor & res1, Tensor & res2, const Tensor & self, bool eigenvectors) const {
    auto res1_ = checked_cast_tensor<CPUFloatTensor>(res1.pImpl,"res1",0, false);
    auto res2_ = checked_cast_tensor<CPUFloatTensor>(res2.pImpl,"res2",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_geev(res1_->tensor, res2_->tensor, self_->tensor, (eigenvectors) ? "V" : "N");
    bool maybe_scalar = self_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(res1, res2);
}
std::tuple<Tensor,Tensor> CPUFloatType::eig(const Tensor & self, bool eigenvectors) const {
    auto res1_ = new CPUFloatTensor(context);
    auto res1 = Tensor(res1_, false);
    auto res2_ = new CPUFloatTensor(context);
    auto res2 = Tensor(res2_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_geev(res1_->tensor, res2_->tensor, self_->tensor, (eigenvectors) ? "V" : "N");
    bool maybe_scalar = self_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(res1, res2);
}
std::tuple<Tensor &,Tensor &,Tensor &> CPUFloatType::svd_out(Tensor & res1, Tensor & res2, Tensor & res3, const Tensor & self, bool some) const {
    auto res1_ = checked_cast_tensor<CPUFloatTensor>(res1.pImpl,"res1",0, false);
    auto res2_ = checked_cast_tensor<CPUFloatTensor>(res2.pImpl,"res2",0, false);
    auto res3_ = checked_cast_tensor<CPUFloatTensor>(res3.pImpl,"res3",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_gesvd(res1_->tensor, res2_->tensor, res3_->tensor, self_->tensor, (some) ? "S" : "A");
    bool maybe_scalar = self_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    res3_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(res1, res2, res3);
}
std::tuple<Tensor,Tensor,Tensor> CPUFloatType::svd(const Tensor & self, bool some) const {
    auto res1_ = new CPUFloatTensor(context);
    auto res1 = Tensor(res1_, false);
    auto res2_ = new CPUFloatTensor(context);
    auto res2 = Tensor(res2_, false);
    auto res3_ = new CPUFloatTensor(context);
    auto res3 = Tensor(res3_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_gesvd(res1_->tensor, res2_->tensor, res3_->tensor, self_->tensor, (some) ? "S" : "A");
    bool maybe_scalar = self_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    res3_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(res1, res2, res3);
}
Tensor & CPUFloatType::inverse_out(Tensor & output, const Tensor & self) const {
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_getri(output_->tensor, self_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUFloatType::inverse(const Tensor & self) const {
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_getri(output_->tensor, self_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUFloatType::potrf_out(Tensor & output, const Tensor & self, bool upper) const {
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_potrf(output_->tensor, self_->tensor, (upper) ? "U" : "L");
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUFloatType::potrf(const Tensor & self, bool upper) const {
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_potrf(output_->tensor, self_->tensor, (upper) ? "U" : "L");
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUFloatType::potrs_out(Tensor & result, const Tensor & self, const Tensor & input2, bool upper) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto input2_ = checked_cast_tensor<CPUFloatTensor>(input2.pImpl,"input2",2, false);
    THFloatTensor_potrs(result_->tensor, self_->tensor, input2_->tensor, (upper) ? "U" : "L");
    result_->maybeScalar(self_->isScalar() && input2_->isScalar());
    return result;
}
Tensor CPUFloatType::potrs(const Tensor & self, const Tensor & input2, bool upper) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto input2_ = checked_cast_tensor<CPUFloatTensor>(input2.pImpl,"input2",2, false);
    THFloatTensor_potrs(result_->tensor, self_->tensor, input2_->tensor, (upper) ? "U" : "L");
    result_->maybeScalar(self_->isScalar() && input2_->isScalar());
    return result;
}
Tensor & CPUFloatType::potri_out(Tensor & output, const Tensor & self, bool upper) const {
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_potri(output_->tensor, self_->tensor, (upper) ? "U" : "L");
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUFloatType::potri(const Tensor & self, bool upper) const {
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_potri(output_->tensor, self_->tensor, (upper) ? "U" : "L");
    output_->maybeScalar(self_->isScalar());
    return output;
}
std::tuple<Tensor &,Tensor &> CPUFloatType::pstrf_out(Tensor & res1, Tensor & res2, const Tensor & self, bool upper, Scalar tol) const {
    auto res1_ = checked_cast_tensor<CPUFloatTensor>(res1.pImpl,"res1",0, false);
    auto res2_ = checked_cast_tensor<CPUIntTensor>(res2.pImpl,"res2",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto tol_ = tol.toFloat();
    THFloatTensor_pstrf(res1_->tensor, res2_->tensor, self_->tensor, (upper) ? "U" : "L", tol_);
    res2 -= 1;  // LAPACK returns 1-indexed pivots
    return std::tuple<Tensor &, Tensor &>(res1, res2);
}
std::tuple<Tensor,Tensor> CPUFloatType::pstrf(const Tensor & self, bool upper, Scalar tol) const {
    auto res1_ = new CPUFloatTensor(context);
    auto res1 = Tensor(res1_, false);
    auto res2_ = new CPUIntTensor(context);
    auto res2 = Tensor(res2_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto tol_ = tol.toFloat();
    THFloatTensor_pstrf(res1_->tensor, res2_->tensor, self_->tensor, (upper) ? "U" : "L", tol_);
    res2 -= 1;  // LAPACK returns 1-indexed pivots
    return std::tuple<Tensor, Tensor>(res1, res2);
}
std::tuple<Tensor &,Tensor &> CPUFloatType::qr_out(Tensor & res1, Tensor & res2, const Tensor & self) const {
    auto res1_ = checked_cast_tensor<CPUFloatTensor>(res1.pImpl,"res1",0, false);
    auto res2_ = checked_cast_tensor<CPUFloatTensor>(res2.pImpl,"res2",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_qr(res1_->tensor, res2_->tensor, self_->tensor);
    bool maybe_scalar = self_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(res1, res2);
}
std::tuple<Tensor,Tensor> CPUFloatType::qr(const Tensor & self) const {
    auto res1_ = new CPUFloatTensor(context);
    auto res1 = Tensor(res1_, false);
    auto res2_ = new CPUFloatTensor(context);
    auto res2 = Tensor(res2_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_qr(res1_->tensor, res2_->tensor, self_->tensor);
    bool maybe_scalar = self_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(res1, res2);
}
std::tuple<Tensor &,Tensor &> CPUFloatType::geqrf_out(Tensor & res1, Tensor & res2, const Tensor & self) const {
    auto res1_ = checked_cast_tensor<CPUFloatTensor>(res1.pImpl,"res1",0, false);
    auto res2_ = checked_cast_tensor<CPUFloatTensor>(res2.pImpl,"res2",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_geqrf(res1_->tensor, res2_->tensor, self_->tensor);
    bool maybe_scalar = self_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(res1, res2);
}
std::tuple<Tensor,Tensor> CPUFloatType::geqrf(const Tensor & self) const {
    auto res1_ = new CPUFloatTensor(context);
    auto res1 = Tensor(res1_, false);
    auto res2_ = new CPUFloatTensor(context);
    auto res2 = Tensor(res2_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    THFloatTensor_geqrf(res1_->tensor, res2_->tensor, self_->tensor);
    bool maybe_scalar = self_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(res1, res2);
}
Tensor & CPUFloatType::orgqr_out(Tensor & result, const Tensor & self, const Tensor & input2) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto input2_ = checked_cast_tensor<CPUFloatTensor>(input2.pImpl,"input2",2, false);
    THFloatTensor_orgqr(result_->tensor, self_->tensor, input2_->tensor);
    result_->maybeScalar(self_->isScalar() && input2_->isScalar());
    return result;
}
Tensor CPUFloatType::orgqr(const Tensor & self, const Tensor & input2) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto input2_ = checked_cast_tensor<CPUFloatTensor>(input2.pImpl,"input2",2, false);
    THFloatTensor_orgqr(result_->tensor, self_->tensor, input2_->tensor);
    result_->maybeScalar(self_->isScalar() && input2_->isScalar());
    return result;
}
Tensor & CPUFloatType::ormqr_out(Tensor & result, const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto input2_ = checked_cast_tensor<CPUFloatTensor>(input2.pImpl,"input2",2, false);
    auto input3_ = checked_cast_tensor<CPUFloatTensor>(input3.pImpl,"input3",3, false);
    THFloatTensor_ormqr(result_->tensor, self_->tensor, input2_->tensor, input3_->tensor, (left) ? "L" : "R", (transpose) ? "T" : "N");
    result_->maybeScalar(self_->isScalar() && input2_->isScalar() && input3_->isScalar());
    return result;
}
Tensor CPUFloatType::ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto input2_ = checked_cast_tensor<CPUFloatTensor>(input2.pImpl,"input2",2, false);
    auto input3_ = checked_cast_tensor<CPUFloatTensor>(input3.pImpl,"input3",3, false);
    THFloatTensor_ormqr(result_->tensor, self_->tensor, input2_->tensor, input3_->tensor, (left) ? "L" : "R", (transpose) ? "T" : "N");
    result_->maybeScalar(self_->isScalar() && input2_->isScalar() && input3_->isScalar());
    return result;
}
std::tuple<Tensor &,Tensor &> CPUFloatType::btrifact_out(Tensor & result, Tensor & pivots, const Tensor & self, bool pivot) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto pivots_ = checked_cast_tensor<CPUIntTensor>(pivots.pImpl,"pivots",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    THFloatTensor_btrifact(result_->tensor, pivots_->tensor, NULL, pivot, self_->tensor);
    bool maybe_scalar = self_->isScalar();
    result_->maybeScalar(maybe_scalar);
    pivots_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(result, pivots);
}
std::tuple<Tensor,Tensor> CPUFloatType::btrifact(const Tensor & self, bool pivot) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto pivots_ = new CPUIntTensor(context);
    auto pivots = Tensor(pivots_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    THFloatTensor_btrifact(result_->tensor, pivots_->tensor, NULL, pivot, self_->tensor);
    bool maybe_scalar = self_->isScalar();
    result_->maybeScalar(maybe_scalar);
    pivots_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(result, pivots);
}
std::tuple<Tensor &,Tensor &,Tensor &> CPUFloatType::btrifact_with_info_out(Tensor & result, Tensor & pivots, Tensor & info, const Tensor & self, bool pivot) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto pivots_ = checked_cast_tensor<CPUIntTensor>(pivots.pImpl,"pivots",0, false);
    auto info_ = checked_cast_tensor<CPUIntTensor>(info.pImpl,"info",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    THFloatTensor_btrifact(result_->tensor, pivots_->tensor, info_->tensor, pivot, self_->tensor);
    bool maybe_scalar = self_->isScalar();
    result_->maybeScalar(maybe_scalar);
    pivots_->maybeScalar(maybe_scalar);
    info_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(result, pivots, info);
}
std::tuple<Tensor,Tensor,Tensor> CPUFloatType::btrifact_with_info(const Tensor & self, bool pivot) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto pivots_ = new CPUIntTensor(context);
    auto pivots = Tensor(pivots_, false);
    auto info_ = new CPUIntTensor(context);
    auto info = Tensor(info_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    THFloatTensor_btrifact(result_->tensor, pivots_->tensor, info_->tensor, pivot, self_->tensor);
    bool maybe_scalar = self_->isScalar();
    result_->maybeScalar(maybe_scalar);
    pivots_->maybeScalar(maybe_scalar);
    info_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(result, pivots, info);
}
Tensor & CPUFloatType::btrisolve_out(Tensor & result, const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto LU_data_ = checked_cast_tensor<CPUFloatTensor>(LU_data.pImpl,"LU_data",2, false);
    auto LU_pivots_ = checked_cast_tensor<CPUIntTensor>(LU_pivots.pImpl,"LU_pivots",3, false);
    THFloatTensor_btrisolve(result_->tensor, self_->tensor, LU_data_->tensor, LU_pivots_->tensor);
    result_->maybeScalar(self_->isScalar() && LU_data_->isScalar() && LU_pivots_->isScalar());
    return result;
}
Tensor CPUFloatType::btrisolve(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto LU_data_ = checked_cast_tensor<CPUFloatTensor>(LU_data.pImpl,"LU_data",2, false);
    auto LU_pivots_ = checked_cast_tensor<CPUIntTensor>(LU_pivots.pImpl,"LU_pivots",3, false);
    THFloatTensor_btrisolve(result_->tensor, self_->tensor, LU_data_->tensor, LU_pivots_->tensor);
    result_->maybeScalar(self_->isScalar() && LU_data_->isScalar() && LU_pivots_->isScalar());
    return result;
}
Tensor & CPUFloatType::random_(Tensor & self, int64_t from, int64_t to, Generator * generator) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    THFloatTensor_clampedRandom(self_->tensor, generator_->generator, from, to);
    return self;
}
Tensor & CPUFloatType::random_(Tensor & self, int64_t to, Generator * generator) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    THFloatTensor_cappedRandom(self_->tensor, generator_->generator, to);
    return self;
}
Tensor & CPUFloatType::random_(Tensor & self, Generator * generator) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    THFloatTensor_random(self_->tensor, generator_->generator);
    return self;
}
Tensor & CPUFloatType::multinomial_out(Tensor & result, const Tensor & self, int64_t num_samples, bool replacement, Generator * generator) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    THFloatTensor_multinomial(result_->tensor, generator_->generator, self_->tensor, num_samples, replacement);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUFloatType::multinomial(const Tensor & self, int64_t num_samples, bool replacement, Generator * generator) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    THFloatTensor_multinomial(result_->tensor, generator_->generator, self_->tensor, num_samples, replacement);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::uniform_(Tensor & self, double from, double to, Generator * generator) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    THFloatTensor_uniform(self_->tensor, generator_->generator, from, to);
    return self;
}
Tensor & CPUFloatType::normal_out(Tensor & output, const Tensor & mean, double std, Generator * generator) const {
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",0, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    auto mean_ = checked_cast_tensor<CPUFloatTensor>(mean.pImpl,"mean",2, false);
    THFloatTensor_normal_means(output_->tensor, generator_->generator, mean_->tensor, std);
    output_->maybeScalar(mean_->isScalar());
    return output;
}
Tensor CPUFloatType::normal(const Tensor & mean, double std, Generator * generator) const {
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    auto mean_ = checked_cast_tensor<CPUFloatTensor>(mean.pImpl,"mean",2, false);
    THFloatTensor_normal_means(output_->tensor, generator_->generator, mean_->tensor, std);
    output_->maybeScalar(mean_->isScalar());
    return output;
}
Tensor & CPUFloatType::normal_out(Tensor & output, double mean, const Tensor & std, Generator * generator) const {
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",0, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    auto std_ = checked_cast_tensor<CPUFloatTensor>(std.pImpl,"std",3, false);
    THFloatTensor_normal_stddevs(output_->tensor, generator_->generator, mean, std_->tensor);
    output_->maybeScalar(std_->isScalar());
    return output;
}
Tensor CPUFloatType::normal(double mean, const Tensor & std, Generator * generator) const {
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    auto std_ = checked_cast_tensor<CPUFloatTensor>(std.pImpl,"std",3, false);
    THFloatTensor_normal_stddevs(output_->tensor, generator_->generator, mean, std_->tensor);
    output_->maybeScalar(std_->isScalar());
    return output;
}
Tensor & CPUFloatType::normal_out(Tensor & output, const Tensor & mean, const Tensor & std, Generator * generator) const {
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",0, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    auto mean_ = checked_cast_tensor<CPUFloatTensor>(mean.pImpl,"mean",2, false);
    auto std_ = checked_cast_tensor<CPUFloatTensor>(std.pImpl,"std",3, false);
    THFloatTensor_normal_means_stddevs(output_->tensor, generator_->generator, mean_->tensor, std_->tensor);
    output_->maybeScalar(mean_->isScalar() && std_->isScalar());
    return output;
}
Tensor CPUFloatType::normal(const Tensor & mean, const Tensor & std, Generator * generator) const {
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    auto mean_ = checked_cast_tensor<CPUFloatTensor>(mean.pImpl,"mean",2, false);
    auto std_ = checked_cast_tensor<CPUFloatTensor>(std.pImpl,"std",3, false);
    THFloatTensor_normal_means_stddevs(output_->tensor, generator_->generator, mean_->tensor, std_->tensor);
    output_->maybeScalar(mean_->isScalar() && std_->isScalar());
    return output;
}
Tensor & CPUFloatType::normal_(Tensor & self, double mean, double std, Generator * generator) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    THFloatTensor_normal(self_->tensor, generator_->generator, mean, std);
    return self;
}
Tensor & CPUFloatType::cauchy_(Tensor & self, double median, double sigma, Generator * generator) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    THFloatTensor_cauchy(self_->tensor, generator_->generator, median, sigma);
    return self;
}
Tensor & CPUFloatType::log_normal_(Tensor & self, double mean, double std, Generator * generator) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    THFloatTensor_logNormal(self_->tensor, generator_->generator, mean, std);
    return self;
}
Tensor & CPUFloatType::exponential_(Tensor & self, double lambd, Generator * generator) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    THFloatTensor_exponential(self_->tensor, generator_->generator, lambd);
    return self;
}
Tensor & CPUFloatType::geometric_(Tensor & self, double p, Generator * generator) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    THFloatTensor_geometric(self_->tensor, generator_->generator, p);
    return self;
}
Tensor & CPUFloatType::bernoulli_out(Tensor & output, const Tensor & self, Generator * generator) const {
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",0, false);
    output.resize_(self.sizes());
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    THFloatTensor_bernoulli_Tensor(output_->tensor, generator_->generator, self_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUFloatType::bernoulli(const Tensor & self, Generator * generator) const {
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    output.resize_(self.sizes());
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    THFloatTensor_bernoulli_Tensor(output_->tensor, generator_->generator, self_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUFloatType::_standard_gamma_out(Tensor & output, const Tensor & self, Generator * generator) const {
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",0, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    THFloatTensor_standard_gamma(output_->tensor, generator_->generator, self_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUFloatType::_standard_gamma(const Tensor & self, Generator * generator) const {
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    THFloatTensor_standard_gamma(output_->tensor, generator_->generator, self_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUFloatType::_dirichlet_grad_out(Tensor & output, const Tensor & x, const Tensor & alpha, const Tensor & total) const {
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",0, false);
    auto x_ = checked_cast_tensor<CPUFloatTensor>(x.pImpl,"x",1, false);
    auto alpha_ = checked_cast_tensor<CPUFloatTensor>(alpha.pImpl,"alpha",2, false);
    auto total_ = checked_cast_tensor<CPUFloatTensor>(total.pImpl,"total",3, false);
    THFloatTensor_dirichlet_grad(output_->tensor, x_->tensor, alpha_->tensor, total_->tensor);
    output_->maybeScalar(x_->isScalar() && alpha_->isScalar() && total_->isScalar());
    return output;
}
Tensor CPUFloatType::_dirichlet_grad(const Tensor & x, const Tensor & alpha, const Tensor & total) const {
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    auto x_ = checked_cast_tensor<CPUFloatTensor>(x.pImpl,"x",1, false);
    auto alpha_ = checked_cast_tensor<CPUFloatTensor>(alpha.pImpl,"alpha",2, false);
    auto total_ = checked_cast_tensor<CPUFloatTensor>(total.pImpl,"total",3, false);
    THFloatTensor_dirichlet_grad(output_->tensor, x_->tensor, alpha_->tensor, total_->tensor);
    output_->maybeScalar(x_->isScalar() && alpha_->isScalar() && total_->isScalar());
    return output;
}
Tensor CPUFloatType::tensor(Storage & storage, int64_t storageOffset, IntList size, IntList stride) const {
    auto storage_ = checked_cast_storage<CPUFloatStorage>(&storage,"storage",1);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    return Tensor((new CPUFloatTensor(context, THFloatTensor_newWithStorage(storage_->storage, storageOffset, size_, stride_)))->maybeScalar(size.size() == 0),false);
}
Tensor CPUFloatType::tensor(IntList size) const {
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new CPUFloatTensor(context, THFloatTensor_newWithSize(size_, NULL)))->maybeScalar(size.size() == 0),false);
}
Tensor CPUFloatType::tensor(IntList size, IntList stride) const {
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    return Tensor((new CPUFloatTensor(context, THFloatTensor_newWithSize(size_, stride_)))->maybeScalar(size.size() == 0),false);
}
Tensor CPUFloatType::tensor() const {
    return Tensor((new CPUFloatTensor(context, THFloatTensor_new())),false);
}
Tensor CPUFloatType::alias(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPUFloatTensor(context, THFloatTensor_newWithTensor(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor & CPUFloatType::as_strided_out(Tensor & result, const Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    auto result_ = checked_cast_tensor<CPUFloatTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    if (storage_offset == -1) {
      storage_offset = self_->tensor->storageOffset;
    }
    THFloatTensor_setStorage(result_->tensor, self_->tensor->storage, storage_offset, size_, stride_);
    result_->maybeScalar(size.size() == 0);
    return result;
}
Tensor CPUFloatType::as_strided(const Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    auto result_ = new CPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    if (storage_offset == -1) {
      storage_offset = self_->tensor->storageOffset;
    }
    THFloatTensor_setStorage(result_->tensor, self_->tensor->storage, storage_offset, size_, stride_);
    result_->maybeScalar(size.size() == 0);
    return result;
}
Tensor & CPUFloatType::as_strided_(Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    if (storage_offset == -1) {
      storage_offset = self_->tensor->storageOffset;
    }
    THFloatTensor_setStorage(self_->tensor, self_->tensor->storage, storage_offset, size_, stride_);
    self_->maybeScalar(size.size() == 0);
    return self;
}
Tensor & CPUFloatType::_cat_out(Tensor & self, TensorList tensors, int64_t dim) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",0, false);
    auto tensors_ = tensor_list_checked_cast<CPUFloatTensor, Tensor, THFloatTensor>(tensors,"tensors",1);
    THFloatTensor_catArray(self_->tensor, tensors_.data(), tensors_.size(), dim);
    return self;
}
Tensor CPUFloatType::_cat(TensorList tensors, int64_t dim) const {
    auto self_ = new CPUFloatTensor(context);
    auto self = Tensor(self_, false);
    auto tensors_ = tensor_list_checked_cast<CPUFloatTensor, Tensor, THFloatTensor>(tensors,"tensors",1);
    THFloatTensor_catArray(self_->tensor, tensors_.data(), tensors_.size(), dim);
    return self;
}
Tensor CPUFloatType::_sparse_mask(const Tensor & self, SparseTensor mask) const {
    auto result_ = new SparseCPUFloatTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<SparseCPUFloatTensor>(mask.tref.pImpl,"mask",2,false);
    THFloatTensor_sparseMask(result_->tensor, self_->tensor, mask_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUFloatType::binary_cross_entropy_forward_out(Tensor & output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CPUFloatTensor>(target.pImpl,"target",2, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",3, true);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",5, false);
    THNN_FloatBCECriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, weight_ ? weight_->tensor : NULL, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor CPUFloatType::binary_cross_entropy_forward(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CPUFloatTensor>(target.pImpl,"target",2, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",3, true);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_FloatBCECriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, weight_ ? weight_->tensor : NULL, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor & CPUFloatType::binary_cross_entropy_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CPUFloatTensor>(target.pImpl,"target",3, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",4, true);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",6, false);
    THNN_FloatBCECriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, weight_ ? weight_->tensor : NULL, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::binary_cross_entropy_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CPUFloatTensor>(target.pImpl,"target",3, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",4, true);
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatBCECriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, weight_ ? weight_->tensor : NULL, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUFloatType::kl_div_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CPUFloatTensor>(target.pImpl,"target",2, false);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",4, false);
    THNN_FloatDistKLDivCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor CPUFloatType::kl_div_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CPUFloatTensor>(target.pImpl,"target",2, false);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_FloatDistKLDivCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor & CPUFloatType::kl_div_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CPUFloatTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",5, false);
    THNN_FloatDistKLDivCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::kl_div_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CPUFloatTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatDistKLDivCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUFloatType::l1_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CPUFloatTensor>(target.pImpl,"target",2, false);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",4, false);
    THNN_FloatAbsCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor CPUFloatType::l1_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CPUFloatTensor>(target.pImpl,"target",2, false);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_FloatAbsCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor & CPUFloatType::l1_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CPUFloatTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",5, false);
    THNN_FloatAbsCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CPUFloatTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatAbsCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUFloatType::mse_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CPUFloatTensor>(target.pImpl,"target",2, false);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",4, false);
    THNN_FloatMSECriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor CPUFloatType::mse_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CPUFloatTensor>(target.pImpl,"target",2, false);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_FloatMSECriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor & CPUFloatType::mse_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CPUFloatTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",5, false);
    THNN_FloatMSECriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::mse_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CPUFloatTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatMSECriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUFloatType::multi_margin_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CPULongTensor>(target.pImpl,"target",2, false);
    auto p_ = p.toDouble();
    auto margin_ = margin.toDouble();
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",5, true);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",7, false);
    THNN_FloatMultiMarginCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, p_, weight_ ? weight_->tensor : NULL, margin_, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor CPUFloatType::multi_margin_loss_forward(const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CPULongTensor>(target.pImpl,"target",2, false);
    auto p_ = p.toDouble();
    auto margin_ = margin.toDouble();
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",5, true);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_FloatMultiMarginCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, p_, weight_ ? weight_->tensor : NULL, margin_, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor & CPUFloatType::multi_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CPULongTensor>(target.pImpl,"target",3, false);
    auto p_ = p.toDouble();
    auto margin_ = margin.toDouble();
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",6, true);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",8, false);
    THNN_FloatMultiMarginCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, p_, weight_ ? weight_->tensor : NULL, margin_, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::multi_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CPULongTensor>(target.pImpl,"target",3, false);
    auto p_ = p.toDouble();
    auto margin_ = margin.toDouble();
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",6, true);
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatMultiMarginCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, p_, weight_ ? weight_->tensor : NULL, margin_, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &> CPUFloatType::multilabel_margin_loss_forward_out(Tensor & output, Tensor & is_target, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CPULongTensor>(target.pImpl,"target",2, false);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",4, false);
    auto is_target_ = checked_cast_tensor<CPUFloatTensor>(is_target.pImpl,"is_target",4, false);
    THNN_FloatMultiLabelMarginCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, is_target_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    is_target_->maybeScalar(target_->isScalar());
    return std::tuple<Tensor &, Tensor &>(output, is_target);
}
std::tuple<Tensor,Tensor> CPUFloatType::multilabel_margin_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CPULongTensor>(target.pImpl,"target",2, false);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    auto is_target_ = new CPUFloatTensor(context);
    auto is_target = Tensor(is_target_, false);
    THNN_FloatMultiLabelMarginCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, is_target_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    is_target_->maybeScalar(target_->isScalar());
    return std::tuple<Tensor, Tensor>(output, is_target);
}
Tensor & CPUFloatType::multilabel_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce, const Tensor & is_target) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CPULongTensor>(target.pImpl,"target",3, false);
    auto is_target_ = checked_cast_tensor<CPUFloatTensor>(is_target.pImpl,"is_target",6, false);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",6, false);
    THNN_FloatMultiLabelMarginCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, is_target_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::multilabel_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce, const Tensor & is_target) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CPULongTensor>(target.pImpl,"target",3, false);
    auto is_target_ = checked_cast_tensor<CPUFloatTensor>(is_target.pImpl,"is_target",6, false);
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatMultiLabelMarginCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, is_target_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &> CPUFloatType::nll_loss_forward_out(Tensor & output, Tensor & total_weight, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CPULongTensor>(target.pImpl,"target",2, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",3, true);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",6, false);
    auto total_weight_ = checked_cast_tensor<CPUFloatTensor>(total_weight.pImpl,"total_weight",6, false);
    THNN_FloatClassNLLCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, weight_ ? weight_->tensor : NULL, total_weight_->tensor, ignore_index, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    total_weight_->maybeScalar(true);
    return std::tuple<Tensor &, Tensor &>(output, total_weight);
}
std::tuple<Tensor,Tensor> CPUFloatType::nll_loss_forward(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CPULongTensor>(target.pImpl,"target",2, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",3, true);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    auto total_weight_ = new CPUFloatTensor(context);
    auto total_weight = Tensor(total_weight_, false);
    THNN_FloatClassNLLCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, weight_ ? weight_->tensor : NULL, total_weight_->tensor, ignore_index, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    total_weight_->maybeScalar(true);
    return std::tuple<Tensor, Tensor>(output, total_weight);
}
Tensor & CPUFloatType::nll_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CPULongTensor>(target.pImpl,"target",3, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",4, true);
    auto total_weight_ = checked_cast_tensor<CPUFloatTensor>(total_weight.pImpl,"total_weight",8, false);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",8, false);
    THNN_FloatClassNLLCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, weight_ ? weight_->tensor : NULL, total_weight_->tensor, ignore_index, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::nll_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CPULongTensor>(target.pImpl,"target",3, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",4, true);
    auto total_weight_ = checked_cast_tensor<CPUFloatTensor>(total_weight.pImpl,"total_weight",8, false);
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatClassNLLCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, weight_ ? weight_->tensor : NULL, total_weight_->tensor, ignore_index, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &> CPUFloatType::nll_loss2d_forward_out(Tensor & output, Tensor & total_weight, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CPULongTensor>(target.pImpl,"target",2, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",3, true);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",6, false);
    auto total_weight_ = checked_cast_tensor<CPUFloatTensor>(total_weight.pImpl,"total_weight",6, false);
    THNN_FloatSpatialClassNLLCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, weight_ ? weight_->tensor : NULL, total_weight_->tensor, ignore_index, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    total_weight_->maybeScalar(true);
    return std::tuple<Tensor &, Tensor &>(output, total_weight);
}
std::tuple<Tensor,Tensor> CPUFloatType::nll_loss2d_forward(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CPULongTensor>(target.pImpl,"target",2, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",3, true);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    auto total_weight_ = new CPUFloatTensor(context);
    auto total_weight = Tensor(total_weight_, false);
    THNN_FloatSpatialClassNLLCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, weight_ ? weight_->tensor : NULL, total_weight_->tensor, ignore_index, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    total_weight_->maybeScalar(true);
    return std::tuple<Tensor, Tensor>(output, total_weight);
}
Tensor & CPUFloatType::nll_loss2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CPULongTensor>(target.pImpl,"target",3, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",4, true);
    auto total_weight_ = checked_cast_tensor<CPUFloatTensor>(total_weight.pImpl,"total_weight",8, false);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",8, false);
    THNN_FloatSpatialClassNLLCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, weight_ ? weight_->tensor : NULL, total_weight_->tensor, ignore_index, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::nll_loss2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CPULongTensor>(target.pImpl,"target",3, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",4, true);
    auto total_weight_ = checked_cast_tensor<CPUFloatTensor>(total_weight.pImpl,"total_weight",8, false);
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatSpatialClassNLLCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, weight_ ? weight_->tensor : NULL, total_weight_->tensor, ignore_index, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUFloatType::smooth_l1_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CPUFloatTensor>(target.pImpl,"target",2, false);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",4, false);
    THNN_FloatSmoothL1Criterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor CPUFloatType::smooth_l1_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CPUFloatTensor>(target.pImpl,"target",2, false);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_FloatSmoothL1Criterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor & CPUFloatType::smooth_l1_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CPUFloatTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",5, false);
    THNN_FloatSmoothL1Criterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::smooth_l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CPUFloatTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatSmoothL1Criterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUFloatType::soft_margin_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CPUFloatTensor>(target.pImpl,"target",2, false);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",4, false);
    THNN_FloatSoftMarginCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor CPUFloatType::soft_margin_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CPUFloatTensor>(target.pImpl,"target",2, false);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_FloatSoftMarginCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor & CPUFloatType::soft_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CPUFloatTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",5, false);
    THNN_FloatSoftMarginCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::soft_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CPUFloatTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatSoftMarginCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUFloatType::elu_forward_out(Tensor & output, const Tensor & self, Scalar alpha, Scalar scale) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toDouble();
    auto scale_ = scale.toDouble();
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",3, false);
    THNN_FloatELU_updateOutput(context->thc_state, self_->tensor, output_->tensor, alpha_, scale_, false);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUFloatType::elu_forward(const Tensor & self, Scalar alpha, Scalar scale) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toDouble();
    auto scale_ = scale.toDouble();
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_FloatELU_updateOutput(context->thc_state, self_->tensor, output_->tensor, alpha_, scale_, false);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUFloatType::elu_backward_out(Tensor & grad_input, const Tensor & grad_output, Scalar alpha, Scalar scale, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto alpha_ = alpha.toDouble();
    auto scale_ = scale.toDouble();
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",4, false);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_FloatELU_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, output_->tensor, alpha_, scale_);
    grad_input_->maybeScalar(output_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::elu_backward(const Tensor & grad_output, Scalar alpha, Scalar scale, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto alpha_ = alpha.toDouble();
    auto scale_ = scale.toDouble();
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",4, false);
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatELU_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, output_->tensor, alpha_, scale_);
    grad_input_->maybeScalar(output_->isScalar());
    return grad_input;
}
Tensor & CPUFloatType::elu_forward_(Tensor & self, Scalar alpha, Scalar scale) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toDouble();
    auto scale_ = scale.toDouble();
    THNN_FloatELU_updateOutput(context->thc_state, self_->tensor, self_->tensor, alpha_, scale_, true);
    return self;
}
Tensor & CPUFloatType::glu_forward_out(Tensor & output, const Tensor & self, int64_t dim) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",2, false);
    THNN_FloatGatedLinear_updateOutput(context->thc_state, self_->tensor, output_->tensor, dim);
    output_->maybeScalar(false);
    return output;
}
Tensor CPUFloatType::glu_forward(const Tensor & self, int64_t dim) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_FloatGatedLinear_updateOutput(context->thc_state, self_->tensor, output_->tensor, dim);
    output_->maybeScalar(false);
    return output;
}
Tensor & CPUFloatType::glu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    dim = maybe_wrap_dim(dim, self_);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_FloatGatedLinear_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, dim);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::glu_backward(const Tensor & grad_output, const Tensor & self, int64_t dim) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    dim = maybe_wrap_dim(dim, self_);
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatGatedLinear_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, dim);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUFloatType::hardshrink_forward_out(Tensor & output, const Tensor & self, Scalar lambd) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto lambd_ = lambd.toDouble();
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",2, false);
    THNN_FloatHardShrink_updateOutput(context->thc_state, self_->tensor, output_->tensor, lambd_);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUFloatType::hardshrink_forward(const Tensor & self, Scalar lambd) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto lambd_ = lambd.toDouble();
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_FloatHardShrink_updateOutput(context->thc_state, self_->tensor, output_->tensor, lambd_);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUFloatType::hardshrink_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar lambd) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto lambd_ = lambd.toDouble();
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_FloatHardShrink_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, lambd_);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::hardshrink_backward(const Tensor & grad_output, const Tensor & self, Scalar lambd) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto lambd_ = lambd.toDouble();
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatHardShrink_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, lambd_);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUFloatType::hardtanh_forward_out(Tensor & output, const Tensor & self, Scalar min_val, Scalar max_val) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto min_val_ = min_val.toDouble();
    auto max_val_ = max_val.toDouble();
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",3, false);
    THNN_FloatHardTanh_updateOutput(context->thc_state, self_->tensor, output_->tensor, min_val_, max_val_, false);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUFloatType::hardtanh_forward(const Tensor & self, Scalar min_val, Scalar max_val) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto min_val_ = min_val.toDouble();
    auto max_val_ = max_val.toDouble();
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_FloatHardTanh_updateOutput(context->thc_state, self_->tensor, output_->tensor, min_val_, max_val_, false);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUFloatType::hardtanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto min_val_ = min_val.toDouble();
    auto max_val_ = max_val.toDouble();
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_FloatHardTanh_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, min_val_, max_val_, false);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::hardtanh_backward(const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto min_val_ = min_val.toDouble();
    auto max_val_ = max_val.toDouble();
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatHardTanh_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, min_val_, max_val_, false);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUFloatType::hardtanh_forward_(Tensor & self, Scalar min_val, Scalar max_val) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto min_val_ = min_val.toDouble();
    auto max_val_ = max_val.toDouble();
    THNN_FloatHardTanh_updateOutput(context->thc_state, self_->tensor, self_->tensor, min_val_, max_val_, true);
    return self;
}
Tensor & CPUFloatType::leaky_relu_forward_out(Tensor & output, const Tensor & self, Scalar negative_slope) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto negative_slope_ = negative_slope.toDouble();
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",2, false);
    THNN_FloatLeakyReLU_updateOutput(context->thc_state, self_->tensor, output_->tensor, negative_slope_, false);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUFloatType::leaky_relu_forward(const Tensor & self, Scalar negative_slope) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto negative_slope_ = negative_slope.toDouble();
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_FloatLeakyReLU_updateOutput(context->thc_state, self_->tensor, output_->tensor, negative_slope_, false);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUFloatType::leaky_relu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar negative_slope) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto negative_slope_ = negative_slope.toDouble();
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_FloatLeakyReLU_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, negative_slope_, false);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::leaky_relu_backward(const Tensor & grad_output, const Tensor & self, Scalar negative_slope) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto negative_slope_ = negative_slope.toDouble();
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatLeakyReLU_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, negative_slope_, false);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUFloatType::leaky_relu_forward_(Tensor & self, Scalar negative_slope) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto negative_slope_ = negative_slope.toDouble();
    THNN_FloatLeakyReLU_updateOutput(context->thc_state, self_->tensor, self_->tensor, negative_slope_, true);
    return self;
}
std::tuple<Tensor &,Tensor &> CPUFloatType::log_sigmoid_forward_out(Tensor & output, Tensor & buffer, const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",1, false);
    auto buffer_ = checked_cast_tensor<CPUFloatTensor>(buffer.pImpl,"buffer",1, false);
    THNN_FloatLogSigmoid_updateOutput(context->thc_state, self_->tensor, output_->tensor, buffer_->tensor);
    output_->maybeScalar(self_->isScalar());
    buffer_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &>(output, buffer);
}
std::tuple<Tensor,Tensor> CPUFloatType::log_sigmoid_forward(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    auto buffer_ = new CPUFloatTensor(context);
    auto buffer = Tensor(buffer_, false);
    THNN_FloatLogSigmoid_updateOutput(context->thc_state, self_->tensor, output_->tensor, buffer_->tensor);
    output_->maybeScalar(self_->isScalar());
    buffer_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor>(output, buffer);
}
Tensor & CPUFloatType::log_sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & buffer) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto buffer_ = checked_cast_tensor<CPUFloatTensor>(buffer.pImpl,"buffer",3, false);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_FloatLogSigmoid_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, buffer_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::log_sigmoid_backward(const Tensor & grad_output, const Tensor & self, const Tensor & buffer) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto buffer_ = checked_cast_tensor<CPUFloatTensor>(buffer.pImpl,"buffer",3, false);
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatLogSigmoid_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, buffer_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUFloatType::log_softmax_forward_out(Tensor & output, const Tensor & self, int64_t dim) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",2, false);
    THNN_FloatLogSoftMax_updateOutput(context->thc_state, self_->tensor, output_->tensor, dim);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUFloatType::log_softmax_forward(const Tensor & self, int64_t dim) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_FloatLogSoftMax_updateOutput(context->thc_state, self_->tensor, output_->tensor, dim);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUFloatType::log_softmax_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",4, false);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_FloatLogSoftMax_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, output_->tensor, dim);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::log_softmax_backward(const Tensor & grad_output, const Tensor & self, int64_t dim, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",4, false);
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatLogSoftMax_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, output_->tensor, dim);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUFloatType::prelu_forward_out(Tensor & output, const Tensor & self, const Tensor & weight) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",2, false);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",2, false);
    THNN_FloatPReLU_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUFloatType::prelu_forward(const Tensor & self, const Tensor & weight) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",2, false);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_FloatPReLU_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
std::tuple<Tensor &,Tensor &> CPUFloatType::prelu_backward_out(Tensor & grad_input, Tensor & grad_weight, const Tensor & grad_output, const Tensor & self, const Tensor & weight) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",3, false);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",3, true);
    auto grad_weight_ = checked_cast_tensor<CPUFloatTensor>(grad_weight.pImpl,"grad_weight",3, true);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    if (grad_input_) THNN_FloatPReLU_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor);
    if (grad_weight_) THNN_FloatPReLU_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &>(grad_input, grad_weight);
}
std::tuple<Tensor,Tensor> CPUFloatType::prelu_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, std::array<bool,2> output_mask) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",3, false);
    auto grad_input_ = output_mask[0] ? new CPUFloatTensor(context) : nullptr;
    auto grad_input = Tensor(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_input_, false);
    auto grad_weight_ = output_mask[1] ? new CPUFloatTensor(context) : nullptr;
    auto grad_weight = Tensor(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_weight_, false);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    if (grad_input_) THNN_FloatPReLU_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor);
    if (grad_weight_) THNN_FloatPReLU_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor>(grad_input, grad_weight);
}
Tensor & CPUFloatType::rrelu_with_noise_forward_out(Tensor & output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto noise_ = checked_cast_tensor<CPUFloatTensor>(noise.pImpl,"noise",2, false);
    auto lower_ = lower.toDouble();
    auto upper_ = upper.toDouble();
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",6, false);
    THNN_FloatRReLU_updateOutput(context->thc_state, self_->tensor, output_->tensor, noise_->tensor, lower_, upper_, training, false, generator_->generator);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUFloatType::rrelu_with_noise_forward(const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto noise_ = checked_cast_tensor<CPUFloatTensor>(noise.pImpl,"noise",2, false);
    auto lower_ = lower.toDouble();
    auto upper_ = upper.toDouble();
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_FloatRReLU_updateOutput(context->thc_state, self_->tensor, output_->tensor, noise_->tensor, lower_, upper_, training, false, generator_->generator);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUFloatType::rrelu_with_noise_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto noise_ = checked_cast_tensor<CPUFloatTensor>(noise.pImpl,"noise",3, false);
    auto lower_ = lower.toDouble();
    auto upper_ = upper.toDouble();
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",6, false);
    THNN_FloatRReLU_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, noise_->tensor, lower_, upper_, training, false);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::rrelu_with_noise_backward(const Tensor & grad_output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto noise_ = checked_cast_tensor<CPUFloatTensor>(noise.pImpl,"noise",3, false);
    auto lower_ = lower.toDouble();
    auto upper_ = upper.toDouble();
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatRReLU_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, noise_->tensor, lower_, upper_, training, false);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUFloatType::rrelu_with_noise_forward_(Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto noise_ = checked_cast_tensor<CPUFloatTensor>(noise.pImpl,"noise",2, false);
    auto lower_ = lower.toDouble();
    auto upper_ = upper.toDouble();
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    THNN_FloatRReLU_updateOutput(context->thc_state, self_->tensor, self_->tensor, noise_->tensor, lower_, upper_, training, true, generator_->generator);
    return self;
}
Tensor & CPUFloatType::softmax_forward_out(Tensor & output, const Tensor & self, int64_t dim) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",2, false);
    THNN_FloatSoftMax_updateOutput(context->thc_state, self_->tensor, output_->tensor, dim);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUFloatType::softmax_forward(const Tensor & self, int64_t dim) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_FloatSoftMax_updateOutput(context->thc_state, self_->tensor, output_->tensor, dim);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUFloatType::softmax_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",4, false);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_FloatSoftMax_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, output_->tensor, dim);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::softmax_backward(const Tensor & grad_output, const Tensor & self, int64_t dim, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",4, false);
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatSoftMax_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, output_->tensor, dim);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUFloatType::softplus_forward_out(Tensor & output, const Tensor & self, Scalar beta, Scalar threshold) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toDouble();
    auto threshold_ = threshold.toDouble();
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",3, false);
    THNN_FloatSoftPlus_updateOutput(context->thc_state, self_->tensor, output_->tensor, beta_, threshold_);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUFloatType::softplus_forward(const Tensor & self, Scalar beta, Scalar threshold) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toDouble();
    auto threshold_ = threshold.toDouble();
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_FloatSoftPlus_updateOutput(context->thc_state, self_->tensor, output_->tensor, beta_, threshold_);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUFloatType::softplus_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto beta_ = beta.toDouble();
    auto threshold_ = threshold.toDouble();
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",5, false);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",5, false);
    THNN_FloatSoftPlus_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, output_->tensor, beta_, threshold_);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::softplus_backward(const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto beta_ = beta.toDouble();
    auto threshold_ = threshold.toDouble();
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",5, false);
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatSoftPlus_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, output_->tensor, beta_, threshold_);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUFloatType::softshrink_forward_out(Tensor & output, const Tensor & self, Scalar lambd) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto lambd_ = lambd.toDouble();
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",2, false);
    THNN_FloatSoftShrink_updateOutput(context->thc_state, self_->tensor, output_->tensor, lambd_);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUFloatType::softshrink_forward(const Tensor & self, Scalar lambd) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto lambd_ = lambd.toDouble();
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_FloatSoftShrink_updateOutput(context->thc_state, self_->tensor, output_->tensor, lambd_);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUFloatType::softshrink_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar lambd) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto lambd_ = lambd.toDouble();
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_FloatSoftShrink_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, lambd_);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::softshrink_backward(const Tensor & grad_output, const Tensor & self, Scalar lambd) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto lambd_ = lambd.toDouble();
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatSoftShrink_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, lambd_);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUFloatType::threshold_forward_out(Tensor & output, const Tensor & self, Scalar threshold, Scalar value) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto threshold_ = threshold.toDouble();
    auto value_ = value.toDouble();
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",3, false);
    THNN_FloatThreshold_updateOutput(context->thc_state, self_->tensor, output_->tensor, threshold_, value_, false);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUFloatType::threshold_forward(const Tensor & self, Scalar threshold, Scalar value) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto threshold_ = threshold.toDouble();
    auto value_ = value.toDouble();
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_FloatThreshold_updateOutput(context->thc_state, self_->tensor, output_->tensor, threshold_, value_, false);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUFloatType::threshold_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar threshold, Scalar value) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto threshold_ = threshold.toDouble();
    auto value_ = value.toDouble();
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_FloatThreshold_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, threshold_, value_, false);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::threshold_backward(const Tensor & grad_output, const Tensor & self, Scalar threshold, Scalar value) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto threshold_ = threshold.toDouble();
    auto value_ = value.toDouble();
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatThreshold_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, threshold_, value_, false);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUFloatType::threshold_forward_(Tensor & self, Scalar threshold, Scalar value) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto threshold_ = threshold.toDouble();
    auto value_ = value.toDouble();
    THNN_FloatThreshold_updateOutput(context->thc_state, self_->tensor, self_->tensor, threshold_, value_, true);
    return self;
}
Tensor & CPUFloatType::adaptive_avg_pool2d_forward_out(Tensor & output, const Tensor & self, IntList output_size) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 2);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",2, false);
    THNN_FloatSpatialAdaptiveAveragePooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[1], output_size_[0]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUFloatType::adaptive_avg_pool2d_forward(const Tensor & self, IntList output_size) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 2);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_FloatSpatialAdaptiveAveragePooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[1], output_size_[0]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUFloatType::adaptive_avg_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",2, false);
    THNN_FloatSpatialAdaptiveAveragePooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::adaptive_avg_pool2d_backward(const Tensor & grad_output, const Tensor & self) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatSpatialAdaptiveAveragePooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUFloatType::adaptive_avg_pool3d_forward_out(Tensor & output, const Tensor & self, IntList output_size) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 2);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",2, false);
    THNN_FloatVolumetricAdaptiveAveragePooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[0], output_size_[2], output_size_[1]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUFloatType::adaptive_avg_pool3d_forward(const Tensor & self, IntList output_size) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 2);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_FloatVolumetricAdaptiveAveragePooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[0], output_size_[2], output_size_[1]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUFloatType::adaptive_avg_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",2, false);
    THNN_FloatVolumetricAdaptiveAveragePooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::adaptive_avg_pool3d_backward(const Tensor & grad_output, const Tensor & self) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatVolumetricAdaptiveAveragePooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &> CPUFloatType::adaptive_max_pool2d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList output_size) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 2);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",2, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",2, false);
    THNN_FloatSpatialAdaptiveMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, output_size_[1], output_size_[0]);
    bool maybe_scalar = self_->isScalar();
    output_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(output, indices);
}
std::tuple<Tensor,Tensor> CPUFloatType::adaptive_max_pool2d_forward(const Tensor & self, IntList output_size) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 2);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    auto indices_ = new CPULongTensor(context);
    auto indices = Tensor(indices_, false);
    THNN_FloatSpatialAdaptiveMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, output_size_[1], output_size_[0]);
    bool maybe_scalar = self_->isScalar();
    output_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(output, indices);
}
Tensor & CPUFloatType::adaptive_max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",3, false);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_FloatSpatialAdaptiveMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::adaptive_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",3, false);
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatSpatialAdaptiveMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &> CPUFloatType::adaptive_max_pool3d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList output_size) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 2);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",2, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",2, false);
    THNN_FloatVolumetricAdaptiveMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, output_size_[0], output_size_[2], output_size_[1]);
    bool maybe_scalar = self_->isScalar();
    output_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(output, indices);
}
std::tuple<Tensor,Tensor> CPUFloatType::adaptive_max_pool3d_forward(const Tensor & self, IntList output_size) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 2);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    auto indices_ = new CPULongTensor(context);
    auto indices = Tensor(indices_, false);
    THNN_FloatVolumetricAdaptiveMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, output_size_[0], output_size_[2], output_size_[1]);
    bool maybe_scalar = self_->isScalar();
    output_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(output, indices);
}
Tensor & CPUFloatType::adaptive_max_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",3, false);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_FloatVolumetricAdaptiveMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::adaptive_max_pool3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",3, false);
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatVolumetricAdaptiveMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUFloatType::avg_pool2d_forward_out(Tensor & output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 2);
    auto stride_ = check_intlist<2>(stride, "stride", 3, kernel_size);
    auto padding_ = check_intlist<2>(padding, "padding", 4);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",6, false);
    THNN_FloatSpatialAveragePooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], ceil_mode, count_include_pad);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUFloatType::avg_pool2d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 2);
    auto stride_ = check_intlist<2>(stride, "stride", 3, kernel_size);
    auto padding_ = check_intlist<2>(padding, "padding", 4);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_FloatSpatialAveragePooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], ceil_mode, count_include_pad);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUFloatType::avg_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto stride_ = check_intlist<2>(stride, "stride", 4, kernel_size);
    auto padding_ = check_intlist<2>(padding, "padding", 5);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",7, false);
    THNN_FloatSpatialAveragePooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], ceil_mode, count_include_pad);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::avg_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto stride_ = check_intlist<2>(stride, "stride", 4, kernel_size);
    auto padding_ = check_intlist<2>(padding, "padding", 5);
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatSpatialAveragePooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], ceil_mode, count_include_pad);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUFloatType::avg_pool3d_forward_out(Tensor & output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 2);
    auto stride_ = check_intlist<3>(stride, "stride", 3, kernel_size);
    auto padding_ = check_intlist<3>(padding, "padding", 4);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",6, false);
    THNN_FloatVolumetricAveragePooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], ceil_mode, count_include_pad);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUFloatType::avg_pool3d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 2);
    auto stride_ = check_intlist<3>(stride, "stride", 3, kernel_size);
    auto padding_ = check_intlist<3>(padding, "padding", 4);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_FloatVolumetricAveragePooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], ceil_mode, count_include_pad);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUFloatType::avg_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto stride_ = check_intlist<3>(stride, "stride", 4, kernel_size);
    auto padding_ = check_intlist<3>(padding, "padding", 5);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",7, false);
    THNN_FloatVolumetricAveragePooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], ceil_mode, count_include_pad);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::avg_pool3d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto stride_ = check_intlist<3>(stride, "stride", 4, kernel_size);
    auto padding_ = check_intlist<3>(padding, "padding", 5);
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatVolumetricAveragePooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], ceil_mode, count_include_pad);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &> CPUFloatType::fractional_max_pool2d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & random_samples) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 2);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 3);
    auto random_samples_ = checked_cast_tensor<CPUFloatTensor>(random_samples.pImpl,"random_samples",4, false);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",4, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",4, false);
    THNN_FloatSpatialFractionalMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[1], output_size_[0], kernel_size_[1], kernel_size_[0], indices_->tensor, random_samples_->tensor);
    output_->maybeScalar(false);
    return std::tuple<Tensor &, Tensor &>(output, indices);
}
std::tuple<Tensor,Tensor> CPUFloatType::fractional_max_pool2d_forward(const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & random_samples) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 2);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 3);
    auto random_samples_ = checked_cast_tensor<CPUFloatTensor>(random_samples.pImpl,"random_samples",4, false);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    auto indices_ = new CPULongTensor(context);
    auto indices = Tensor(indices_, false);
    THNN_FloatSpatialFractionalMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[1], output_size_[0], kernel_size_[1], kernel_size_[0], indices_->tensor, random_samples_->tensor);
    output_->maybeScalar(false);
    return std::tuple<Tensor, Tensor>(output, indices);
}
Tensor & CPUFloatType::fractional_max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 4);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",5, false);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",5, false);
    THNN_FloatSpatialFractionalMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, output_size_[1], output_size_[0], kernel_size_[1], kernel_size_[0], indices_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::fractional_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 4);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",5, false);
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatSpatialFractionalMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, output_size_[1], output_size_[0], kernel_size_[1], kernel_size_[0], indices_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &> CPUFloatType::max_pool2d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 2);
    auto stride_ = check_intlist<2>(stride, "stride", 3, kernel_size);
    auto padding_ = check_intlist<2>(padding, "padding", 4);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 5);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",6, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",6, false);
    THNN_FloatSpatialDilatedMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], ceil_mode);
    bool maybe_scalar = self_->isScalar();
    output_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(output, indices);
}
std::tuple<Tensor,Tensor> CPUFloatType::max_pool2d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 2);
    auto stride_ = check_intlist<2>(stride, "stride", 3, kernel_size);
    auto padding_ = check_intlist<2>(padding, "padding", 4);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 5);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    auto indices_ = new CPULongTensor(context);
    auto indices = Tensor(indices_, false);
    THNN_FloatSpatialDilatedMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], ceil_mode);
    bool maybe_scalar = self_->isScalar();
    output_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(output, indices);
}
Tensor & CPUFloatType::max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto stride_ = check_intlist<2>(stride, "stride", 4, kernel_size);
    auto padding_ = check_intlist<2>(padding, "padding", 5);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 6);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",8, false);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",8, false);
    THNN_FloatSpatialDilatedMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], ceil_mode);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::max_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto stride_ = check_intlist<2>(stride, "stride", 4, kernel_size);
    auto padding_ = check_intlist<2>(padding, "padding", 5);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 6);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",8, false);
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatSpatialDilatedMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], ceil_mode);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &> CPUFloatType::max_pool3d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 2);
    auto stride_ = check_intlist<3>(stride, "stride", 3, kernel_size);
    auto padding_ = check_intlist<3>(padding, "padding", 4);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 5);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",6, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",6, false);
    THNN_FloatVolumetricDilatedMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], ceil_mode);
    bool maybe_scalar = self_->isScalar();
    output_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(output, indices);
}
std::tuple<Tensor,Tensor> CPUFloatType::max_pool3d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 2);
    auto stride_ = check_intlist<3>(stride, "stride", 3, kernel_size);
    auto padding_ = check_intlist<3>(padding, "padding", 4);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 5);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    auto indices_ = new CPULongTensor(context);
    auto indices = Tensor(indices_, false);
    THNN_FloatVolumetricDilatedMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], ceil_mode);
    bool maybe_scalar = self_->isScalar();
    output_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(output, indices);
}
Tensor & CPUFloatType::max_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto stride_ = check_intlist<3>(stride, "stride", 4, kernel_size);
    auto padding_ = check_intlist<3>(padding, "padding", 5);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 6);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",8, false);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",8, false);
    THNN_FloatVolumetricDilatedMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], ceil_mode);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::max_pool3d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto stride_ = check_intlist<3>(stride, "stride", 4, kernel_size);
    auto padding_ = check_intlist<3>(padding, "padding", 5);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 6);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",8, false);
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatVolumetricDilatedMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], ceil_mode);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUFloatType::max_unpool2d_forward_out(Tensor & output, const Tensor & self, const Tensor & indices, IntList output_size) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",2, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 3);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",3, false);
    THNN_FloatSpatialMaxUnpooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, output_size_[1], output_size_[0]);
    output_->maybeScalar(self_->isScalar() && indices_->isScalar());
    return output;
}
Tensor CPUFloatType::max_unpool2d_forward(const Tensor & self, const Tensor & indices, IntList output_size) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",2, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 3);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_FloatSpatialMaxUnpooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, output_size_[1], output_size_[0]);
    output_->maybeScalar(self_->isScalar() && indices_->isScalar());
    return output;
}
Tensor & CPUFloatType::max_unpool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",3, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 4);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_FloatSpatialMaxUnpooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor, output_size_[1], output_size_[0]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::max_unpool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",3, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 4);
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatSpatialMaxUnpooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor, output_size_[1], output_size_[0]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUFloatType::max_unpool3d_forward_out(Tensor & output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",2, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 3);
    auto stride_ = check_intlist<3>(stride, "stride", 4);
    auto padding_ = check_intlist<3>(padding, "padding", 5);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",5, false);
    THNN_FloatVolumetricMaxUnpooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, output_size_[0], output_size_[2], output_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1]);
    output_->maybeScalar(self_->isScalar() && indices_->isScalar());
    return output;
}
Tensor CPUFloatType::max_unpool3d_forward(const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",2, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 3);
    auto stride_ = check_intlist<3>(stride, "stride", 4);
    auto padding_ = check_intlist<3>(padding, "padding", 5);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_FloatVolumetricMaxUnpooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, output_size_[0], output_size_[2], output_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1]);
    output_->maybeScalar(self_->isScalar() && indices_->isScalar());
    return output;
}
Tensor & CPUFloatType::max_unpool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",3, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 4);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",6, false);
    THNN_FloatVolumetricMaxUnpooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor, output_size_[0], output_size_[2], output_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::max_unpool3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",3, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 4);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatVolumetricMaxUnpooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor, output_size_[0], output_size_[2], output_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUFloatType::reflection_pad1d_forward_out(Tensor & output, const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<2>(padding, "padding", 2);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",2, false);
    THNN_FloatTemporalReflectionPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUFloatType::reflection_pad1d_forward(const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<2>(padding, "padding", 2);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_FloatTemporalReflectionPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUFloatType::reflection_pad1d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<2>(padding, "padding", 3);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_FloatTemporalReflectionPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::reflection_pad1d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<2>(padding, "padding", 3);
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatTemporalReflectionPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUFloatType::reflection_pad2d_forward_out(Tensor & output, const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<4>(padding, "padding", 2);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",2, false);
    THNN_FloatSpatialReflectionPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1], padding_[2], padding_[3]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUFloatType::reflection_pad2d_forward(const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<4>(padding, "padding", 2);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_FloatSpatialReflectionPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1], padding_[2], padding_[3]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUFloatType::reflection_pad2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<4>(padding, "padding", 3);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_FloatSpatialReflectionPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1], padding_[2], padding_[3]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::reflection_pad2d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<4>(padding, "padding", 3);
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatSpatialReflectionPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1], padding_[2], padding_[3]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUFloatType::replication_pad1d_forward_out(Tensor & output, const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<2>(padding, "padding", 2);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",2, false);
    THNN_FloatTemporalReplicationPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUFloatType::replication_pad1d_forward(const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<2>(padding, "padding", 2);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_FloatTemporalReplicationPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUFloatType::replication_pad1d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<2>(padding, "padding", 3);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_FloatTemporalReplicationPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::replication_pad1d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<2>(padding, "padding", 3);
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatTemporalReplicationPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUFloatType::replication_pad2d_forward_out(Tensor & output, const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<4>(padding, "padding", 2);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",2, false);
    THNN_FloatSpatialReplicationPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1], padding_[2], padding_[3]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUFloatType::replication_pad2d_forward(const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<4>(padding, "padding", 2);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_FloatSpatialReplicationPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1], padding_[2], padding_[3]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUFloatType::replication_pad2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<4>(padding, "padding", 3);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_FloatSpatialReplicationPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1], padding_[2], padding_[3]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::replication_pad2d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<4>(padding, "padding", 3);
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatSpatialReplicationPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1], padding_[2], padding_[3]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUFloatType::replication_pad3d_forward_out(Tensor & output, const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<6>(padding, "padding", 2);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",2, false);
    THNN_FloatVolumetricReplicationPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1], padding_[2], padding_[3], padding_[4], padding_[5]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUFloatType::replication_pad3d_forward(const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<6>(padding, "padding", 2);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_FloatVolumetricReplicationPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1], padding_[2], padding_[3], padding_[4], padding_[5]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUFloatType::replication_pad3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<6>(padding, "padding", 3);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_FloatVolumetricReplicationPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1], padding_[2], padding_[3], padding_[4], padding_[5]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::replication_pad3d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<6>(padding, "padding", 3);
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatVolumetricReplicationPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1], padding_[2], padding_[3], padding_[4], padding_[5]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUFloatType::upsample_linear1d_forward_out(Tensor & output, const Tensor & self, IntList output_size, bool align_corners) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<1>(output_size, "output_size", 2);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",3, false);
    THNN_FloatTemporalUpSamplingLinear_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[0], align_corners);
    return output;
}
Tensor CPUFloatType::upsample_linear1d_forward(const Tensor & self, IntList output_size, bool align_corners) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<1>(output_size, "output_size", 2);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_FloatTemporalUpSamplingLinear_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[0], align_corners);
    return output;
}
Tensor & CPUFloatType::upsample_linear1d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_size_ = check_intlist<1>(output_size, "output_size", 2);
    auto input_size_ = check_intlist<3>(input_size, "input_size", 3);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_FloatTemporalUpSamplingLinear_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, input_size_[0], input_size_[1], input_size_[2], output_size_[0], align_corners);
    grad_input_->maybeScalar(false);
    return grad_input;
}
Tensor CPUFloatType::upsample_linear1d_backward(const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_size_ = check_intlist<1>(output_size, "output_size", 2);
    auto input_size_ = check_intlist<3>(input_size, "input_size", 3);
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatTemporalUpSamplingLinear_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, input_size_[0], input_size_[1], input_size_[2], output_size_[0], align_corners);
    grad_input_->maybeScalar(false);
    return grad_input;
}
Tensor & CPUFloatType::upsample_bilinear2d_forward_out(Tensor & output, const Tensor & self, IntList output_size, bool align_corners) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 2);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",3, false);
    THNN_FloatSpatialUpSamplingBilinear_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[0], output_size_[1], align_corners);
    return output;
}
Tensor CPUFloatType::upsample_bilinear2d_forward(const Tensor & self, IntList output_size, bool align_corners) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 2);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_FloatSpatialUpSamplingBilinear_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[0], output_size_[1], align_corners);
    return output;
}
Tensor & CPUFloatType::upsample_bilinear2d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 2);
    auto input_size_ = check_intlist<4>(input_size, "input_size", 3);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_FloatSpatialUpSamplingBilinear_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, input_size_[0], input_size_[1], input_size_[2], input_size_[3], output_size_[0], output_size_[1], align_corners);
    grad_input_->maybeScalar(false);
    return grad_input;
}
Tensor CPUFloatType::upsample_bilinear2d_backward(const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 2);
    auto input_size_ = check_intlist<4>(input_size, "input_size", 3);
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatSpatialUpSamplingBilinear_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, input_size_[0], input_size_[1], input_size_[2], input_size_[3], output_size_[0], output_size_[1], align_corners);
    grad_input_->maybeScalar(false);
    return grad_input;
}
Tensor & CPUFloatType::upsample_trilinear3d_forward_out(Tensor & output, const Tensor & self, IntList output_size, bool align_corners) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 2);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",3, false);
    THNN_FloatVolumetricUpSamplingTrilinear_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[0], output_size_[1], output_size_[2], align_corners);
    return output;
}
Tensor CPUFloatType::upsample_trilinear3d_forward(const Tensor & self, IntList output_size, bool align_corners) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 2);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_FloatVolumetricUpSamplingTrilinear_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[0], output_size_[1], output_size_[2], align_corners);
    return output;
}
Tensor & CPUFloatType::upsample_trilinear3d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 2);
    auto input_size_ = check_intlist<5>(input_size, "input_size", 3);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_FloatVolumetricUpSamplingTrilinear_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, input_size_[0], input_size_[1], input_size_[2], input_size_[3], input_size_[4], output_size_[0], output_size_[1], output_size_[2], align_corners);
    grad_input_->maybeScalar(false);
    return grad_input;
}
Tensor CPUFloatType::upsample_trilinear3d_backward(const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 2);
    auto input_size_ = check_intlist<5>(input_size, "input_size", 3);
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatVolumetricUpSamplingTrilinear_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, input_size_[0], input_size_[1], input_size_[2], input_size_[3], input_size_[4], output_size_[0], output_size_[1], output_size_[2], align_corners);
    grad_input_->maybeScalar(false);
    return grad_input;
}
Tensor & CPUFloatType::upsample_nearest1d_forward_out(Tensor & output, const Tensor & self, int64_t scale_factor) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",2, false);
    THNN_FloatTemporalUpSamplingNearest_updateOutput(context->thc_state, self_->tensor, output_->tensor, scale_factor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUFloatType::upsample_nearest1d_forward(const Tensor & self, int64_t scale_factor) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_FloatTemporalUpSamplingNearest_updateOutput(context->thc_state, self_->tensor, output_->tensor, scale_factor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUFloatType::upsample_nearest1d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_FloatTemporalUpSamplingNearest_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, scale_factor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::upsample_nearest1d_backward(const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatTemporalUpSamplingNearest_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, scale_factor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUFloatType::upsample_nearest2d_forward_out(Tensor & output, const Tensor & self, int64_t scale_factor) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",2, false);
    THNN_FloatSpatialUpSamplingNearest_updateOutput(context->thc_state, self_->tensor, output_->tensor, scale_factor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUFloatType::upsample_nearest2d_forward(const Tensor & self, int64_t scale_factor) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_FloatSpatialUpSamplingNearest_updateOutput(context->thc_state, self_->tensor, output_->tensor, scale_factor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUFloatType::upsample_nearest2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_FloatSpatialUpSamplingNearest_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, scale_factor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::upsample_nearest2d_backward(const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatSpatialUpSamplingNearest_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, scale_factor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUFloatType::upsample_nearest3d_forward_out(Tensor & output, const Tensor & self, int64_t scale_factor) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",2, false);
    THNN_FloatVolumetricUpSamplingNearest_updateOutput(context->thc_state, self_->tensor, output_->tensor, scale_factor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUFloatType::upsample_nearest3d_forward(const Tensor & self, int64_t scale_factor) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_FloatVolumetricUpSamplingNearest_updateOutput(context->thc_state, self_->tensor, output_->tensor, scale_factor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUFloatType::upsample_nearest3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_FloatVolumetricUpSamplingNearest_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, scale_factor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::upsample_nearest3d_backward(const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatVolumetricUpSamplingNearest_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, scale_factor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUFloatType::_sigmoid_forward_out(Tensor & output, const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",1, false);
    THNN_FloatSigmoid_updateOutput(context->thc_state, self_->tensor, output_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUFloatType::_sigmoid_forward(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_FloatSigmoid_updateOutput(context->thc_state, self_->tensor, output_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUFloatType::_sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",2, false);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",2, false);
    THNN_FloatSigmoid_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, output_->tensor);
    grad_input_->maybeScalar(output_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::_sigmoid_backward(const Tensor & grad_output, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",2, false);
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatSigmoid_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, output_->tensor);
    grad_input_->maybeScalar(output_->isScalar());
    return grad_input;
}
Tensor & CPUFloatType::_tanh_forward_out(Tensor & output, const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",1, false);
    THNN_FloatTanh_updateOutput(context->thc_state, self_->tensor, output_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUFloatType::_tanh_forward(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    THNN_FloatTanh_updateOutput(context->thc_state, self_->tensor, output_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUFloatType::_tanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",2, false);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",2, false);
    THNN_FloatTanh_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, output_->tensor);
    grad_input_->maybeScalar(output_->isScalar());
    return grad_input;
}
Tensor CPUFloatType::_tanh_backward(const Tensor & grad_output, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",2, false);
    auto grad_input_ = new CPUFloatTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_FloatTanh_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, output_->tensor);
    grad_input_->maybeScalar(output_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &,Tensor &> CPUFloatType::thnn_batch_norm_forward_out(Tensor & output, Tensor & save_mean, Tensor & save_std, const Tensor & self, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",2, true);
    auto bias_ = checked_cast_tensor<CPUFloatTensor>(bias.pImpl,"bias",3, true);
    auto running_mean_ = checked_cast_tensor<CPUFloatTensor>(running_mean.pImpl,"running_mean",4, true);
    auto running_var_ = checked_cast_tensor<CPUFloatTensor>(running_var.pImpl,"running_var",5, true);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",8, false);
    auto save_mean_ = checked_cast_tensor<CPUFloatTensor>(save_mean.pImpl,"save_mean",8, false);
    auto save_std_ = checked_cast_tensor<CPUFloatTensor>(save_std.pImpl,"save_std",8, false);
    THNN_FloatBatchNormalization_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_ ? weight_->tensor : NULL, bias_ ? bias_->tensor : NULL, running_mean_ ? running_mean_->tensor : NULL, running_var_ ? running_var_->tensor : NULL, save_mean_->tensor, save_std_->tensor, training, momentum, eps);
    bool maybe_scalar = self_->isScalar() && (!weight_ || weight_->isScalar()) && (!bias_ || bias_->isScalar()) && (!running_mean_ || running_mean_->isScalar()) && (!running_var_ || running_var_->isScalar());
    output_->maybeScalar(maybe_scalar);
    save_mean_->maybeScalar(maybe_scalar);
    save_std_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(output, save_mean, save_std);
}
std::tuple<Tensor,Tensor,Tensor> CPUFloatType::thnn_batch_norm_forward(const Tensor & self, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",2, true);
    auto bias_ = checked_cast_tensor<CPUFloatTensor>(bias.pImpl,"bias",3, true);
    auto running_mean_ = checked_cast_tensor<CPUFloatTensor>(running_mean.pImpl,"running_mean",4, true);
    auto running_var_ = checked_cast_tensor<CPUFloatTensor>(running_var.pImpl,"running_var",5, true);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    auto save_mean_ = new CPUFloatTensor(context);
    auto save_mean = Tensor(save_mean_, false);
    auto save_std_ = new CPUFloatTensor(context);
    auto save_std = Tensor(save_std_, false);
    THNN_FloatBatchNormalization_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_ ? weight_->tensor : NULL, bias_ ? bias_->tensor : NULL, running_mean_ ? running_mean_->tensor : NULL, running_var_ ? running_var_->tensor : NULL, save_mean_->tensor, save_std_->tensor, training, momentum, eps);
    bool maybe_scalar = self_->isScalar() && (!weight_ || weight_->isScalar()) && (!bias_ || bias_->isScalar()) && (!running_mean_ || running_mean_->isScalar()) && (!running_var_ || running_var_->isScalar());
    output_->maybeScalar(maybe_scalar);
    save_mean_->maybeScalar(maybe_scalar);
    save_std_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(output, save_mean, save_std);
}
std::tuple<Tensor &,Tensor &,Tensor &> CPUFloatType::thnn_batch_norm_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, bool training, double eps, const Tensor & save_mean, const Tensor & save_std) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",3, true);
    auto running_mean_ = checked_cast_tensor<CPUFloatTensor>(running_mean.pImpl,"running_mean",4, true);
    auto running_var_ = checked_cast_tensor<CPUFloatTensor>(running_var.pImpl,"running_var",5, true);
    auto save_mean_ = checked_cast_tensor<CPUFloatTensor>(save_mean.pImpl,"save_mean",8, true);
    auto save_std_ = checked_cast_tensor<CPUFloatTensor>(save_std.pImpl,"save_std",9, true);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",9, true);
    auto grad_weight_ = checked_cast_tensor<CPUFloatTensor>(grad_weight.pImpl,"grad_weight",9, true);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = checked_cast_tensor<CPUFloatTensor>(grad_bias.pImpl,"grad_bias",9, true);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    THNN_FloatBatchNormalization_backward(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, weight_ ? weight_->tensor : NULL, running_mean_ ? running_mean_->tensor : NULL, running_var_ ? running_var_->tensor : NULL, save_mean_ ? save_mean_->tensor : NULL, save_std_ ? save_std_->tensor : NULL, training, 1, eps);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &, Tensor &>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> CPUFloatType::thnn_batch_norm_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, bool training, double eps, const Tensor & save_mean, const Tensor & save_std, std::array<bool,3> output_mask) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",3, true);
    auto running_mean_ = checked_cast_tensor<CPUFloatTensor>(running_mean.pImpl,"running_mean",4, true);
    auto running_var_ = checked_cast_tensor<CPUFloatTensor>(running_var.pImpl,"running_var",5, true);
    auto save_mean_ = checked_cast_tensor<CPUFloatTensor>(save_mean.pImpl,"save_mean",8, true);
    auto save_std_ = checked_cast_tensor<CPUFloatTensor>(save_std.pImpl,"save_std",9, true);
    auto grad_input_ = output_mask[0] ? new CPUFloatTensor(context) : nullptr;
    auto grad_input = Tensor(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_input_, false);
    auto grad_weight_ = output_mask[1] ? new CPUFloatTensor(context) : nullptr;
    auto grad_weight = Tensor(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_weight_, false);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = output_mask[2] ? new CPUFloatTensor(context) : nullptr;
    auto grad_bias = Tensor(grad_bias_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_bias_, false);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    THNN_FloatBatchNormalization_backward(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, weight_ ? weight_->tensor : NULL, running_mean_ ? running_mean_->tensor : NULL, running_var_ ? running_var_->tensor : NULL, save_mean_ ? save_mean_->tensor : NULL, save_std_ ? save_std_->tensor : NULL, training, 1, eps);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor &,Tensor &,Tensor &> CPUFloatType::thnn_conv_transpose2d_forward_out(Tensor & output, Tensor & columns, Tensor & ones, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CPUFloatTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto output_padding_ = check_intlist<2>(output_padding, "output_padding", 7);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 8);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",8, false);
    auto columns_ = checked_cast_tensor<CPUFloatTensor>(columns.pImpl,"columns",8, false);
    auto ones_ = checked_cast_tensor<CPUFloatTensor>(ones.pImpl,"ones",8, false);
    THNN_FloatSpatialFullDilatedConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], output_padding_[1], output_padding_[0]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    columns_->maybeScalar(maybe_scalar);
    ones_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(output, columns, ones);
}
std::tuple<Tensor,Tensor,Tensor> CPUFloatType::thnn_conv_transpose2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CPUFloatTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto output_padding_ = check_intlist<2>(output_padding, "output_padding", 7);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 8);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    auto columns_ = new CPUFloatTensor(context);
    auto columns = Tensor(columns_, false);
    auto ones_ = new CPUFloatTensor(context);
    auto ones = Tensor(ones_, false);
    THNN_FloatSpatialFullDilatedConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], output_padding_[1], output_padding_[0]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    columns_->maybeScalar(maybe_scalar);
    ones_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(output, columns, ones);
}
std::tuple<Tensor &,Tensor &,Tensor &> CPUFloatType::thnn_conv_transpose2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & columns, const Tensor & ones) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto output_padding_ = check_intlist<2>(output_padding, "output_padding", 7);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 8);
    auto columns_ = checked_cast_tensor<CPUFloatTensor>(columns.pImpl,"columns",9, false);
    auto ones_ = checked_cast_tensor<CPUFloatTensor>(ones.pImpl,"ones",10, false);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",10, true);
    auto grad_weight_ = checked_cast_tensor<CPUFloatTensor>(grad_weight.pImpl,"grad_weight",10, true);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = checked_cast_tensor<CPUFloatTensor>(grad_bias.pImpl,"grad_bias",10, true);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(1) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_FloatSpatialFullDilatedConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, columns_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], output_padding_[1], output_padding_[0]);
    if (grad_weight_ || grad_bias_) THNN_FloatSpatialFullDilatedConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], output_padding_[1], output_padding_[0], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &, Tensor &>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> CPUFloatType::thnn_conv_transpose2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto output_padding_ = check_intlist<2>(output_padding, "output_padding", 7);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 8);
    auto columns_ = checked_cast_tensor<CPUFloatTensor>(columns.pImpl,"columns",9, false);
    auto ones_ = checked_cast_tensor<CPUFloatTensor>(ones.pImpl,"ones",10, false);
    auto grad_input_ = output_mask[0] ? new CPUFloatTensor(context) : nullptr;
    auto grad_input = Tensor(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_input_, false);
    auto grad_weight_ = output_mask[1] ? new CPUFloatTensor(context) : nullptr;
    auto grad_weight = Tensor(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_weight_, false);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = output_mask[2] ? new CPUFloatTensor(context) : nullptr;
    auto grad_bias = Tensor(grad_bias_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_bias_, false);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(1) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_FloatSpatialFullDilatedConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, columns_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], output_padding_[1], output_padding_[0]);
    if (grad_weight_ || grad_bias_) THNN_FloatSpatialFullDilatedConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], output_padding_[1], output_padding_[0], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor &,Tensor &,Tensor &> CPUFloatType::thnn_conv_transpose3d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CPUFloatTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto output_padding_ = check_intlist<3>(output_padding, "output_padding", 7);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 8);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",8, false);
    auto finput_ = checked_cast_tensor<CPUFloatTensor>(finput.pImpl,"finput",8, false);
    auto fgrad_input_ = checked_cast_tensor<CPUFloatTensor>(fgrad_input.pImpl,"fgrad_input",8, false);
    THNN_FloatVolumetricFullDilatedConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], output_padding_[0], output_padding_[2], output_padding_[1]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    finput_->maybeScalar(maybe_scalar);
    fgrad_input_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(output, finput, fgrad_input);
}
std::tuple<Tensor,Tensor,Tensor> CPUFloatType::thnn_conv_transpose3d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CPUFloatTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto output_padding_ = check_intlist<3>(output_padding, "output_padding", 7);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 8);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    auto finput_ = new CPUFloatTensor(context);
    auto finput = Tensor(finput_, false);
    auto fgrad_input_ = new CPUFloatTensor(context);
    auto fgrad_input = Tensor(fgrad_input_, false);
    THNN_FloatVolumetricFullDilatedConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], output_padding_[0], output_padding_[2], output_padding_[1]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    finput_->maybeScalar(maybe_scalar);
    fgrad_input_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(output, finput, fgrad_input);
}
std::tuple<Tensor &,Tensor &,Tensor &> CPUFloatType::thnn_conv_transpose3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & finput, const Tensor & fgrad_input) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto output_padding_ = check_intlist<3>(output_padding, "output_padding", 7);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 8);
    auto finput_ = checked_cast_tensor<CPUFloatTensor>(finput.pImpl,"finput",9, false);
    auto fgrad_input_ = checked_cast_tensor<CPUFloatTensor>(fgrad_input.pImpl,"fgrad_input",10, false);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",10, true);
    auto grad_weight_ = checked_cast_tensor<CPUFloatTensor>(grad_weight.pImpl,"grad_weight",10, true);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = checked_cast_tensor<CPUFloatTensor>(grad_bias.pImpl,"grad_bias",10, true);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(1) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_FloatVolumetricFullDilatedConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], output_padding_[0], output_padding_[2], output_padding_[1]);
    if (grad_weight_ || grad_bias_) THNN_FloatVolumetricFullDilatedConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], output_padding_[0], output_padding_[2], output_padding_[1], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &, Tensor &>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> CPUFloatType::thnn_conv_transpose3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto output_padding_ = check_intlist<3>(output_padding, "output_padding", 7);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 8);
    auto finput_ = checked_cast_tensor<CPUFloatTensor>(finput.pImpl,"finput",9, false);
    auto fgrad_input_ = checked_cast_tensor<CPUFloatTensor>(fgrad_input.pImpl,"fgrad_input",10, false);
    auto grad_input_ = output_mask[0] ? new CPUFloatTensor(context) : nullptr;
    auto grad_input = Tensor(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_input_, false);
    auto grad_weight_ = output_mask[1] ? new CPUFloatTensor(context) : nullptr;
    auto grad_weight = Tensor(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_weight_, false);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = output_mask[2] ? new CPUFloatTensor(context) : nullptr;
    auto grad_bias = Tensor(grad_bias_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_bias_, false);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(1) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_FloatVolumetricFullDilatedConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], output_padding_[0], output_padding_[2], output_padding_[1]);
    if (grad_weight_ || grad_bias_) THNN_FloatVolumetricFullDilatedConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], output_padding_[0], output_padding_[2], output_padding_[1], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor &,Tensor &,Tensor &> CPUFloatType::thnn_conv2d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CPUFloatTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",6, false);
    auto finput_ = checked_cast_tensor<CPUFloatTensor>(finput.pImpl,"finput",6, false);
    auto fgrad_input_ = checked_cast_tensor<CPUFloatTensor>(fgrad_input.pImpl,"fgrad_input",6, false);
    THNN_FloatSpatialConvolutionMM_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    finput_->maybeScalar(maybe_scalar);
    fgrad_input_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(output, finput, fgrad_input);
}
std::tuple<Tensor,Tensor,Tensor> CPUFloatType::thnn_conv2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CPUFloatTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    auto finput_ = new CPUFloatTensor(context);
    auto finput = Tensor(finput_, false);
    auto fgrad_input_ = new CPUFloatTensor(context);
    auto fgrad_input = Tensor(fgrad_input_, false);
    THNN_FloatSpatialConvolutionMM_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    finput_->maybeScalar(maybe_scalar);
    fgrad_input_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(output, finput, fgrad_input);
}
std::tuple<Tensor &,Tensor &,Tensor &> CPUFloatType::thnn_conv2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto finput_ = checked_cast_tensor<CPUFloatTensor>(finput.pImpl,"finput",7, false);
    auto fgrad_input_ = checked_cast_tensor<CPUFloatTensor>(fgrad_input.pImpl,"fgrad_input",8, false);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",8, true);
    auto grad_weight_ = checked_cast_tensor<CPUFloatTensor>(grad_weight.pImpl,"grad_weight",8, true);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = checked_cast_tensor<CPUFloatTensor>(grad_bias.pImpl,"grad_bias",8, true);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_FloatSpatialConvolutionMM_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, finput_->tensor, fgrad_input_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
    if (grad_weight_ || grad_bias_) THNN_FloatSpatialConvolutionMM_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &, Tensor &>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> CPUFloatType::thnn_conv2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto finput_ = checked_cast_tensor<CPUFloatTensor>(finput.pImpl,"finput",7, false);
    auto fgrad_input_ = checked_cast_tensor<CPUFloatTensor>(fgrad_input.pImpl,"fgrad_input",8, false);
    auto grad_input_ = output_mask[0] ? new CPUFloatTensor(context) : nullptr;
    auto grad_input = Tensor(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_input_, false);
    auto grad_weight_ = output_mask[1] ? new CPUFloatTensor(context) : nullptr;
    auto grad_weight = Tensor(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_weight_, false);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = output_mask[2] ? new CPUFloatTensor(context) : nullptr;
    auto grad_bias = Tensor(grad_bias_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_bias_, false);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_FloatSpatialConvolutionMM_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, finput_->tensor, fgrad_input_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
    if (grad_weight_ || grad_bias_) THNN_FloatSpatialConvolutionMM_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor &,Tensor &,Tensor &> CPUFloatType::thnn_conv3d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CPUFloatTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",6, false);
    auto finput_ = checked_cast_tensor<CPUFloatTensor>(finput.pImpl,"finput",6, false);
    auto fgrad_input_ = checked_cast_tensor<CPUFloatTensor>(fgrad_input.pImpl,"fgrad_input",6, false);
    THNN_FloatVolumetricConvolutionMM_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    finput_->maybeScalar(maybe_scalar);
    fgrad_input_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(output, finput, fgrad_input);
}
std::tuple<Tensor,Tensor,Tensor> CPUFloatType::thnn_conv3d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CPUFloatTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    auto finput_ = new CPUFloatTensor(context);
    auto finput = Tensor(finput_, false);
    auto fgrad_input_ = new CPUFloatTensor(context);
    auto fgrad_input = Tensor(fgrad_input_, false);
    THNN_FloatVolumetricConvolutionMM_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    finput_->maybeScalar(maybe_scalar);
    fgrad_input_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(output, finput, fgrad_input);
}
std::tuple<Tensor &,Tensor &,Tensor &> CPUFloatType::thnn_conv3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto finput_ = checked_cast_tensor<CPUFloatTensor>(finput.pImpl,"finput",7, false);
    auto fgrad_input_ = checked_cast_tensor<CPUFloatTensor>(fgrad_input.pImpl,"fgrad_input",8, false);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",8, true);
    auto grad_weight_ = checked_cast_tensor<CPUFloatTensor>(grad_weight.pImpl,"grad_weight",8, true);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = checked_cast_tensor<CPUFloatTensor>(grad_bias.pImpl,"grad_bias",8, true);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_FloatVolumetricConvolutionMM_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1]);
    if (grad_weight_ || grad_bias_) THNN_FloatVolumetricConvolutionMM_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &, Tensor &>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> CPUFloatType::thnn_conv3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto finput_ = checked_cast_tensor<CPUFloatTensor>(finput.pImpl,"finput",7, false);
    auto fgrad_input_ = checked_cast_tensor<CPUFloatTensor>(fgrad_input.pImpl,"fgrad_input",8, false);
    auto grad_input_ = output_mask[0] ? new CPUFloatTensor(context) : nullptr;
    auto grad_input = Tensor(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_input_, false);
    auto grad_weight_ = output_mask[1] ? new CPUFloatTensor(context) : nullptr;
    auto grad_weight = Tensor(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_weight_, false);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = output_mask[2] ? new CPUFloatTensor(context) : nullptr;
    auto grad_bias = Tensor(grad_bias_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_bias_, false);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_FloatVolumetricConvolutionMM_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1]);
    if (grad_weight_ || grad_bias_) THNN_FloatVolumetricConvolutionMM_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor &,Tensor &,Tensor &> CPUFloatType::thnn_conv_dilated2d_forward_out(Tensor & output, Tensor & columns, Tensor & ones, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CPUFloatTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",7, false);
    auto columns_ = checked_cast_tensor<CPUFloatTensor>(columns.pImpl,"columns",7, false);
    auto ones_ = checked_cast_tensor<CPUFloatTensor>(ones.pImpl,"ones",7, false);
    THNN_FloatSpatialDilatedConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    columns_->maybeScalar(maybe_scalar);
    ones_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(output, columns, ones);
}
std::tuple<Tensor,Tensor,Tensor> CPUFloatType::thnn_conv_dilated2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CPUFloatTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    auto columns_ = new CPUFloatTensor(context);
    auto columns = Tensor(columns_, false);
    auto ones_ = new CPUFloatTensor(context);
    auto ones = Tensor(ones_, false);
    THNN_FloatSpatialDilatedConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    columns_->maybeScalar(maybe_scalar);
    ones_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(output, columns, ones);
}
std::tuple<Tensor &,Tensor &,Tensor &> CPUFloatType::thnn_conv_dilated2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
    auto columns_ = checked_cast_tensor<CPUFloatTensor>(columns.pImpl,"columns",8, false);
    auto ones_ = checked_cast_tensor<CPUFloatTensor>(ones.pImpl,"ones",9, false);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",9, true);
    auto grad_weight_ = checked_cast_tensor<CPUFloatTensor>(grad_weight.pImpl,"grad_weight",9, true);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = checked_cast_tensor<CPUFloatTensor>(grad_bias.pImpl,"grad_bias",9, true);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_FloatSpatialDilatedConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, columns_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
    if (grad_weight_ || grad_bias_) THNN_FloatSpatialDilatedConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &, Tensor &>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> CPUFloatType::thnn_conv_dilated2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
    auto columns_ = checked_cast_tensor<CPUFloatTensor>(columns.pImpl,"columns",8, false);
    auto ones_ = checked_cast_tensor<CPUFloatTensor>(ones.pImpl,"ones",9, false);
    auto grad_input_ = output_mask[0] ? new CPUFloatTensor(context) : nullptr;
    auto grad_input = Tensor(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_input_, false);
    auto grad_weight_ = output_mask[1] ? new CPUFloatTensor(context) : nullptr;
    auto grad_weight = Tensor(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_weight_, false);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = output_mask[2] ? new CPUFloatTensor(context) : nullptr;
    auto grad_bias = Tensor(grad_bias_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_bias_, false);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_FloatSpatialDilatedConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, columns_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
    if (grad_weight_ || grad_bias_) THNN_FloatSpatialDilatedConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor &,Tensor &,Tensor &> CPUFloatType::thnn_conv_dilated3d_forward_out(Tensor & output, Tensor & columns, Tensor & ones, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CPUFloatTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 7);
    auto output_ = checked_cast_tensor<CPUFloatTensor>(output.pImpl,"output",7, false);
    auto columns_ = checked_cast_tensor<CPUFloatTensor>(columns.pImpl,"columns",7, false);
    auto ones_ = checked_cast_tensor<CPUFloatTensor>(ones.pImpl,"ones",7, false);
    THNN_FloatVolumetricDilatedConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    columns_->maybeScalar(maybe_scalar);
    ones_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(output, columns, ones);
}
std::tuple<Tensor,Tensor,Tensor> CPUFloatType::thnn_conv_dilated3d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CPUFloatTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 7);
    auto output_ = new CPUFloatTensor(context);
    auto output = Tensor(output_, false);
    auto columns_ = new CPUFloatTensor(context);
    auto columns = Tensor(columns_, false);
    auto ones_ = new CPUFloatTensor(context);
    auto ones = Tensor(ones_, false);
    THNN_FloatVolumetricDilatedConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    columns_->maybeScalar(maybe_scalar);
    ones_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(output, columns, ones);
}
std::tuple<Tensor &,Tensor &,Tensor &> CPUFloatType::thnn_conv_dilated3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 7);
    auto columns_ = checked_cast_tensor<CPUFloatTensor>(columns.pImpl,"columns",8, false);
    auto ones_ = checked_cast_tensor<CPUFloatTensor>(ones.pImpl,"ones",9, false);
    auto grad_input_ = checked_cast_tensor<CPUFloatTensor>(grad_input.pImpl,"grad_input",9, true);
    auto grad_weight_ = checked_cast_tensor<CPUFloatTensor>(grad_weight.pImpl,"grad_weight",9, true);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = checked_cast_tensor<CPUFloatTensor>(grad_bias.pImpl,"grad_bias",9, true);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_FloatVolumetricDilatedConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, columns_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1]);
    if (grad_weight_ || grad_bias_) THNN_FloatVolumetricDilatedConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &, Tensor &>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> CPUFloatType::thnn_conv_dilated3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) const {
    auto grad_output_ = checked_cast_tensor<CPUFloatTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUFloatTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CPUFloatTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 7);
    auto columns_ = checked_cast_tensor<CPUFloatTensor>(columns.pImpl,"columns",8, false);
    auto ones_ = checked_cast_tensor<CPUFloatTensor>(ones.pImpl,"ones",9, false);
    auto grad_input_ = output_mask[0] ? new CPUFloatTensor(context) : nullptr;
    auto grad_input = Tensor(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_input_, false);
    auto grad_weight_ = output_mask[1] ? new CPUFloatTensor(context) : nullptr;
    auto grad_weight = Tensor(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_weight_, false);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = output_mask[2] ? new CPUFloatTensor(context) : nullptr;
    auto grad_bias = Tensor(grad_bias_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_bias_, false);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_FloatVolumetricDilatedConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, columns_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1]);
    if (grad_weight_ || grad_bias_) THNN_FloatVolumetricDilatedConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}
Tensor & CPUFloatType::abs_out(Tensor & result, const Tensor & self) const {
    return  at::native::_abs_out_cpu(result, self);
}
Tensor & CPUFloatType::ceil_out(Tensor & result, const Tensor & self) const {
    return  at::native::_ceil_out_cpu(result, self);
}
Tensor & CPUFloatType::cos_out(Tensor & result, const Tensor & self) const {
    return  at::native::_cos_out_cpu(result, self);
}
Tensor CPUFloatType::embedding_dense_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) const {
    return  at::native::embedding_backward_cpu(grad, indices, num_weights, padding_idx, scale_grad_by_freq);
}
Tensor & CPUFloatType::embedding_renorm_(Tensor & self, const Tensor & indices, double max_norm, double norm_type) const {
    return  at::native::embedding_renorm_cpu_(self, indices, max_norm, norm_type);
}
std::tuple<Tensor,Tensor,Tensor> CPUFloatType::embedding_bag(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse) const {
    return  at::native::embedding_bag_cpu(weight, indices, offsets, scale_grad_by_freq, mode, sparse);
}
Tensor CPUFloatType::embedding_bag_dense_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode) const {
    return  at::native::embedding_bag_backward_cpu(grad, indices, offsets, offset2bag, bag_size, num_weights, scale_grad_by_freq, mode);
}
Tensor & CPUFloatType::exp_out(Tensor & result, const Tensor & self) const {
    return  at::native::_exp_out_cpu(result, self);
}
Tensor & CPUFloatType::eye_out(Tensor & result, int64_t n, int64_t m) const {
    return  at::native::eye_out_cpu(result, n, m);
}
Tensor & CPUFloatType::floor_out(Tensor & result, const Tensor & self) const {
    return  at::native::_floor_out_cpu(result, self);
}
Tensor CPUFloatType::_fft_with_size(const Tensor & self, int64_t signal_ndim, bool complex_input, bool complex_output, bool inverse, IntList checked_signal_sizes, bool normalized, bool onesided, IntList output_sizes) const {
    return  at::native::_fft_mkl(self, signal_ndim, complex_input, complex_output, inverse, checked_signal_sizes, normalized, onesided, output_sizes);
}
Tensor & CPUFloatType::log_out(Tensor & result, const Tensor & self) const {
    return  at::native::_log_out_cpu(result, self);
}
std::tuple<Tensor,Tensor> CPUFloatType::RoiPooling2d_forward(const Tensor & input, const Tensor & rois, int64_t pooledHeight, int64_t pooledWidth, double spatialScale) const {
    return  at::native::RoiPooling2d_forward_cpu(input, rois, pooledHeight, pooledWidth, spatialScale);
}
Tensor CPUFloatType::RoiPooling2d_backward(const Tensor & input, const Tensor & rois, int64_t pooledHeight, int64_t pooledWidth, double spatialScale, const Tensor & gradOutput, const Tensor & argmaxes) const {
    return  at::native::RoiPooling2d_backward_cpu(input, rois, pooledHeight, pooledWidth, spatialScale, gradOutput, argmaxes);
}
Tensor & CPUFloatType::round_out(Tensor & result, const Tensor & self) const {
    return  at::native::_round_out_cpu(result, self);
}
Tensor & CPUFloatType::sin_out(Tensor & result, const Tensor & self) const {
    return  at::native::_sin_out_cpu(result, self);
}
Tensor & CPUFloatType::sspaddmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return  at::native::_sspaddmm_out_only_sparse(result, self, mat1, mat2, beta, alpha);
}
Tensor CPUFloatType::sum(const Tensor & self) const {
    return  at::native::_sum_cpu(self);
}
Tensor & CPUFloatType::sum_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    return  at::native::_sum_out_cpu(result, self, dim, keepdim);
}
Tensor & CPUFloatType::sqrt_out(Tensor & result, const Tensor & self) const {
    return  at::native::_sqrt_out_cpu(result, self);
}
Tensor CPUFloatType::prod(const Tensor & self) const {
    return  at::native::_prod_cpu(self);
}
Tensor & CPUFloatType::prod_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    return  at::native::_prod_out_cpu(result, self, dim, keepdim);
}
Tensor & CPUFloatType::trunc_out(Tensor & result, const Tensor & self) const {
    return  at::native::_trunc_out_cpu(result, self);
}
std::tuple<Tensor,Tensor> CPUFloatType::_unique(const Tensor & self, bool sorted, bool return_inverse) const {
    return  at::native::_unique_cpu(self, sorted, return_inverse);
}
Tensor CPUFloatType::_s_where(const Tensor & condition, const Tensor & self, const Tensor & other) const {
    return  at::native::_s_where_cpu(condition, self, other);
}
Tensor CPUFloatType::_standard_gamma_grad(const Tensor & self, const Tensor & output) const {
    return  at::native::_standard_gamma_grad_cpu(self, output);
}
Tensor CPUFloatType::poisson(const Tensor & self, Generator * generator) const {
    return  at::native::_s_poisson_cpu(self, generator);
}

}
