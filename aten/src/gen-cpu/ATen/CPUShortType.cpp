// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/CPUShortType.h"
#include "ATen/CPUShortStorage.h"
#include "ATen/CPUShortTensor.h"
#include "ATen/CPUGenerator.h"
#include "ATen/CPUByteTensor.h"
#include "ATen/CPUIntTensor.h"
#include "ATen/CPULongTensor.h"
#include "ATen/SparseCPUShortTensor.h"
#include "ATen/CPUShortTensor.h"
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

CPUShortType::CPUShortType(Context* context)
: Type(context, /*is_variable_or_undefined=*/false) {}
ScalarType CPUShortType::scalarType() const {
  return ScalarType::Short;
}
Backend CPUShortType::backend() const {
  return Backend::CPU;
}
bool CPUShortType::is_cuda() const { return backend() == kCUDA || backend() == kSparseCUDA; }
bool CPUShortType::is_sparse() const { return backend() == kSparseCPU || backend() == kSparseCUDA; }
bool CPUShortType::is_distributed() const { return false; }

std::unique_ptr<Storage> CPUShortType::storage() const {
  return std::unique_ptr<Storage>(new CPUShortStorage(context));
}
std::unique_ptr<Storage> CPUShortType::storage(size_t size) const {
  return std::unique_ptr<Storage>(new CPUShortStorage(context,size));
}
std::unique_ptr<Storage> CPUShortType::storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const {
    return std::unique_ptr<Storage>(
      new CPUShortStorage(context,data,size,deleter));
}
std::unique_ptr<Storage> CPUShortType::storageWithAllocator(int64_t size, std::unique_ptr<Allocator> allocator) const {
    return std::unique_ptr<Storage>(
        new CPUShortStorage(context, size, std::move(allocator)));
}
Tensor CPUShortType::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THShortTensor_retain( (THShortTensor*) th_pointer);
  return Tensor(new CPUShortTensor(context,(THShortTensor*)(th_pointer)), false);
}
std::unique_ptr<Storage> CPUShortType::unsafeStorageFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THShortStorage_retain( (THShortStorage*) th_pointer);
  return std::unique_ptr<Storage>(new CPUShortStorage(context, (THShortStorage*) th_pointer));
}
std::unique_ptr<Generator> CPUShortType::generator() const {
  return std::unique_ptr<Generator>(new CPUGenerator(context));
}

const char * CPUShortType::toString() const {
  return CPUShortType::typeString();
}
TypeID CPUShortType::ID() const {
  return TypeID::CPUShort;
}

std::size_t CPUShortType::elementSizeInBytes() const {
  return sizeof(int16_t);
}

const char * CPUShortType::typeString() {
  return "CPUShortType";
}

/* example
Tensor * CPUShortType::add(Tensor & a, Tensor & b) {
  std::cout << "add CPUShortTensor\n";
  return &a;
}
*/

int64_t CPUShortType::storage_offset(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    return static_cast<int64_t>(THShortTensor_storageOffset(self_->tensor));
}
Tensor & CPUShortType::resize_(Tensor & self, IntList size) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    THShortTensor_resize(self_->tensor, size_, NULL);
    self_->maybeScalar(size.size() == 0);
    return self;
}
int64_t CPUShortType::numel(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    return static_cast<int64_t>(THShortTensor_nElement(self_->tensor));
}
Tensor & CPUShortType::set_(Tensor & self, Storage & storage) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto storage_ = checked_cast_storage<CPUShortStorage>(&storage,"storage",2);
    THShortTensor_setStorage(self_->tensor, storage_->storage, 0, THLongStorageView::makeFromLength(static_cast<int64_t>(storage.size())), NULL);
    self_->maybeScalar(false);
    return self;
}
Tensor & CPUShortType::set_(Tensor & self, Storage & sourceStorage, int64_t storage_offset, IntList size, IntList stride) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto sourceStorage_ = checked_cast_storage<CPUShortStorage>(&sourceStorage,"sourceStorage",2);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    THShortTensor_setStorage(self_->tensor, sourceStorage_->storage, storage_offset, size_, stride_);
    self_->maybeScalar(size.size() == 0);
    return self;
}
Tensor & CPUShortType::set_(Tensor & self, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto source_ = checked_cast_tensor<CPUShortTensor>(source.pImpl,"source",2, false);
    THShortTensor_set(self_->tensor, source_->tensor);
    self_->maybeScalar(source_->isScalar());
    return self;
}
Tensor & CPUShortType::set_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    THShortTensor_setStorage(self_->tensor, NULL, 0, NULL, NULL);
    self_->maybeScalar(false);
    return self;
}
Tensor & CPUShortType::fill_(Tensor & self, Scalar value) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toShort();
    THShortTensor_fill(self_->tensor, value_);
    return self;
}
Tensor & CPUShortType::fill_(Tensor & self, const Tensor & value) const {
    if (value.dim() == 0) {
        return static_cast<const Type*>(this)->fill_(self, Scalar(value));
    }
    AT_ERROR("fill_ only supports a 0-dimensional value tensor, but got tensor "
        "with %" PRId64 " dimension(s)", value.dim());
}
bool CPUShortType::is_contiguous(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    return THShortTensor_isContiguous(self_->tensor);
}
bool CPUShortType::is_set_to(const Tensor & self, const Tensor & tensor) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto tensor_ = checked_cast_tensor<CPUShortTensor>(tensor.pImpl,"tensor",2, false);
    return THShortTensor_isSetTo(self_->tensor, tensor_->tensor);
}
Tensor & CPUShortType::s_masked_fill_(Tensor & self, const Tensor & mask, Scalar value) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CPUByteTensor>(mask.pImpl,"mask",2, false);
    auto value_ = value.toShort();
    THShortTensor_maskedFill(self_->tensor, mask_->tensor, value_);
    return self;
}
Tensor & CPUShortType::s_masked_fill_(Tensor & self, const Tensor & mask, const Tensor & value) const {
    if (value.dim() == 0) {
        return static_cast<const Type*>(this)->masked_fill_(self, mask, Scalar(value));
    }
    AT_ERROR("masked_fill_ only supports a 0-dimensional value tensor, but got tensor "
        "with %" PRId64 " dimension(s)", value.dim());
}
Tensor & CPUShortType::s_masked_scatter_(Tensor & self, const Tensor & mask, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CPUByteTensor>(mask.pImpl,"mask",2, false);
    auto source_ = checked_cast_tensor<CPUShortTensor>(source.pImpl,"source",3, false);
    THShortTensor_maskedCopy(self_->tensor, mask_->tensor, source_->tensor);
    return self;
}
Tensor & CPUShortType::s_masked_select_out(Tensor & result, const Tensor & self, const Tensor & mask) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CPUByteTensor>(mask.pImpl,"mask",2, false);
    THShortTensor_maskedSelect(result_->tensor, self_->tensor, mask_->tensor);
    result_->maybeScalar(self_->isScalar() && mask_->isScalar());
    return result;
}
Tensor CPUShortType::s_masked_select(const Tensor & self, const Tensor & mask) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CPUByteTensor>(mask.pImpl,"mask",2, false);
    THShortTensor_maskedSelect(result_->tensor, self_->tensor, mask_->tensor);
    result_->maybeScalar(self_->isScalar() && mask_->isScalar());
    return result;
}
Tensor CPUShortType::transpose(const Tensor & self, int64_t dim0, int64_t dim1) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    dim0 = maybe_wrap_dim(dim0, self_);
    dim1 = maybe_wrap_dim(dim1, self_);
    return Tensor((new CPUShortTensor(context, THShortTensor_newTranspose(self_->tensor, dim0, dim1)))->maybeScalar(self_->isScalar()),false);
}
Tensor CPUShortType::t(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPUShortTensor(context, THShortTensor_newTranspose(self_->tensor, 0, 1)))->maybeScalar(self_->isScalar()),false);
}
Tensor & CPUShortType::nonzero_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    THShortTensor_nonzero(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUShortType::nonzero(const Tensor & self) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    THShortTensor_nonzero(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUShortType::contiguous(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPUShortTensor(context, THShortTensor_newContiguous(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor CPUShortType::clone(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPUShortTensor(context, THShortTensor_newClone(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor CPUShortType::view(const Tensor & self, IntList size) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new CPUShortTensor(context, THShortTensor_newView(self_->tensor, size_)))->maybeScalar(size.size() == 0),false);
}
Tensor & CPUShortType::resize_as_(Tensor & self, const Tensor & the_template) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto the_template_ = checked_cast_tensor<CPUShortTensor>(the_template.pImpl,"the_template",2, false);
    THShortTensor_resizeAs(self_->tensor, the_template_->tensor);
    self_->maybeScalar(the_template_->isScalar());
    return self;
}
Tensor & CPUShortType::index_select_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    THShortTensor_indexSelect(result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
Tensor CPUShortType::index_select(const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    THShortTensor_indexSelect(result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
Tensor & CPUShortType::_indexCopy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    auto source_ = checked_cast_tensor<CPUShortTensor>(source.pImpl,"source",4, false);
    THShortTensor_indexCopy(self_->tensor, dim, index_->tensor, source_->tensor);
    return self;
}
Tensor & CPUShortType::take_out(Tensor & result, const Tensor & self, const Tensor & index) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",2, false);
    THShortTensor_take(result_->tensor, self_->tensor, index_->tensor);
    result_->maybeScalar(index_->isScalar());
    return result;
}
Tensor CPUShortType::take(const Tensor & self, const Tensor & index) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",2, false);
    THShortTensor_take(result_->tensor, self_->tensor, index_->tensor);
    result_->maybeScalar(index_->isScalar());
    return result;
}
Tensor & CPUShortType::put_(Tensor & self, const Tensor & index, const Tensor & source, bool accumulate) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",2, false);
    auto source_ = checked_cast_tensor<CPUShortTensor>(source.pImpl,"source",3, false);
    THShortTensor_put(self_->tensor, index_->tensor, source_->tensor, accumulate);
    return self;
}
Tensor & CPUShortType::index_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    auto source_ = checked_cast_tensor<CPUShortTensor>(source.pImpl,"source",4, false);
    THShortTensor_indexAdd(self_->tensor, dim, index_->tensor, source_->tensor);
    return self;
}
Tensor & CPUShortType::index_fill_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    auto value_ = value.toShort();
    THShortTensor_indexFill(self_->tensor, dim, index_->tensor, value_);
    return self;
}
Tensor & CPUShortType::index_fill_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & value) const {
    if (value.dim() == 0) {
        return static_cast<const Type*>(this)->index_fill_(self, dim, index, Scalar(value));
    }
    AT_ERROR("index_fill_ only supports a 0-dimensional value tensor, but got tensor "
        "with %" PRId64 " dimension(s)", value.dim());
}
Tensor CPUShortType::unfold(const Tensor & self, int64_t dimension, int64_t size, int64_t step) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    dimension = maybe_wrap_dim(dimension, self_);
    THShortTensor_unfold(result_->tensor, self_->tensor, dimension, size, step);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUShortType::_range_out(Tensor & result, Scalar start, Scalar end, Scalar step) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto start_ = start.toLong();
    auto end_ = end.toLong();
    auto step_ = step.toLong();
    THShortTensor_range(result_->tensor, start_, end_, step_);
    return result;
}
Tensor CPUShortType::_range(Scalar start, Scalar end, Scalar step) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto start_ = start.toLong();
    auto end_ = end.toLong();
    auto step_ = step.toLong();
    THShortTensor_range(result_->tensor, start_, end_, step_);
    return result;
}
Tensor & CPUShortType::_arange_out(Tensor & result, Scalar start, Scalar end, Scalar step) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto start_ = start.toLong();
    auto end_ = end.toLong();
    auto step_ = step.toLong();
    THShortTensor_arange(result_->tensor, start_, end_, step_);
    return result;
}
Tensor CPUShortType::_arange(Scalar start, Scalar end, Scalar step) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto start_ = start.toLong();
    auto end_ = end.toLong();
    auto step_ = step.toLong();
    THShortTensor_arange(result_->tensor, start_, end_, step_);
    return result;
}
Tensor & CPUShortType::_arange_out(Tensor & result, Scalar end) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto end_ = end.toLong();
    THShortTensor_arange(result_->tensor, 0, end_, 1);
    return result;
}
Tensor CPUShortType::_arange(Scalar end) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto end_ = end.toLong();
    THShortTensor_arange(result_->tensor, 0, end_, 1);
    return result;
}
Tensor & CPUShortType::scatter_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
    if (src.dim() == 0) {
        return static_cast<const Type*>(this)->scatter_(self, dim, index, Scalar(src));
    }
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    auto src_ = checked_cast_tensor<CPUShortTensor>(src.pImpl,"src",4, false);
    THShortTensor_scatter(self_->tensor, dim, index_->tensor, src_->tensor);
    return self;
}
Tensor & CPUShortType::scatter_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    auto value_ = value.toShort();
    THShortTensor_scatterFill(self_->tensor, dim, index_->tensor, value_);
    return self;
}
Tensor & CPUShortType::scatter_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    auto src_ = checked_cast_tensor<CPUShortTensor>(src.pImpl,"src",4, false);
    THShortTensor_scatterAdd(self_->tensor, dim, index_->tensor, src_->tensor);
    return self;
}
Tensor & CPUShortType::gather_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    result.resize_(index.sizes());
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    THShortTensor_gather(result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
Tensor CPUShortType::gather(const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    result.resize_(index.sizes());
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    THShortTensor_gather(result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
void* CPUShortType::data_ptr(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    return THShortTensor_data(self_->tensor);
}
bool CPUShortType::equal(const Tensor & self, const Tensor & other) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",2, false);
    return THShortTensor_equal(self_->tensor, other_->tensor);
}
Tensor & CPUShortType::__and___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_bitand(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUShortType::__and__(const Tensor & self, Scalar other) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_bitand(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUShortType::s___and___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__and___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",2, false);
    THShortTensor_cbitand(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUShortType::s___and__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__and__(self, Scalar(other));
    }
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",2, false);
    THShortTensor_cbitand(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUShortType::__iand__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_bitand(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUShortType::s___iand__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__iand__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",3, false);
    THShortTensor_cbitand(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUShortType::__or___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_bitor(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUShortType::__or__(const Tensor & self, Scalar other) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_bitor(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUShortType::s___or___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__or___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",2, false);
    THShortTensor_cbitor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUShortType::s___or__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__or__(self, Scalar(other));
    }
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",2, false);
    THShortTensor_cbitor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUShortType::__ior__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_bitor(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUShortType::s___ior__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__ior__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",3, false);
    THShortTensor_cbitor(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUShortType::__xor___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_bitxor(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUShortType::__xor__(const Tensor & self, Scalar other) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_bitxor(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUShortType::s___xor___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__xor___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",2, false);
    THShortTensor_cbitxor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUShortType::s___xor__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__xor__(self, Scalar(other));
    }
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",2, false);
    THShortTensor_cbitxor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUShortType::__ixor__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_bitxor(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUShortType::s___ixor__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__ixor__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",3, false);
    THShortTensor_cbitxor(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUShortType::__lshift___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_lshift(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUShortType::__lshift__(const Tensor & self, Scalar other) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_lshift(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUShortType::s___lshift___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__lshift___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",2, false);
    THShortTensor_clshift(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUShortType::s___lshift__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__lshift__(self, Scalar(other));
    }
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",2, false);
    THShortTensor_clshift(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUShortType::__ilshift__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_lshift(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUShortType::s___ilshift__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__ilshift__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",3, false);
    THShortTensor_clshift(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUShortType::__rshift___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_rshift(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUShortType::__rshift__(const Tensor & self, Scalar other) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_rshift(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUShortType::s___rshift___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__rshift___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",2, false);
    THShortTensor_crshift(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUShortType::s___rshift__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__rshift__(self, Scalar(other));
    }
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",2, false);
    THShortTensor_crshift(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUShortType::__irshift__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_rshift(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUShortType::s___irshift__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__irshift__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",3, false);
    THShortTensor_crshift(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUShortType::lt_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_ltValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUShortType::lt(const Tensor & self, Scalar other) const {
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_ltValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUShortType::s_lt_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->lt_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",2, false);
    THShortTensor_ltTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUShortType::s_lt(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->lt(self, Scalar(other));
    }
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",2, false);
    THShortTensor_ltTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUShortType::lt_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_ltValueT(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUShortType::s_lt_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->lt_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",3, false);
    THShortTensor_ltTensorT(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUShortType::gt_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_gtValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUShortType::gt(const Tensor & self, Scalar other) const {
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_gtValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUShortType::s_gt_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->gt_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",2, false);
    THShortTensor_gtTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUShortType::s_gt(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->gt(self, Scalar(other));
    }
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",2, false);
    THShortTensor_gtTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUShortType::gt_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_gtValueT(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUShortType::s_gt_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->gt_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",3, false);
    THShortTensor_gtTensorT(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUShortType::le_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_leValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUShortType::le(const Tensor & self, Scalar other) const {
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_leValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUShortType::s_le_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->le_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",2, false);
    THShortTensor_leTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUShortType::s_le(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->le(self, Scalar(other));
    }
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",2, false);
    THShortTensor_leTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUShortType::le_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_leValueT(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUShortType::s_le_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->le_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",3, false);
    THShortTensor_leTensorT(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUShortType::ge_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_geValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUShortType::ge(const Tensor & self, Scalar other) const {
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_geValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUShortType::s_ge_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ge_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",2, false);
    THShortTensor_geTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUShortType::s_ge(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ge(self, Scalar(other));
    }
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",2, false);
    THShortTensor_geTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUShortType::ge_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_geValueT(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUShortType::s_ge_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ge_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",3, false);
    THShortTensor_geTensorT(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUShortType::eq_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_eqValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUShortType::eq(const Tensor & self, Scalar other) const {
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_eqValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUShortType::s_eq_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->eq_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",2, false);
    THShortTensor_eqTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUShortType::s_eq(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->eq(self, Scalar(other));
    }
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",2, false);
    THShortTensor_eqTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUShortType::eq_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_eqValueT(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUShortType::s_eq_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->eq_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",3, false);
    THShortTensor_eqTensorT(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUShortType::ne_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_neValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUShortType::ne(const Tensor & self, Scalar other) const {
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_neValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUShortType::s_ne_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ne_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",2, false);
    THShortTensor_neTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUShortType::s_ne(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ne(self, Scalar(other));
    }
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",2, false);
    THShortTensor_neTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUShortType::ne_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_neValueT(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUShortType::s_ne_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ne_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",3, false);
    THShortTensor_neTensorT(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
std::tuple<Tensor &,Tensor &> CPUShortType::min_out(Tensor & min, Tensor & min_indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto min_ = checked_cast_tensor<CPUShortTensor>(min.pImpl,"min",0, false);
    auto min_indices_ = checked_cast_tensor<CPULongTensor>(min_indices.pImpl,"min_indices",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THShortTensor_min(min_->tensor, min_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    min_->maybeScalar(maybe_scalar);
    min_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(min, min_indices);
}
std::tuple<Tensor,Tensor> CPUShortType::min(const Tensor & self, int64_t dim, bool keepdim) const {
    auto min_ = new CPUShortTensor(context);
    auto min = Tensor(min_, false);
    auto min_indices_ = new CPULongTensor(context);
    auto min_indices = Tensor(min_indices_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THShortTensor_min(min_->tensor, min_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    min_->maybeScalar(maybe_scalar);
    min_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(min, min_indices);
}
Tensor & CPUShortType::s_min_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",2, false);
    THShortTensor_cmin(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUShortType::s_min(const Tensor & self, const Tensor & other) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",2, false);
    THShortTensor_cmin(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUShortType::min(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<int16_t>(THShortTensor_minall(self_->tensor)));
}
std::tuple<Tensor &,Tensor &> CPUShortType::max_out(Tensor & max, Tensor & max_indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto max_ = checked_cast_tensor<CPUShortTensor>(max.pImpl,"max",0, false);
    auto max_indices_ = checked_cast_tensor<CPULongTensor>(max_indices.pImpl,"max_indices",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THShortTensor_max(max_->tensor, max_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    max_->maybeScalar(maybe_scalar);
    max_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(max, max_indices);
}
std::tuple<Tensor,Tensor> CPUShortType::max(const Tensor & self, int64_t dim, bool keepdim) const {
    auto max_ = new CPUShortTensor(context);
    auto max = Tensor(max_, false);
    auto max_indices_ = new CPULongTensor(context);
    auto max_indices = Tensor(max_indices_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THShortTensor_max(max_->tensor, max_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    max_->maybeScalar(maybe_scalar);
    max_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(max, max_indices);
}
Tensor & CPUShortType::s_max_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",2, false);
    THShortTensor_cmax(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUShortType::s_max(const Tensor & self, const Tensor & other) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",2, false);
    THShortTensor_cmax(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUShortType::max(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<int16_t>(THShortTensor_maxall(self_->tensor)));
}
std::tuple<Tensor &,Tensor &> CPUShortType::kthvalue_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool keepdim) const {
    auto values_ = checked_cast_tensor<CPUShortTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THShortTensor_kthvalue(values_->tensor, indices_->tensor, self_->tensor, k, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CPUShortType::kthvalue(const Tensor & self, int64_t k, int64_t dim, bool keepdim) const {
    auto values_ = new CPUShortTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CPULongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THShortTensor_kthvalue(values_->tensor, indices_->tensor, self_->tensor, k, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
std::tuple<Tensor &,Tensor &> CPUShortType::mode_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = checked_cast_tensor<CPUShortTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THShortTensor_mode(values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CPUShortType::mode(const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = new CPUShortTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CPULongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THShortTensor_mode(values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
std::tuple<Tensor &,Tensor &> CPUShortType::median_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = checked_cast_tensor<CPUShortTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THShortTensor_median(values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CPUShortType::median(const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = new CPUShortTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CPULongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THShortTensor_median(values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
Tensor CPUShortType::median(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<int16_t>(THShortTensor_medianall(self_->tensor)));
}
std::tuple<Tensor &,Tensor &> CPUShortType::sort_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool descending) const {
    auto values_ = checked_cast_tensor<CPUShortTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THShortTensor_sort(values_->tensor, indices_->tensor, self_->tensor, dim, descending);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CPUShortType::sort(const Tensor & self, int64_t dim, bool descending) const {
    auto values_ = new CPUShortTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CPULongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THShortTensor_sort(values_->tensor, indices_->tensor, self_->tensor, dim, descending);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
std::tuple<Tensor &,Tensor &> CPUShortType::topk_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) const {
    auto values_ = checked_cast_tensor<CPUShortTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THShortTensor_topk(values_->tensor, indices_->tensor, self_->tensor, k, dim, largest, sorted);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CPUShortType::topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) const {
    auto values_ = new CPUShortTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CPULongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THShortTensor_topk(values_->tensor, indices_->tensor, self_->tensor, k, dim, largest, sorted);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
Tensor & CPUShortType::_abs_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    THShortTensor_abs(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUShortType::_abs(const Tensor & self) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    THShortTensor_abs(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUShortType::neg_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    THShortTensor_neg(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUShortType::neg(const Tensor & self) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    THShortTensor_neg(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUShortType::neg_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    THShortTensor_neg(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUShortType::pow_out(Tensor & result, const Tensor & self, Scalar exponent) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toShort();
    THShortTensor_pow(result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUShortType::pow(const Tensor & self, Scalar exponent) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toShort();
    THShortTensor_pow(result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUShortType::s_pow_out(Tensor & result, const Tensor & self, const Tensor & exponent) const {
    if (exponent.dim() == 0) {
        return static_cast<const Type*>(this)->pow_out(result, self, Scalar(exponent));
    }
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto exponent_ = checked_cast_tensor<CPUShortTensor>(exponent.pImpl,"exponent",2, false);
    THShortTensor_cpow(result_->tensor, self_->tensor, exponent_->tensor);
    result_->maybeScalar(self_->isScalar() && exponent_->isScalar());
    return result;
}
Tensor CPUShortType::s_pow(const Tensor & self, const Tensor & exponent) const {
    if (exponent.dim() == 0) {
        return static_cast<const Type*>(this)->pow(self, Scalar(exponent));
    }
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto exponent_ = checked_cast_tensor<CPUShortTensor>(exponent.pImpl,"exponent",2, false);
    THShortTensor_cpow(result_->tensor, self_->tensor, exponent_->tensor);
    result_->maybeScalar(self_->isScalar() && exponent_->isScalar());
    return result;
}
Tensor & CPUShortType::pow_out(Tensor & result, Scalar base, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto base_ = base.toShort();
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",2, false);
    THShortTensor_tpow(result_->tensor, base_, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUShortType::pow(Scalar base, const Tensor & self) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto base_ = base.toShort();
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",2, false);
    THShortTensor_tpow(result_->tensor, base_, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUShortType::pow_(Tensor & self, Scalar exponent) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toShort();
    THShortTensor_pow(self_->tensor, self_->tensor, exponent_);
    return self;
}
Tensor & CPUShortType::s_pow_(Tensor & self, const Tensor & exponent) const {
    if (exponent.dim() == 0) {
        return static_cast<const Type*>(this)->pow_(self, Scalar(exponent));
    }
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto exponent_ = checked_cast_tensor<CPUShortTensor>(exponent.pImpl,"exponent",3, false);
    THShortTensor_cpow(self_->tensor, self_->tensor, exponent_->tensor);
    return self;
}
Tensor & CPUShortType::zero_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    THShortTensor_zero(self_->tensor);
    return self;
}
Tensor CPUShortType::_sumall(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<int16_t>(THShortTensor_sumall(self_->tensor)));
}
Tensor & CPUShortType::_sum_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THShortTensor_sum(result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CPUShortType::_sum(const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THShortTensor_sum(result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CPUShortType::_prodall(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<int16_t>(THShortTensor_prodall(self_->tensor)));
}
Tensor & CPUShortType::_prod_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THShortTensor_prod(result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CPUShortType::_prod(const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THShortTensor_prod(result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor & CPUShortType::cumsum_out(Tensor & result, const Tensor & self, int64_t dim) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THShortTensor_cumsum(result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUShortType::cumsum(const Tensor & self, int64_t dim) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THShortTensor_cumsum(result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUShortType::cumprod_out(Tensor & result, const Tensor & self, int64_t dim) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THShortTensor_cumprod(result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUShortType::cumprod(const Tensor & self, int64_t dim) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THShortTensor_cumprod(result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUShortType::sign_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    THShortTensor_sign(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUShortType::sign(const Tensor & self) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    THShortTensor_sign(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUShortType::sign_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    THShortTensor_sign(self_->tensor, self_->tensor);
    return self;
}
Tensor CPUShortType::trace(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<int16_t>(THShortTensor_trace(self_->tensor)));
}
Tensor & CPUShortType::add_out(Tensor & result, const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    auto alpha_ = alpha.toShort();
    THShortTensor_add_scaled(result_->tensor, self_->tensor, other_, alpha_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUShortType::add(const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    auto alpha_ = alpha.toShort();
    THShortTensor_add_scaled(result_->tensor, self_->tensor, other_, alpha_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUShortType::s_add_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if(other.type().is_sparse()) {
        return static_cast<const Type*>(this)->add_out(result, self, SparseTensor(other), alpha);
    }
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toShort();
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",3, false);
    THShortTensor_cadd(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUShortType::s_add(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if(other.type().is_sparse()) {
        return static_cast<const Type*>(this)->add(self, SparseTensor(other), alpha);
    }
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add(self, Scalar(other), alpha);
    }
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toShort();
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",3, false);
    THShortTensor_cadd(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUShortType::add_out(Tensor & result, const Tensor & self, SparseTensor other, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toShort();
    auto other_ = checked_cast_tensor<SparseCPUShortTensor>(other.tref.pImpl,"other",3,false);
    THSShortTensor_spcadd(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUShortType::add(const Tensor & self, SparseTensor other, Scalar alpha) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toShort();
    auto other_ = checked_cast_tensor<SparseCPUShortTensor>(other.tref.pImpl,"other",3,false);
    THSShortTensor_spcadd(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUShortType::add_(Tensor & self, Scalar other, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    auto alpha_ = alpha.toShort();
    THShortTensor_add_scaled(self_->tensor, self_->tensor, other_, alpha_);
    return self;
}
Tensor & CPUShortType::s_add_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if(other.type().is_sparse()) {
        return static_cast<const Type*>(this)->add_(self, SparseTensor(other), alpha);
    }
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toShort();
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",4, false);
    THShortTensor_cadd(self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & CPUShortType::add_(Tensor & self, SparseTensor other, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toShort();
    auto other_ = checked_cast_tensor<SparseCPUShortTensor>(other.tref.pImpl,"other",4,false);
    THSShortTensor_spcadd(self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & CPUShortType::sub_out(Tensor & result, const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    auto alpha_ = alpha.toShort();
    THShortTensor_sub_scaled(result_->tensor, self_->tensor, other_, alpha_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUShortType::sub(const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    auto alpha_ = alpha.toShort();
    THShortTensor_sub_scaled(result_->tensor, self_->tensor, other_, alpha_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUShortType::s_sub_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toShort();
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",3, false);
    THShortTensor_csub(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUShortType::s_sub(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub(self, Scalar(other), alpha);
    }
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toShort();
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",3, false);
    THShortTensor_csub(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUShortType::sub_(Tensor & self, Scalar other, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    auto alpha_ = alpha.toShort();
    THShortTensor_sub_scaled(self_->tensor, self_->tensor, other_, alpha_);
    return self;
}
Tensor & CPUShortType::s_sub_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toShort();
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",4, false);
    THShortTensor_csub(self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & CPUShortType::mul_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_mul(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUShortType::mul(const Tensor & self, Scalar other) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_mul(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUShortType::s_mul_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",2, false);
    THShortTensor_cmul(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUShortType::s_mul(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul(self, Scalar(other));
    }
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",2, false);
    THShortTensor_cmul(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUShortType::mul_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_mul(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUShortType::s_mul_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",3, false);
    THShortTensor_cmul(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUShortType::div_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_div(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUShortType::div(const Tensor & self, Scalar other) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_div(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUShortType::s_div_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->div_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",2, false);
    THShortTensor_cdiv(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUShortType::s_div(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->div(self, Scalar(other));
    }
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",2, false);
    THShortTensor_cdiv(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUShortType::div_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_div(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUShortType::s_div_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->div_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",3, false);
    THShortTensor_cdiv(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUShortType::fmod_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_fmod(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUShortType::fmod(const Tensor & self, Scalar other) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_fmod(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUShortType::s_fmod_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->fmod_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",2, false);
    THShortTensor_cfmod(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUShortType::s_fmod(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->fmod(self, Scalar(other));
    }
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",2, false);
    THShortTensor_cfmod(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUShortType::fmod_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_fmod(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUShortType::s_fmod_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->fmod_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",3, false);
    THShortTensor_cfmod(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUShortType::remainder_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_remainder(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUShortType::remainder(const Tensor & self, Scalar other) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_remainder(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUShortType::s_remainder_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->remainder_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",2, false);
    THShortTensor_cremainder(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUShortType::s_remainder(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->remainder(self, Scalar(other));
    }
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",2, false);
    THShortTensor_cremainder(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUShortType::remainder_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toShort();
    THShortTensor_remainder(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUShortType::s_remainder_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->remainder_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",3, false);
    THShortTensor_cremainder(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUShortType::clamp_out(Tensor & result, const Tensor & self, Scalar min, Scalar max) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toShort();
    auto max_ = max.toShort();
    THShortTensor_clamp(result_->tensor, self_->tensor, min_, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUShortType::clamp(const Tensor & self, Scalar min, Scalar max) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toShort();
    auto max_ = max.toShort();
    THShortTensor_clamp(result_->tensor, self_->tensor, min_, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUShortType::clamp_(Tensor & self, Scalar min, Scalar max) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toShort();
    auto max_ = max.toShort();
    THShortTensor_clamp(self_->tensor, self_->tensor, min_, max_);
    return self;
}
Tensor & CPUShortType::clamp_min_out(Tensor & result, const Tensor & self, Scalar min) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toShort();
    THShortTensor_cmaxValue(result_->tensor, self_->tensor, min_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUShortType::clamp_min(const Tensor & self, Scalar min) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toShort();
    THShortTensor_cmaxValue(result_->tensor, self_->tensor, min_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUShortType::clamp_min_(Tensor & self, Scalar min) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toShort();
    THShortTensor_cmaxValue(self_->tensor, self_->tensor, min_);
    return self;
}
Tensor & CPUShortType::clamp_max_out(Tensor & result, const Tensor & self, Scalar max) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto max_ = max.toShort();
    THShortTensor_cminValue(result_->tensor, self_->tensor, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUShortType::clamp_max(const Tensor & self, Scalar max) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto max_ = max.toShort();
    THShortTensor_cminValue(result_->tensor, self_->tensor, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUShortType::clamp_max_(Tensor & self, Scalar max) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto max_ = max.toShort();
    THShortTensor_cminValue(self_->tensor, self_->tensor, max_);
    return self;
}
Tensor CPUShortType::_dot(const Tensor & self, const Tensor & tensor) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto tensor_ = checked_cast_tensor<CPUShortTensor>(tensor.pImpl,"tensor",2, false);
    return scalarTensor(convert<int16_t>(THShortTensor_dot(self_->tensor, tensor_->tensor)));
}
Tensor & CPUShortType::tril_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    THShortTensor_tril(result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUShortType::tril(const Tensor & self, int64_t diagonal) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    THShortTensor_tril(result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUShortType::tril_(Tensor & self, int64_t diagonal) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    THShortTensor_tril(self_->tensor, self_->tensor, diagonal);
    return self;
}
Tensor & CPUShortType::triu_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    THShortTensor_triu(result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUShortType::triu(const Tensor & self, int64_t diagonal) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    THShortTensor_triu(result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUShortType::triu_(Tensor & self, int64_t diagonal) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    THShortTensor_triu(self_->tensor, self_->tensor, diagonal);
    return self;
}
Tensor & CPUShortType::cross_out(Tensor & result, const Tensor & self, const Tensor & other, int64_t dim) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",2, false);
    THShortTensor_cross(result_->tensor, self_->tensor, other_->tensor, dim);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUShortType::cross(const Tensor & self, const Tensor & other, int64_t dim) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUShortTensor>(other.pImpl,"other",2, false);
    THShortTensor_cross(result_->tensor, self_->tensor, other_->tensor, dim);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUShortType::diag_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    if (self_->isScalar()) {
      throw std::runtime_error("Input must be 1-d or 2-d");
    }
    THShortTensor_diag(result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUShortType::diag(const Tensor & self, int64_t diagonal) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    if (self_->isScalar()) {
      throw std::runtime_error("Input must be 1-d or 2-d");
    }
    THShortTensor_diag(result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUShortType::s_addmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    if(mat1.type().is_sparse()) {
        return static_cast<const Type*>(this)->addmm_out(result, self, SparseTensor(mat1), mat2, beta, alpha);
    }
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toShort();
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toShort();
    auto mat1_ = checked_cast_tensor<CPUShortTensor>(mat1.pImpl,"mat1",4, false);
    auto mat2_ = checked_cast_tensor<CPUShortTensor>(mat2.pImpl,"mat2",5, false);
    THShortTensor_addmm(result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CPUShortType::s_addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    if(mat1.type().is_sparse()) {
        return static_cast<const Type*>(this)->addmm(self, SparseTensor(mat1), mat2, beta, alpha);
    }
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toShort();
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toShort();
    auto mat1_ = checked_cast_tensor<CPUShortTensor>(mat1.pImpl,"mat1",4, false);
    auto mat2_ = checked_cast_tensor<CPUShortTensor>(mat2.pImpl,"mat2",5, false);
    THShortTensor_addmm(result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CPUShortType::addmm_out(Tensor & result, const Tensor & self, SparseTensor mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toShort();
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toShort();
    auto mat1_ = checked_cast_tensor<SparseCPUShortTensor>(mat1.tref.pImpl,"mat1",4,false);
    auto mat2_ = checked_cast_tensor<CPUShortTensor>(mat2.pImpl,"mat2",5, false);
    THSShortTensor_spaddmm(result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CPUShortType::addmm(const Tensor & self, SparseTensor mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toShort();
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toShort();
    auto mat1_ = checked_cast_tensor<SparseCPUShortTensor>(mat1.tref.pImpl,"mat1",4,false);
    auto mat2_ = checked_cast_tensor<CPUShortTensor>(mat2.pImpl,"mat2",5, false);
    THSShortTensor_spaddmm(result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CPUShortType::addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    if(mat1.type().is_sparse()) {
        return static_cast<const Type*>(this)->addmm_(self, SparseTensor(mat1), mat2, beta, alpha);
    }
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toShort();
    auto alpha_ = alpha.toShort();
    auto mat1_ = checked_cast_tensor<CPUShortTensor>(mat1.pImpl,"mat1",5, false);
    auto mat2_ = checked_cast_tensor<CPUShortTensor>(mat2.pImpl,"mat2",6, false);
    THShortTensor_addmm(self_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    return self;
}
Tensor & CPUShortType::addmm_(Tensor & self, SparseTensor mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toShort();
    auto alpha_ = alpha.toShort();
    auto mat1_ = checked_cast_tensor<SparseCPUShortTensor>(mat1.tref.pImpl,"mat1",5,false);
    auto mat2_ = checked_cast_tensor<CPUShortTensor>(mat2.pImpl,"mat2",6, false);
    THSShortTensor_spaddmm(self_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    return self;
}
Tensor & CPUShortType::s__addmv_out(Tensor & result, const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toShort();
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toShort();
    auto mat_ = checked_cast_tensor<CPUShortTensor>(mat.pImpl,"mat",4, false);
    auto vec_ = checked_cast_tensor<CPUShortTensor>(vec.pImpl,"vec",5, false);
    THShortTensor_addmv(result_->tensor, beta_, self_->tensor, alpha_, mat_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && mat_->isScalar() && vec_->isScalar());
    return result;
}
Tensor CPUShortType::s__addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toShort();
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toShort();
    auto mat_ = checked_cast_tensor<CPUShortTensor>(mat.pImpl,"mat",4, false);
    auto vec_ = checked_cast_tensor<CPUShortTensor>(vec.pImpl,"vec",5, false);
    THShortTensor_addmv(result_->tensor, beta_, self_->tensor, alpha_, mat_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && mat_->isScalar() && vec_->isScalar());
    return result;
}
Tensor & CPUShortType::_addmv_(Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toShort();
    auto alpha_ = alpha.toShort();
    auto mat_ = checked_cast_tensor<CPUShortTensor>(mat.pImpl,"mat",5, false);
    auto vec_ = checked_cast_tensor<CPUShortTensor>(vec.pImpl,"vec",6, false);
    THShortTensor_addmv(self_->tensor, beta_, self_->tensor, alpha_, mat_->tensor, vec_->tensor);
    return self;
}
Tensor & CPUShortType::s__addr_out(Tensor & result, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toShort();
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toShort();
    auto vec1_ = checked_cast_tensor<CPUShortTensor>(vec1.pImpl,"vec1",4, false);
    auto vec2_ = checked_cast_tensor<CPUShortTensor>(vec2.pImpl,"vec2",5, false);
    THShortTensor_addr(result_->tensor, beta_, self_->tensor, alpha_, vec1_->tensor, vec2_->tensor);
    result_->maybeScalar(self_->isScalar() && vec1_->isScalar() && vec2_->isScalar());
    return result;
}
Tensor CPUShortType::s__addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toShort();
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toShort();
    auto vec1_ = checked_cast_tensor<CPUShortTensor>(vec1.pImpl,"vec1",4, false);
    auto vec2_ = checked_cast_tensor<CPUShortTensor>(vec2.pImpl,"vec2",5, false);
    THShortTensor_addr(result_->tensor, beta_, self_->tensor, alpha_, vec1_->tensor, vec2_->tensor);
    result_->maybeScalar(self_->isScalar() && vec1_->isScalar() && vec2_->isScalar());
    return result;
}
Tensor & CPUShortType::_addr_(Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toShort();
    auto alpha_ = alpha.toShort();
    auto vec1_ = checked_cast_tensor<CPUShortTensor>(vec1.pImpl,"vec1",5, false);
    auto vec2_ = checked_cast_tensor<CPUShortTensor>(vec2.pImpl,"vec2",6, false);
    THShortTensor_addr(self_->tensor, beta_, self_->tensor, alpha_, vec1_->tensor, vec2_->tensor);
    return self;
}
Tensor & CPUShortType::_ger_out(Tensor & result, const Tensor & self, const Tensor & vec2) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.dim() == 0 ? 1 : self.size(0),vec2.dim() == 0 ? 1 : vec2.size(0) });
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto vec2_ = checked_cast_tensor<CPUShortTensor>(vec2.pImpl,"vec2",2, false);
    THShortTensor_addr(result_->tensor, int16_t(0), result_->tensor, int16_t(1), self_->tensor, vec2_->tensor);
    result_->maybeScalar(false);
    return result;
}
Tensor CPUShortType::_ger(const Tensor & self, const Tensor & vec2) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.dim() == 0 ? 1 : self.size(0),vec2.dim() == 0 ? 1 : vec2.size(0) });
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto vec2_ = checked_cast_tensor<CPUShortTensor>(vec2.pImpl,"vec2",2, false);
    THShortTensor_addr(result_->tensor, int16_t(0), result_->tensor, int16_t(1), self_->tensor, vec2_->tensor);
    result_->maybeScalar(false);
    return result;
}
Tensor & CPUShortType::_mv_out(Tensor & result, const Tensor & self, const Tensor & vec) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.size(0) });
    result.zero_();
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto vec_ = checked_cast_tensor<CPUShortTensor>(vec.pImpl,"vec",2, false);
    THShortTensor_addmv(result_->tensor, int16_t(0), result_->tensor, int16_t(1), self_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && vec_->isScalar());
    return result;
}
Tensor CPUShortType::_mv(const Tensor & self, const Tensor & vec) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.size(0) });
    result.zero_();
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto vec_ = checked_cast_tensor<CPUShortTensor>(vec.pImpl,"vec",2, false);
    THShortTensor_addmv(result_->tensor, int16_t(0), result_->tensor, int16_t(1), self_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && vec_->isScalar());
    return result;
}
Tensor & CPUShortType::_mm_out(Tensor & result, const Tensor & self, const Tensor & mat2) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.size(0),mat2.size(1) });
    result.zero_();
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CPUShortTensor>(mat2.pImpl,"mat2",2, false);
    THShortTensor_addmm(result_->tensor, int16_t(0), result_->tensor, int16_t(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CPUShortType::_mm(const Tensor & self, const Tensor & mat2) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.size(0),mat2.size(1) });
    result.zero_();
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CPUShortTensor>(mat2.pImpl,"mat2",2, false);
    THShortTensor_addmm(result_->tensor, int16_t(0), result_->tensor, int16_t(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CPUShortType::bmm_out(Tensor & result, const Tensor & self, const Tensor & mat2) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.size(0),self.size(1),mat2.size(2) });
    result.zero_();
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CPUShortTensor>(mat2.pImpl,"mat2",2, false);
    THShortTensor_baddbmm(result_->tensor, int16_t(0), result_->tensor, int16_t(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CPUShortType::bmm(const Tensor & self, const Tensor & mat2) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.size(0),self.size(1),mat2.size(2) });
    result.zero_();
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CPUShortTensor>(mat2.pImpl,"mat2",2, false);
    THShortTensor_baddbmm(result_->tensor, int16_t(0), result_->tensor, int16_t(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CPUShortType::s_addbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toShort();
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toShort();
    auto batch1_ = checked_cast_tensor<CPUShortTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CPUShortTensor>(batch2.pImpl,"batch2",5, false);
    THShortTensor_addbmm(result_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor CPUShortType::s_addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toShort();
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toShort();
    auto batch1_ = checked_cast_tensor<CPUShortTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CPUShortTensor>(batch2.pImpl,"batch2",5, false);
    THShortTensor_addbmm(result_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor & CPUShortType::addbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toShort();
    auto alpha_ = alpha.toShort();
    auto batch1_ = checked_cast_tensor<CPUShortTensor>(batch1.pImpl,"batch1",5, false);
    auto batch2_ = checked_cast_tensor<CPUShortTensor>(batch2.pImpl,"batch2",6, false);
    THShortTensor_addbmm(self_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    return self;
}
Tensor & CPUShortType::s_baddbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toShort();
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toShort();
    auto batch1_ = checked_cast_tensor<CPUShortTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CPUShortTensor>(batch2.pImpl,"batch2",5, false);
    THShortTensor_baddbmm(result_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor CPUShortType::s_baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toShort();
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toShort();
    auto batch1_ = checked_cast_tensor<CPUShortTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CPUShortTensor>(batch2.pImpl,"batch2",5, false);
    THShortTensor_baddbmm(result_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor & CPUShortType::baddbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toShort();
    auto alpha_ = alpha.toShort();
    auto batch1_ = checked_cast_tensor<CPUShortTensor>(batch1.pImpl,"batch1",5, false);
    auto batch2_ = checked_cast_tensor<CPUShortTensor>(batch2.pImpl,"batch2",6, false);
    THShortTensor_baddbmm(self_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    return self;
}
Tensor & CPUShortType::s_addcmul_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toShort();
    auto tensor1_ = checked_cast_tensor<CPUShortTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CPUShortTensor>(tensor2.pImpl,"tensor2",4, false);
    THShortTensor_addcmul(result_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor CPUShortType::s_addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toShort();
    auto tensor1_ = checked_cast_tensor<CPUShortTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CPUShortTensor>(tensor2.pImpl,"tensor2",4, false);
    THShortTensor_addcmul(result_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor & CPUShortType::s_addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toShort();
    auto tensor1_ = checked_cast_tensor<CPUShortTensor>(tensor1.pImpl,"tensor1",4, false);
    auto tensor2_ = checked_cast_tensor<CPUShortTensor>(tensor2.pImpl,"tensor2",5, false);
    THShortTensor_addcmul(self_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    return self;
}
Tensor & CPUShortType::s_addcdiv_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toShort();
    auto tensor1_ = checked_cast_tensor<CPUShortTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CPUShortTensor>(tensor2.pImpl,"tensor2",4, false);
    THShortTensor_addcdiv(result_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor CPUShortType::s_addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toShort();
    auto tensor1_ = checked_cast_tensor<CPUShortTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CPUShortTensor>(tensor2.pImpl,"tensor2",4, false);
    THShortTensor_addcdiv(result_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor & CPUShortType::s_addcdiv_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toShort();
    auto tensor1_ = checked_cast_tensor<CPUShortTensor>(tensor1.pImpl,"tensor1",4, false);
    auto tensor2_ = checked_cast_tensor<CPUShortTensor>(tensor2.pImpl,"tensor2",5, false);
    THShortTensor_addcdiv(self_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    return self;
}
Tensor & CPUShortType::random_(Tensor & self, int64_t from, int64_t to, Generator * generator) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    THShortTensor_clampedRandom(self_->tensor, generator_->generator, from, to);
    return self;
}
Tensor & CPUShortType::random_(Tensor & self, int64_t to, Generator * generator) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    THShortTensor_cappedRandom(self_->tensor, generator_->generator, to);
    return self;
}
Tensor & CPUShortType::random_(Tensor & self, Generator * generator) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    THShortTensor_random(self_->tensor, generator_->generator);
    return self;
}
Tensor & CPUShortType::geometric_(Tensor & self, double p, Generator * generator) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    THShortTensor_geometric(self_->tensor, generator_->generator, p);
    return self;
}
Tensor CPUShortType::tensor(Storage & storage, int64_t storageOffset, IntList size, IntList stride) const {
    auto storage_ = checked_cast_storage<CPUShortStorage>(&storage,"storage",1);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    return Tensor((new CPUShortTensor(context, THShortTensor_newWithStorage(storage_->storage, storageOffset, size_, stride_)))->maybeScalar(size.size() == 0),false);
}
Tensor CPUShortType::tensor(IntList size) const {
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new CPUShortTensor(context, THShortTensor_newWithSize(size_, NULL)))->maybeScalar(size.size() == 0),false);
}
Tensor CPUShortType::tensor(IntList size, IntList stride) const {
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    return Tensor((new CPUShortTensor(context, THShortTensor_newWithSize(size_, stride_)))->maybeScalar(size.size() == 0),false);
}
Tensor CPUShortType::tensor() const {
    return Tensor((new CPUShortTensor(context, THShortTensor_new())),false);
}
Tensor CPUShortType::alias(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPUShortTensor(context, THShortTensor_newWithTensor(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor & CPUShortType::as_strided_out(Tensor & result, const Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    auto result_ = checked_cast_tensor<CPUShortTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    if (storage_offset == -1) {
      storage_offset = self_->tensor->storageOffset;
    }
    THShortTensor_setStorage(result_->tensor, self_->tensor->storage, storage_offset, size_, stride_);
    result_->maybeScalar(size.size() == 0);
    return result;
}
Tensor CPUShortType::as_strided(const Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    auto result_ = new CPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    if (storage_offset == -1) {
      storage_offset = self_->tensor->storageOffset;
    }
    THShortTensor_setStorage(result_->tensor, self_->tensor->storage, storage_offset, size_, stride_);
    result_->maybeScalar(size.size() == 0);
    return result;
}
Tensor & CPUShortType::as_strided_(Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    if (storage_offset == -1) {
      storage_offset = self_->tensor->storageOffset;
    }
    THShortTensor_setStorage(self_->tensor, self_->tensor->storage, storage_offset, size_, stride_);
    self_->maybeScalar(size.size() == 0);
    return self;
}
Tensor & CPUShortType::_cat_out(Tensor & self, TensorList tensors, int64_t dim) const {
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",0, false);
    auto tensors_ = tensor_list_checked_cast<CPUShortTensor, Tensor, THShortTensor>(tensors,"tensors",1);
    THShortTensor_catArray(self_->tensor, tensors_.data(), tensors_.size(), dim);
    return self;
}
Tensor CPUShortType::_cat(TensorList tensors, int64_t dim) const {
    auto self_ = new CPUShortTensor(context);
    auto self = Tensor(self_, false);
    auto tensors_ = tensor_list_checked_cast<CPUShortTensor, Tensor, THShortTensor>(tensors,"tensors",1);
    THShortTensor_catArray(self_->tensor, tensors_.data(), tensors_.size(), dim);
    return self;
}
Tensor CPUShortType::_sparse_mask(const Tensor & self, SparseTensor mask) const {
    auto result_ = new SparseCPUShortTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUShortTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<SparseCPUShortTensor>(mask.tref.pImpl,"mask",2,false);
    THShortTensor_sparseMask(result_->tensor, self_->tensor, mask_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUShortType::abs_out(Tensor & result, const Tensor & self) const {
    return  at::native::_abs_out_cpu(result, self);
}
Tensor & CPUShortType::ceil_out(Tensor & result, const Tensor & self) const {
    return  at::native::_ceil_out_cpu(result, self);
}
Tensor & CPUShortType::cos_out(Tensor & result, const Tensor & self) const {
    return  at::native::_cos_out_cpu(result, self);
}
Tensor CPUShortType::embedding_dense_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) const {
    return  at::native::embedding_backward_cpu(grad, indices, num_weights, padding_idx, scale_grad_by_freq);
}
Tensor & CPUShortType::embedding_renorm_(Tensor & self, const Tensor & indices, double max_norm, double norm_type) const {
    return  at::native::embedding_renorm_cpu_(self, indices, max_norm, norm_type);
}
std::tuple<Tensor,Tensor,Tensor> CPUShortType::embedding_bag(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse) const {
    return  at::native::embedding_bag_cpu(weight, indices, offsets, scale_grad_by_freq, mode, sparse);
}
Tensor CPUShortType::embedding_bag_dense_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode) const {
    return  at::native::embedding_bag_backward_cpu(grad, indices, offsets, offset2bag, bag_size, num_weights, scale_grad_by_freq, mode);
}
Tensor & CPUShortType::exp_out(Tensor & result, const Tensor & self) const {
    return  at::native::_exp_out_cpu(result, self);
}
Tensor & CPUShortType::eye_out(Tensor & result, int64_t n, int64_t m) const {
    return  at::native::eye_out_cpu(result, n, m);
}
Tensor & CPUShortType::floor_out(Tensor & result, const Tensor & self) const {
    return  at::native::_floor_out_cpu(result, self);
}
Tensor CPUShortType::_fft_with_size(const Tensor & self, int64_t signal_ndim, bool complex_input, bool complex_output, bool inverse, IntList checked_signal_sizes, bool normalized, bool onesided, IntList output_sizes) const {
    return  at::native::_fft_mkl(self, signal_ndim, complex_input, complex_output, inverse, checked_signal_sizes, normalized, onesided, output_sizes);
}
Tensor & CPUShortType::log_out(Tensor & result, const Tensor & self) const {
    return  at::native::_log_out_cpu(result, self);
}
std::tuple<Tensor,Tensor> CPUShortType::RoiPooling2d_forward(const Tensor & input, const Tensor & rois, int64_t pooledHeight, int64_t pooledWidth, double spatialScale) const {
    return  at::native::RoiPooling2d_forward_cpu(input, rois, pooledHeight, pooledWidth, spatialScale);
}
Tensor CPUShortType::RoiPooling2d_backward(const Tensor & input, const Tensor & rois, int64_t pooledHeight, int64_t pooledWidth, double spatialScale, const Tensor & gradOutput, const Tensor & argmaxes) const {
    return  at::native::RoiPooling2d_backward_cpu(input, rois, pooledHeight, pooledWidth, spatialScale, gradOutput, argmaxes);
}
Tensor & CPUShortType::round_out(Tensor & result, const Tensor & self) const {
    return  at::native::_round_out_cpu(result, self);
}
Tensor & CPUShortType::sin_out(Tensor & result, const Tensor & self) const {
    return  at::native::_sin_out_cpu(result, self);
}
Tensor & CPUShortType::sspaddmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return  at::native::_sspaddmm_out_only_sparse(result, self, mat1, mat2, beta, alpha);
}
Tensor CPUShortType::sum(const Tensor & self) const {
    return  at::native::_sum_cpu(self);
}
Tensor & CPUShortType::sum_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    return  at::native::_sum_out_cpu(result, self, dim, keepdim);
}
Tensor & CPUShortType::sqrt_out(Tensor & result, const Tensor & self) const {
    return  at::native::_sqrt_out_cpu(result, self);
}
Tensor CPUShortType::prod(const Tensor & self) const {
    return  at::native::_prod_cpu(self);
}
Tensor & CPUShortType::prod_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    return  at::native::_prod_out_cpu(result, self, dim, keepdim);
}
Tensor & CPUShortType::trunc_out(Tensor & result, const Tensor & self) const {
    return  at::native::_trunc_out_cpu(result, self);
}
std::tuple<Tensor,Tensor> CPUShortType::_unique(const Tensor & self, bool sorted, bool return_inverse) const {
    return  at::native::_unique_cpu(self, sorted, return_inverse);
}
Tensor CPUShortType::_s_where(const Tensor & condition, const Tensor & self, const Tensor & other) const {
    return  at::native::_s_where_cpu(condition, self, other);
}
Tensor CPUShortType::_standard_gamma_grad(const Tensor & self, const Tensor & output) const {
    return  at::native::_standard_gamma_grad_cpu(self, output);
}
Tensor CPUShortType::poisson(const Tensor & self, Generator * generator) const {
    return  at::native::_s_poisson_cpu(self, generator);
}

}
