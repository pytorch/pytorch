// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/CPUCharType.h"
#include "ATen/CPUCharStorage.h"
#include "ATen/CPUCharTensor.h"
#include "ATen/CPUGenerator.h"
#include "ATen/CPUByteTensor.h"
#include "ATen/CPUIntTensor.h"
#include "ATen/CPULongTensor.h"
#include "ATen/SparseCPUCharTensor.h"
#include "ATen/CPUCharTensor.h"
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

CPUCharType::CPUCharType(Context* context)
: Type(context, /*is_variable_or_undefined=*/false) {}
ScalarType CPUCharType::scalarType() const {
  return ScalarType::Char;
}
Backend CPUCharType::backend() const {
  return Backend::CPU;
}
bool CPUCharType::is_cuda() const { return backend() == kCUDA || backend() == kSparseCUDA; }
bool CPUCharType::is_sparse() const { return backend() == kSparseCPU || backend() == kSparseCUDA; }
bool CPUCharType::is_distributed() const { return false; }

std::unique_ptr<Storage> CPUCharType::storage() const {
  return std::unique_ptr<Storage>(new CPUCharStorage(context));
}
std::unique_ptr<Storage> CPUCharType::storage(size_t size) const {
  return std::unique_ptr<Storage>(new CPUCharStorage(context,size));
}
std::unique_ptr<Storage> CPUCharType::storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const {
    return std::unique_ptr<Storage>(
      new CPUCharStorage(context,data,size,deleter));
}
std::unique_ptr<Storage> CPUCharType::storageWithAllocator(int64_t size, std::unique_ptr<Allocator> allocator) const {
    return std::unique_ptr<Storage>(
        new CPUCharStorage(context, size, std::move(allocator)));
}
Tensor CPUCharType::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THCharTensor_retain( (THCharTensor*) th_pointer);
  return Tensor(new CPUCharTensor(context,(THCharTensor*)(th_pointer)), false);
}
std::unique_ptr<Storage> CPUCharType::unsafeStorageFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THCharStorage_retain( (THCharStorage*) th_pointer);
  return std::unique_ptr<Storage>(new CPUCharStorage(context, (THCharStorage*) th_pointer));
}
std::unique_ptr<Generator> CPUCharType::generator() const {
  return std::unique_ptr<Generator>(new CPUGenerator(context));
}

const char * CPUCharType::toString() const {
  return CPUCharType::typeString();
}
TypeID CPUCharType::ID() const {
  return TypeID::CPUChar;
}

std::size_t CPUCharType::elementSizeInBytes() const {
  return sizeof(int8_t);
}

const char * CPUCharType::typeString() {
  return "CPUCharType";
}

/* example
Tensor * CPUCharType::add(Tensor & a, Tensor & b) {
  std::cout << "add CPUCharTensor\n";
  return &a;
}
*/

int64_t CPUCharType::storage_offset(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    return static_cast<int64_t>(THCharTensor_storageOffset(self_->tensor));
}
Tensor & CPUCharType::resize_(Tensor & self, IntList size) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    THCharTensor_resize(self_->tensor, size_, NULL);
    self_->maybeScalar(size.size() == 0);
    return self;
}
int64_t CPUCharType::numel(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    return static_cast<int64_t>(THCharTensor_nElement(self_->tensor));
}
Tensor & CPUCharType::set_(Tensor & self, Storage & storage) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto storage_ = checked_cast_storage<CPUCharStorage>(&storage,"storage",2);
    THCharTensor_setStorage(self_->tensor, storage_->storage, 0, THLongStorageView::makeFromLength(static_cast<int64_t>(storage.size())), NULL);
    self_->maybeScalar(false);
    return self;
}
Tensor & CPUCharType::set_(Tensor & self, Storage & sourceStorage, int64_t storage_offset, IntList size, IntList stride) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto sourceStorage_ = checked_cast_storage<CPUCharStorage>(&sourceStorage,"sourceStorage",2);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    THCharTensor_setStorage(self_->tensor, sourceStorage_->storage, storage_offset, size_, stride_);
    self_->maybeScalar(size.size() == 0);
    return self;
}
Tensor & CPUCharType::set_(Tensor & self, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto source_ = checked_cast_tensor<CPUCharTensor>(source.pImpl,"source",2, false);
    THCharTensor_set(self_->tensor, source_->tensor);
    self_->maybeScalar(source_->isScalar());
    return self;
}
Tensor & CPUCharType::set_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    THCharTensor_setStorage(self_->tensor, NULL, 0, NULL, NULL);
    self_->maybeScalar(false);
    return self;
}
Tensor & CPUCharType::fill_(Tensor & self, Scalar value) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toChar();
    THCharTensor_fill(self_->tensor, value_);
    return self;
}
Tensor & CPUCharType::fill_(Tensor & self, const Tensor & value) const {
    if (value.dim() == 0) {
        return static_cast<const Type*>(this)->fill_(self, Scalar(value));
    }
    AT_ERROR("fill_ only supports a 0-dimensional value tensor, but got tensor "
        "with %" PRId64 " dimension(s)", value.dim());
}
bool CPUCharType::is_contiguous(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    return THCharTensor_isContiguous(self_->tensor);
}
bool CPUCharType::is_set_to(const Tensor & self, const Tensor & tensor) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto tensor_ = checked_cast_tensor<CPUCharTensor>(tensor.pImpl,"tensor",2, false);
    return THCharTensor_isSetTo(self_->tensor, tensor_->tensor);
}
Tensor & CPUCharType::s_masked_fill_(Tensor & self, const Tensor & mask, Scalar value) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CPUByteTensor>(mask.pImpl,"mask",2, false);
    auto value_ = value.toChar();
    THCharTensor_maskedFill(self_->tensor, mask_->tensor, value_);
    return self;
}
Tensor & CPUCharType::s_masked_fill_(Tensor & self, const Tensor & mask, const Tensor & value) const {
    if (value.dim() == 0) {
        return static_cast<const Type*>(this)->masked_fill_(self, mask, Scalar(value));
    }
    AT_ERROR("masked_fill_ only supports a 0-dimensional value tensor, but got tensor "
        "with %" PRId64 " dimension(s)", value.dim());
}
Tensor & CPUCharType::s_masked_scatter_(Tensor & self, const Tensor & mask, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CPUByteTensor>(mask.pImpl,"mask",2, false);
    auto source_ = checked_cast_tensor<CPUCharTensor>(source.pImpl,"source",3, false);
    THCharTensor_maskedCopy(self_->tensor, mask_->tensor, source_->tensor);
    return self;
}
Tensor & CPUCharType::s_masked_select_out(Tensor & result, const Tensor & self, const Tensor & mask) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CPUByteTensor>(mask.pImpl,"mask",2, false);
    THCharTensor_maskedSelect(result_->tensor, self_->tensor, mask_->tensor);
    result_->maybeScalar(self_->isScalar() && mask_->isScalar());
    return result;
}
Tensor CPUCharType::s_masked_select(const Tensor & self, const Tensor & mask) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CPUByteTensor>(mask.pImpl,"mask",2, false);
    THCharTensor_maskedSelect(result_->tensor, self_->tensor, mask_->tensor);
    result_->maybeScalar(self_->isScalar() && mask_->isScalar());
    return result;
}
Tensor CPUCharType::transpose(const Tensor & self, int64_t dim0, int64_t dim1) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    dim0 = maybe_wrap_dim(dim0, self_);
    dim1 = maybe_wrap_dim(dim1, self_);
    return Tensor((new CPUCharTensor(context, THCharTensor_newTranspose(self_->tensor, dim0, dim1)))->maybeScalar(self_->isScalar()),false);
}
Tensor CPUCharType::t(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPUCharTensor(context, THCharTensor_newTranspose(self_->tensor, 0, 1)))->maybeScalar(self_->isScalar()),false);
}
Tensor & CPUCharType::nonzero_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    THCharTensor_nonzero(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUCharType::nonzero(const Tensor & self) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    THCharTensor_nonzero(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUCharType::contiguous(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPUCharTensor(context, THCharTensor_newContiguous(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor CPUCharType::clone(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPUCharTensor(context, THCharTensor_newClone(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor CPUCharType::view(const Tensor & self, IntList size) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new CPUCharTensor(context, THCharTensor_newView(self_->tensor, size_)))->maybeScalar(size.size() == 0),false);
}
Tensor & CPUCharType::resize_as_(Tensor & self, const Tensor & the_template) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto the_template_ = checked_cast_tensor<CPUCharTensor>(the_template.pImpl,"the_template",2, false);
    THCharTensor_resizeAs(self_->tensor, the_template_->tensor);
    self_->maybeScalar(the_template_->isScalar());
    return self;
}
Tensor & CPUCharType::index_select_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    THCharTensor_indexSelect(result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
Tensor CPUCharType::index_select(const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    THCharTensor_indexSelect(result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
Tensor & CPUCharType::_indexCopy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    auto source_ = checked_cast_tensor<CPUCharTensor>(source.pImpl,"source",4, false);
    THCharTensor_indexCopy(self_->tensor, dim, index_->tensor, source_->tensor);
    return self;
}
Tensor & CPUCharType::take_out(Tensor & result, const Tensor & self, const Tensor & index) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",2, false);
    THCharTensor_take(result_->tensor, self_->tensor, index_->tensor);
    result_->maybeScalar(index_->isScalar());
    return result;
}
Tensor CPUCharType::take(const Tensor & self, const Tensor & index) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",2, false);
    THCharTensor_take(result_->tensor, self_->tensor, index_->tensor);
    result_->maybeScalar(index_->isScalar());
    return result;
}
Tensor & CPUCharType::put_(Tensor & self, const Tensor & index, const Tensor & source, bool accumulate) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",2, false);
    auto source_ = checked_cast_tensor<CPUCharTensor>(source.pImpl,"source",3, false);
    THCharTensor_put(self_->tensor, index_->tensor, source_->tensor, accumulate);
    return self;
}
Tensor & CPUCharType::index_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    auto source_ = checked_cast_tensor<CPUCharTensor>(source.pImpl,"source",4, false);
    THCharTensor_indexAdd(self_->tensor, dim, index_->tensor, source_->tensor);
    return self;
}
Tensor & CPUCharType::index_fill_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    auto value_ = value.toChar();
    THCharTensor_indexFill(self_->tensor, dim, index_->tensor, value_);
    return self;
}
Tensor & CPUCharType::index_fill_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & value) const {
    if (value.dim() == 0) {
        return static_cast<const Type*>(this)->index_fill_(self, dim, index, Scalar(value));
    }
    AT_ERROR("index_fill_ only supports a 0-dimensional value tensor, but got tensor "
        "with %" PRId64 " dimension(s)", value.dim());
}
Tensor CPUCharType::unfold(const Tensor & self, int64_t dimension, int64_t size, int64_t step) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    dimension = maybe_wrap_dim(dimension, self_);
    THCharTensor_unfold(result_->tensor, self_->tensor, dimension, size, step);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUCharType::_range_out(Tensor & result, Scalar start, Scalar end, Scalar step) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto start_ = start.toLong();
    auto end_ = end.toLong();
    auto step_ = step.toLong();
    THCharTensor_range(result_->tensor, start_, end_, step_);
    return result;
}
Tensor CPUCharType::_range(Scalar start, Scalar end, Scalar step) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto start_ = start.toLong();
    auto end_ = end.toLong();
    auto step_ = step.toLong();
    THCharTensor_range(result_->tensor, start_, end_, step_);
    return result;
}
Tensor & CPUCharType::_arange_out(Tensor & result, Scalar start, Scalar end, Scalar step) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto start_ = start.toLong();
    auto end_ = end.toLong();
    auto step_ = step.toLong();
    THCharTensor_arange(result_->tensor, start_, end_, step_);
    return result;
}
Tensor CPUCharType::_arange(Scalar start, Scalar end, Scalar step) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto start_ = start.toLong();
    auto end_ = end.toLong();
    auto step_ = step.toLong();
    THCharTensor_arange(result_->tensor, start_, end_, step_);
    return result;
}
Tensor & CPUCharType::_arange_out(Tensor & result, Scalar end) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto end_ = end.toLong();
    THCharTensor_arange(result_->tensor, 0, end_, 1);
    return result;
}
Tensor CPUCharType::_arange(Scalar end) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto end_ = end.toLong();
    THCharTensor_arange(result_->tensor, 0, end_, 1);
    return result;
}
Tensor & CPUCharType::scatter_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
    if (src.dim() == 0) {
        return static_cast<const Type*>(this)->scatter_(self, dim, index, Scalar(src));
    }
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    auto src_ = checked_cast_tensor<CPUCharTensor>(src.pImpl,"src",4, false);
    THCharTensor_scatter(self_->tensor, dim, index_->tensor, src_->tensor);
    return self;
}
Tensor & CPUCharType::scatter_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    auto value_ = value.toChar();
    THCharTensor_scatterFill(self_->tensor, dim, index_->tensor, value_);
    return self;
}
Tensor & CPUCharType::scatter_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    auto src_ = checked_cast_tensor<CPUCharTensor>(src.pImpl,"src",4, false);
    THCharTensor_scatterAdd(self_->tensor, dim, index_->tensor, src_->tensor);
    return self;
}
Tensor & CPUCharType::gather_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    result.resize_(index.sizes());
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    THCharTensor_gather(result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
Tensor CPUCharType::gather(const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    result.resize_(index.sizes());
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    THCharTensor_gather(result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
void* CPUCharType::data_ptr(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    return THCharTensor_data(self_->tensor);
}
bool CPUCharType::equal(const Tensor & self, const Tensor & other) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",2, false);
    return THCharTensor_equal(self_->tensor, other_->tensor);
}
Tensor & CPUCharType::__and___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_bitand(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUCharType::__and__(const Tensor & self, Scalar other) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_bitand(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUCharType::s___and___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__and___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",2, false);
    THCharTensor_cbitand(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUCharType::s___and__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__and__(self, Scalar(other));
    }
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",2, false);
    THCharTensor_cbitand(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUCharType::__iand__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_bitand(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUCharType::s___iand__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__iand__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",3, false);
    THCharTensor_cbitand(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUCharType::__or___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_bitor(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUCharType::__or__(const Tensor & self, Scalar other) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_bitor(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUCharType::s___or___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__or___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",2, false);
    THCharTensor_cbitor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUCharType::s___or__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__or__(self, Scalar(other));
    }
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",2, false);
    THCharTensor_cbitor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUCharType::__ior__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_bitor(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUCharType::s___ior__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__ior__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",3, false);
    THCharTensor_cbitor(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUCharType::__xor___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_bitxor(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUCharType::__xor__(const Tensor & self, Scalar other) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_bitxor(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUCharType::s___xor___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__xor___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",2, false);
    THCharTensor_cbitxor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUCharType::s___xor__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__xor__(self, Scalar(other));
    }
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",2, false);
    THCharTensor_cbitxor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUCharType::__ixor__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_bitxor(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUCharType::s___ixor__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__ixor__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",3, false);
    THCharTensor_cbitxor(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUCharType::__lshift___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_lshift(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUCharType::__lshift__(const Tensor & self, Scalar other) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_lshift(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUCharType::s___lshift___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__lshift___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",2, false);
    THCharTensor_clshift(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUCharType::s___lshift__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__lshift__(self, Scalar(other));
    }
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",2, false);
    THCharTensor_clshift(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUCharType::__ilshift__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_lshift(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUCharType::s___ilshift__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__ilshift__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",3, false);
    THCharTensor_clshift(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUCharType::__rshift___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_rshift(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUCharType::__rshift__(const Tensor & self, Scalar other) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_rshift(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUCharType::s___rshift___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__rshift___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",2, false);
    THCharTensor_crshift(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUCharType::s___rshift__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__rshift__(self, Scalar(other));
    }
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",2, false);
    THCharTensor_crshift(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUCharType::__irshift__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_rshift(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUCharType::s___irshift__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__irshift__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",3, false);
    THCharTensor_crshift(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUCharType::lt_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_ltValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUCharType::lt(const Tensor & self, Scalar other) const {
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_ltValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUCharType::s_lt_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->lt_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",2, false);
    THCharTensor_ltTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUCharType::s_lt(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->lt(self, Scalar(other));
    }
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",2, false);
    THCharTensor_ltTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUCharType::lt_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_ltValueT(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUCharType::s_lt_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->lt_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",3, false);
    THCharTensor_ltTensorT(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUCharType::gt_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_gtValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUCharType::gt(const Tensor & self, Scalar other) const {
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_gtValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUCharType::s_gt_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->gt_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",2, false);
    THCharTensor_gtTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUCharType::s_gt(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->gt(self, Scalar(other));
    }
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",2, false);
    THCharTensor_gtTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUCharType::gt_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_gtValueT(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUCharType::s_gt_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->gt_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",3, false);
    THCharTensor_gtTensorT(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUCharType::le_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_leValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUCharType::le(const Tensor & self, Scalar other) const {
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_leValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUCharType::s_le_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->le_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",2, false);
    THCharTensor_leTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUCharType::s_le(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->le(self, Scalar(other));
    }
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",2, false);
    THCharTensor_leTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUCharType::le_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_leValueT(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUCharType::s_le_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->le_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",3, false);
    THCharTensor_leTensorT(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUCharType::ge_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_geValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUCharType::ge(const Tensor & self, Scalar other) const {
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_geValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUCharType::s_ge_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ge_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",2, false);
    THCharTensor_geTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUCharType::s_ge(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ge(self, Scalar(other));
    }
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",2, false);
    THCharTensor_geTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUCharType::ge_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_geValueT(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUCharType::s_ge_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ge_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",3, false);
    THCharTensor_geTensorT(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUCharType::eq_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_eqValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUCharType::eq(const Tensor & self, Scalar other) const {
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_eqValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUCharType::s_eq_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->eq_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",2, false);
    THCharTensor_eqTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUCharType::s_eq(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->eq(self, Scalar(other));
    }
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",2, false);
    THCharTensor_eqTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUCharType::eq_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_eqValueT(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUCharType::s_eq_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->eq_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",3, false);
    THCharTensor_eqTensorT(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUCharType::ne_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_neValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUCharType::ne(const Tensor & self, Scalar other) const {
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_neValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUCharType::s_ne_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ne_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",2, false);
    THCharTensor_neTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUCharType::s_ne(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ne(self, Scalar(other));
    }
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",2, false);
    THCharTensor_neTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUCharType::ne_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_neValueT(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUCharType::s_ne_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ne_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",3, false);
    THCharTensor_neTensorT(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
std::tuple<Tensor &,Tensor &> CPUCharType::min_out(Tensor & min, Tensor & min_indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto min_ = checked_cast_tensor<CPUCharTensor>(min.pImpl,"min",0, false);
    auto min_indices_ = checked_cast_tensor<CPULongTensor>(min_indices.pImpl,"min_indices",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCharTensor_min(min_->tensor, min_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    min_->maybeScalar(maybe_scalar);
    min_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(min, min_indices);
}
std::tuple<Tensor,Tensor> CPUCharType::min(const Tensor & self, int64_t dim, bool keepdim) const {
    auto min_ = new CPUCharTensor(context);
    auto min = Tensor(min_, false);
    auto min_indices_ = new CPULongTensor(context);
    auto min_indices = Tensor(min_indices_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCharTensor_min(min_->tensor, min_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    min_->maybeScalar(maybe_scalar);
    min_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(min, min_indices);
}
Tensor & CPUCharType::s_min_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",2, false);
    THCharTensor_cmin(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUCharType::s_min(const Tensor & self, const Tensor & other) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",2, false);
    THCharTensor_cmin(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUCharType::min(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<int8_t>(THCharTensor_minall(self_->tensor)));
}
std::tuple<Tensor &,Tensor &> CPUCharType::max_out(Tensor & max, Tensor & max_indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto max_ = checked_cast_tensor<CPUCharTensor>(max.pImpl,"max",0, false);
    auto max_indices_ = checked_cast_tensor<CPULongTensor>(max_indices.pImpl,"max_indices",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCharTensor_max(max_->tensor, max_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    max_->maybeScalar(maybe_scalar);
    max_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(max, max_indices);
}
std::tuple<Tensor,Tensor> CPUCharType::max(const Tensor & self, int64_t dim, bool keepdim) const {
    auto max_ = new CPUCharTensor(context);
    auto max = Tensor(max_, false);
    auto max_indices_ = new CPULongTensor(context);
    auto max_indices = Tensor(max_indices_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCharTensor_max(max_->tensor, max_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    max_->maybeScalar(maybe_scalar);
    max_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(max, max_indices);
}
Tensor & CPUCharType::s_max_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",2, false);
    THCharTensor_cmax(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUCharType::s_max(const Tensor & self, const Tensor & other) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",2, false);
    THCharTensor_cmax(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUCharType::max(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<int8_t>(THCharTensor_maxall(self_->tensor)));
}
std::tuple<Tensor &,Tensor &> CPUCharType::kthvalue_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool keepdim) const {
    auto values_ = checked_cast_tensor<CPUCharTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCharTensor_kthvalue(values_->tensor, indices_->tensor, self_->tensor, k, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CPUCharType::kthvalue(const Tensor & self, int64_t k, int64_t dim, bool keepdim) const {
    auto values_ = new CPUCharTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CPULongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCharTensor_kthvalue(values_->tensor, indices_->tensor, self_->tensor, k, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
std::tuple<Tensor &,Tensor &> CPUCharType::mode_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = checked_cast_tensor<CPUCharTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCharTensor_mode(values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CPUCharType::mode(const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = new CPUCharTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CPULongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCharTensor_mode(values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
std::tuple<Tensor &,Tensor &> CPUCharType::median_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = checked_cast_tensor<CPUCharTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCharTensor_median(values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CPUCharType::median(const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = new CPUCharTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CPULongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCharTensor_median(values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
Tensor CPUCharType::median(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<int8_t>(THCharTensor_medianall(self_->tensor)));
}
std::tuple<Tensor &,Tensor &> CPUCharType::sort_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool descending) const {
    auto values_ = checked_cast_tensor<CPUCharTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCharTensor_sort(values_->tensor, indices_->tensor, self_->tensor, dim, descending);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CPUCharType::sort(const Tensor & self, int64_t dim, bool descending) const {
    auto values_ = new CPUCharTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CPULongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCharTensor_sort(values_->tensor, indices_->tensor, self_->tensor, dim, descending);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
std::tuple<Tensor &,Tensor &> CPUCharType::topk_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) const {
    auto values_ = checked_cast_tensor<CPUCharTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCharTensor_topk(values_->tensor, indices_->tensor, self_->tensor, k, dim, largest, sorted);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CPUCharType::topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) const {
    auto values_ = new CPUCharTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CPULongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCharTensor_topk(values_->tensor, indices_->tensor, self_->tensor, k, dim, largest, sorted);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
Tensor & CPUCharType::neg_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    THCharTensor_neg(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUCharType::neg(const Tensor & self) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    THCharTensor_neg(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUCharType::neg_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    THCharTensor_neg(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUCharType::pow_out(Tensor & result, const Tensor & self, Scalar exponent) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toChar();
    THCharTensor_pow(result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUCharType::pow(const Tensor & self, Scalar exponent) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toChar();
    THCharTensor_pow(result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUCharType::s_pow_out(Tensor & result, const Tensor & self, const Tensor & exponent) const {
    if (exponent.dim() == 0) {
        return static_cast<const Type*>(this)->pow_out(result, self, Scalar(exponent));
    }
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto exponent_ = checked_cast_tensor<CPUCharTensor>(exponent.pImpl,"exponent",2, false);
    THCharTensor_cpow(result_->tensor, self_->tensor, exponent_->tensor);
    result_->maybeScalar(self_->isScalar() && exponent_->isScalar());
    return result;
}
Tensor CPUCharType::s_pow(const Tensor & self, const Tensor & exponent) const {
    if (exponent.dim() == 0) {
        return static_cast<const Type*>(this)->pow(self, Scalar(exponent));
    }
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto exponent_ = checked_cast_tensor<CPUCharTensor>(exponent.pImpl,"exponent",2, false);
    THCharTensor_cpow(result_->tensor, self_->tensor, exponent_->tensor);
    result_->maybeScalar(self_->isScalar() && exponent_->isScalar());
    return result;
}
Tensor & CPUCharType::pow_out(Tensor & result, Scalar base, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto base_ = base.toChar();
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",2, false);
    THCharTensor_tpow(result_->tensor, base_, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUCharType::pow(Scalar base, const Tensor & self) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto base_ = base.toChar();
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",2, false);
    THCharTensor_tpow(result_->tensor, base_, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUCharType::pow_(Tensor & self, Scalar exponent) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toChar();
    THCharTensor_pow(self_->tensor, self_->tensor, exponent_);
    return self;
}
Tensor & CPUCharType::s_pow_(Tensor & self, const Tensor & exponent) const {
    if (exponent.dim() == 0) {
        return static_cast<const Type*>(this)->pow_(self, Scalar(exponent));
    }
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto exponent_ = checked_cast_tensor<CPUCharTensor>(exponent.pImpl,"exponent",3, false);
    THCharTensor_cpow(self_->tensor, self_->tensor, exponent_->tensor);
    return self;
}
Tensor & CPUCharType::zero_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    THCharTensor_zero(self_->tensor);
    return self;
}
Tensor CPUCharType::_sumall(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<int8_t>(THCharTensor_sumall(self_->tensor)));
}
Tensor & CPUCharType::_sum_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCharTensor_sum(result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CPUCharType::_sum(const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCharTensor_sum(result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CPUCharType::_prodall(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<int8_t>(THCharTensor_prodall(self_->tensor)));
}
Tensor & CPUCharType::_prod_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCharTensor_prod(result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CPUCharType::_prod(const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCharTensor_prod(result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor & CPUCharType::cumsum_out(Tensor & result, const Tensor & self, int64_t dim) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCharTensor_cumsum(result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUCharType::cumsum(const Tensor & self, int64_t dim) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCharTensor_cumsum(result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUCharType::cumprod_out(Tensor & result, const Tensor & self, int64_t dim) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCharTensor_cumprod(result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUCharType::cumprod(const Tensor & self, int64_t dim) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THCharTensor_cumprod(result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUCharType::sign_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    THCharTensor_sign(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUCharType::sign(const Tensor & self) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    THCharTensor_sign(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUCharType::sign_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    THCharTensor_sign(self_->tensor, self_->tensor);
    return self;
}
Tensor CPUCharType::trace(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<int8_t>(THCharTensor_trace(self_->tensor)));
}
Tensor & CPUCharType::add_out(Tensor & result, const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    auto alpha_ = alpha.toChar();
    THCharTensor_add_scaled(result_->tensor, self_->tensor, other_, alpha_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUCharType::add(const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    auto alpha_ = alpha.toChar();
    THCharTensor_add_scaled(result_->tensor, self_->tensor, other_, alpha_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUCharType::s_add_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if(other.type().is_sparse()) {
        return static_cast<const Type*>(this)->add_out(result, self, SparseTensor(other), alpha);
    }
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toChar();
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",3, false);
    THCharTensor_cadd(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUCharType::s_add(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if(other.type().is_sparse()) {
        return static_cast<const Type*>(this)->add(self, SparseTensor(other), alpha);
    }
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add(self, Scalar(other), alpha);
    }
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toChar();
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",3, false);
    THCharTensor_cadd(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUCharType::add_out(Tensor & result, const Tensor & self, SparseTensor other, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toChar();
    auto other_ = checked_cast_tensor<SparseCPUCharTensor>(other.tref.pImpl,"other",3,false);
    THSCharTensor_spcadd(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUCharType::add(const Tensor & self, SparseTensor other, Scalar alpha) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toChar();
    auto other_ = checked_cast_tensor<SparseCPUCharTensor>(other.tref.pImpl,"other",3,false);
    THSCharTensor_spcadd(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUCharType::add_(Tensor & self, Scalar other, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    auto alpha_ = alpha.toChar();
    THCharTensor_add_scaled(self_->tensor, self_->tensor, other_, alpha_);
    return self;
}
Tensor & CPUCharType::s_add_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if(other.type().is_sparse()) {
        return static_cast<const Type*>(this)->add_(self, SparseTensor(other), alpha);
    }
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toChar();
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",4, false);
    THCharTensor_cadd(self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & CPUCharType::add_(Tensor & self, SparseTensor other, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toChar();
    auto other_ = checked_cast_tensor<SparseCPUCharTensor>(other.tref.pImpl,"other",4,false);
    THSCharTensor_spcadd(self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & CPUCharType::sub_out(Tensor & result, const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    auto alpha_ = alpha.toChar();
    THCharTensor_sub_scaled(result_->tensor, self_->tensor, other_, alpha_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUCharType::sub(const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    auto alpha_ = alpha.toChar();
    THCharTensor_sub_scaled(result_->tensor, self_->tensor, other_, alpha_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUCharType::s_sub_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toChar();
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",3, false);
    THCharTensor_csub(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUCharType::s_sub(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub(self, Scalar(other), alpha);
    }
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toChar();
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",3, false);
    THCharTensor_csub(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUCharType::sub_(Tensor & self, Scalar other, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    auto alpha_ = alpha.toChar();
    THCharTensor_sub_scaled(self_->tensor, self_->tensor, other_, alpha_);
    return self;
}
Tensor & CPUCharType::s_sub_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toChar();
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",4, false);
    THCharTensor_csub(self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & CPUCharType::mul_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_mul(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUCharType::mul(const Tensor & self, Scalar other) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_mul(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUCharType::s_mul_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",2, false);
    THCharTensor_cmul(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUCharType::s_mul(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul(self, Scalar(other));
    }
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",2, false);
    THCharTensor_cmul(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUCharType::mul_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_mul(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUCharType::s_mul_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",3, false);
    THCharTensor_cmul(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUCharType::div_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_div(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUCharType::div(const Tensor & self, Scalar other) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_div(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUCharType::s_div_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->div_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",2, false);
    THCharTensor_cdiv(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUCharType::s_div(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->div(self, Scalar(other));
    }
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",2, false);
    THCharTensor_cdiv(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUCharType::div_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_div(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUCharType::s_div_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->div_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",3, false);
    THCharTensor_cdiv(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUCharType::fmod_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_fmod(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUCharType::fmod(const Tensor & self, Scalar other) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_fmod(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUCharType::s_fmod_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->fmod_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",2, false);
    THCharTensor_cfmod(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUCharType::s_fmod(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->fmod(self, Scalar(other));
    }
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",2, false);
    THCharTensor_cfmod(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUCharType::fmod_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_fmod(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUCharType::s_fmod_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->fmod_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",3, false);
    THCharTensor_cfmod(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUCharType::remainder_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_remainder(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUCharType::remainder(const Tensor & self, Scalar other) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_remainder(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUCharType::s_remainder_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->remainder_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",2, false);
    THCharTensor_cremainder(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUCharType::s_remainder(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->remainder(self, Scalar(other));
    }
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",2, false);
    THCharTensor_cremainder(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUCharType::remainder_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toChar();
    THCharTensor_remainder(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUCharType::s_remainder_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->remainder_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",3, false);
    THCharTensor_cremainder(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUCharType::clamp_out(Tensor & result, const Tensor & self, Scalar min, Scalar max) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toChar();
    auto max_ = max.toChar();
    THCharTensor_clamp(result_->tensor, self_->tensor, min_, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUCharType::clamp(const Tensor & self, Scalar min, Scalar max) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toChar();
    auto max_ = max.toChar();
    THCharTensor_clamp(result_->tensor, self_->tensor, min_, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUCharType::clamp_(Tensor & self, Scalar min, Scalar max) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toChar();
    auto max_ = max.toChar();
    THCharTensor_clamp(self_->tensor, self_->tensor, min_, max_);
    return self;
}
Tensor & CPUCharType::clamp_min_out(Tensor & result, const Tensor & self, Scalar min) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toChar();
    THCharTensor_cmaxValue(result_->tensor, self_->tensor, min_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUCharType::clamp_min(const Tensor & self, Scalar min) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toChar();
    THCharTensor_cmaxValue(result_->tensor, self_->tensor, min_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUCharType::clamp_min_(Tensor & self, Scalar min) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toChar();
    THCharTensor_cmaxValue(self_->tensor, self_->tensor, min_);
    return self;
}
Tensor & CPUCharType::clamp_max_out(Tensor & result, const Tensor & self, Scalar max) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto max_ = max.toChar();
    THCharTensor_cminValue(result_->tensor, self_->tensor, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUCharType::clamp_max(const Tensor & self, Scalar max) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto max_ = max.toChar();
    THCharTensor_cminValue(result_->tensor, self_->tensor, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUCharType::clamp_max_(Tensor & self, Scalar max) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto max_ = max.toChar();
    THCharTensor_cminValue(self_->tensor, self_->tensor, max_);
    return self;
}
Tensor CPUCharType::_dot(const Tensor & self, const Tensor & tensor) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto tensor_ = checked_cast_tensor<CPUCharTensor>(tensor.pImpl,"tensor",2, false);
    return scalarTensor(convert<int8_t>(THCharTensor_dot(self_->tensor, tensor_->tensor)));
}
Tensor & CPUCharType::tril_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    THCharTensor_tril(result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUCharType::tril(const Tensor & self, int64_t diagonal) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    THCharTensor_tril(result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUCharType::tril_(Tensor & self, int64_t diagonal) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    THCharTensor_tril(self_->tensor, self_->tensor, diagonal);
    return self;
}
Tensor & CPUCharType::triu_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    THCharTensor_triu(result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUCharType::triu(const Tensor & self, int64_t diagonal) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    THCharTensor_triu(result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUCharType::triu_(Tensor & self, int64_t diagonal) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    THCharTensor_triu(self_->tensor, self_->tensor, diagonal);
    return self;
}
Tensor & CPUCharType::cross_out(Tensor & result, const Tensor & self, const Tensor & other, int64_t dim) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",2, false);
    THCharTensor_cross(result_->tensor, self_->tensor, other_->tensor, dim);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUCharType::cross(const Tensor & self, const Tensor & other, int64_t dim) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUCharTensor>(other.pImpl,"other",2, false);
    THCharTensor_cross(result_->tensor, self_->tensor, other_->tensor, dim);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUCharType::diag_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    if (self_->isScalar()) {
      throw std::runtime_error("Input must be 1-d or 2-d");
    }
    THCharTensor_diag(result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUCharType::diag(const Tensor & self, int64_t diagonal) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    if (self_->isScalar()) {
      throw std::runtime_error("Input must be 1-d or 2-d");
    }
    THCharTensor_diag(result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUCharType::s_addmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    if(mat1.type().is_sparse()) {
        return static_cast<const Type*>(this)->addmm_out(result, self, SparseTensor(mat1), mat2, beta, alpha);
    }
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toChar();
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toChar();
    auto mat1_ = checked_cast_tensor<CPUCharTensor>(mat1.pImpl,"mat1",4, false);
    auto mat2_ = checked_cast_tensor<CPUCharTensor>(mat2.pImpl,"mat2",5, false);
    THCharTensor_addmm(result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CPUCharType::s_addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    if(mat1.type().is_sparse()) {
        return static_cast<const Type*>(this)->addmm(self, SparseTensor(mat1), mat2, beta, alpha);
    }
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toChar();
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toChar();
    auto mat1_ = checked_cast_tensor<CPUCharTensor>(mat1.pImpl,"mat1",4, false);
    auto mat2_ = checked_cast_tensor<CPUCharTensor>(mat2.pImpl,"mat2",5, false);
    THCharTensor_addmm(result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CPUCharType::addmm_out(Tensor & result, const Tensor & self, SparseTensor mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toChar();
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toChar();
    auto mat1_ = checked_cast_tensor<SparseCPUCharTensor>(mat1.tref.pImpl,"mat1",4,false);
    auto mat2_ = checked_cast_tensor<CPUCharTensor>(mat2.pImpl,"mat2",5, false);
    THSCharTensor_spaddmm(result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CPUCharType::addmm(const Tensor & self, SparseTensor mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toChar();
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toChar();
    auto mat1_ = checked_cast_tensor<SparseCPUCharTensor>(mat1.tref.pImpl,"mat1",4,false);
    auto mat2_ = checked_cast_tensor<CPUCharTensor>(mat2.pImpl,"mat2",5, false);
    THSCharTensor_spaddmm(result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CPUCharType::addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    if(mat1.type().is_sparse()) {
        return static_cast<const Type*>(this)->addmm_(self, SparseTensor(mat1), mat2, beta, alpha);
    }
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toChar();
    auto alpha_ = alpha.toChar();
    auto mat1_ = checked_cast_tensor<CPUCharTensor>(mat1.pImpl,"mat1",5, false);
    auto mat2_ = checked_cast_tensor<CPUCharTensor>(mat2.pImpl,"mat2",6, false);
    THCharTensor_addmm(self_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    return self;
}
Tensor & CPUCharType::addmm_(Tensor & self, SparseTensor mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toChar();
    auto alpha_ = alpha.toChar();
    auto mat1_ = checked_cast_tensor<SparseCPUCharTensor>(mat1.tref.pImpl,"mat1",5,false);
    auto mat2_ = checked_cast_tensor<CPUCharTensor>(mat2.pImpl,"mat2",6, false);
    THSCharTensor_spaddmm(self_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    return self;
}
Tensor & CPUCharType::s__addmv_out(Tensor & result, const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toChar();
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toChar();
    auto mat_ = checked_cast_tensor<CPUCharTensor>(mat.pImpl,"mat",4, false);
    auto vec_ = checked_cast_tensor<CPUCharTensor>(vec.pImpl,"vec",5, false);
    THCharTensor_addmv(result_->tensor, beta_, self_->tensor, alpha_, mat_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && mat_->isScalar() && vec_->isScalar());
    return result;
}
Tensor CPUCharType::s__addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toChar();
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toChar();
    auto mat_ = checked_cast_tensor<CPUCharTensor>(mat.pImpl,"mat",4, false);
    auto vec_ = checked_cast_tensor<CPUCharTensor>(vec.pImpl,"vec",5, false);
    THCharTensor_addmv(result_->tensor, beta_, self_->tensor, alpha_, mat_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && mat_->isScalar() && vec_->isScalar());
    return result;
}
Tensor & CPUCharType::_addmv_(Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toChar();
    auto alpha_ = alpha.toChar();
    auto mat_ = checked_cast_tensor<CPUCharTensor>(mat.pImpl,"mat",5, false);
    auto vec_ = checked_cast_tensor<CPUCharTensor>(vec.pImpl,"vec",6, false);
    THCharTensor_addmv(self_->tensor, beta_, self_->tensor, alpha_, mat_->tensor, vec_->tensor);
    return self;
}
Tensor & CPUCharType::s__addr_out(Tensor & result, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toChar();
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toChar();
    auto vec1_ = checked_cast_tensor<CPUCharTensor>(vec1.pImpl,"vec1",4, false);
    auto vec2_ = checked_cast_tensor<CPUCharTensor>(vec2.pImpl,"vec2",5, false);
    THCharTensor_addr(result_->tensor, beta_, self_->tensor, alpha_, vec1_->tensor, vec2_->tensor);
    result_->maybeScalar(self_->isScalar() && vec1_->isScalar() && vec2_->isScalar());
    return result;
}
Tensor CPUCharType::s__addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toChar();
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toChar();
    auto vec1_ = checked_cast_tensor<CPUCharTensor>(vec1.pImpl,"vec1",4, false);
    auto vec2_ = checked_cast_tensor<CPUCharTensor>(vec2.pImpl,"vec2",5, false);
    THCharTensor_addr(result_->tensor, beta_, self_->tensor, alpha_, vec1_->tensor, vec2_->tensor);
    result_->maybeScalar(self_->isScalar() && vec1_->isScalar() && vec2_->isScalar());
    return result;
}
Tensor & CPUCharType::_addr_(Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toChar();
    auto alpha_ = alpha.toChar();
    auto vec1_ = checked_cast_tensor<CPUCharTensor>(vec1.pImpl,"vec1",5, false);
    auto vec2_ = checked_cast_tensor<CPUCharTensor>(vec2.pImpl,"vec2",6, false);
    THCharTensor_addr(self_->tensor, beta_, self_->tensor, alpha_, vec1_->tensor, vec2_->tensor);
    return self;
}
Tensor & CPUCharType::_ger_out(Tensor & result, const Tensor & self, const Tensor & vec2) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.dim() == 0 ? 1 : self.size(0),vec2.dim() == 0 ? 1 : vec2.size(0) });
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto vec2_ = checked_cast_tensor<CPUCharTensor>(vec2.pImpl,"vec2",2, false);
    THCharTensor_addr(result_->tensor, int8_t(0), result_->tensor, int8_t(1), self_->tensor, vec2_->tensor);
    result_->maybeScalar(false);
    return result;
}
Tensor CPUCharType::_ger(const Tensor & self, const Tensor & vec2) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.dim() == 0 ? 1 : self.size(0),vec2.dim() == 0 ? 1 : vec2.size(0) });
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto vec2_ = checked_cast_tensor<CPUCharTensor>(vec2.pImpl,"vec2",2, false);
    THCharTensor_addr(result_->tensor, int8_t(0), result_->tensor, int8_t(1), self_->tensor, vec2_->tensor);
    result_->maybeScalar(false);
    return result;
}
Tensor & CPUCharType::_mv_out(Tensor & result, const Tensor & self, const Tensor & vec) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.size(0) });
    result.zero_();
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto vec_ = checked_cast_tensor<CPUCharTensor>(vec.pImpl,"vec",2, false);
    THCharTensor_addmv(result_->tensor, int8_t(0), result_->tensor, int8_t(1), self_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && vec_->isScalar());
    return result;
}
Tensor CPUCharType::_mv(const Tensor & self, const Tensor & vec) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.size(0) });
    result.zero_();
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto vec_ = checked_cast_tensor<CPUCharTensor>(vec.pImpl,"vec",2, false);
    THCharTensor_addmv(result_->tensor, int8_t(0), result_->tensor, int8_t(1), self_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && vec_->isScalar());
    return result;
}
Tensor & CPUCharType::_mm_out(Tensor & result, const Tensor & self, const Tensor & mat2) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.size(0),mat2.size(1) });
    result.zero_();
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CPUCharTensor>(mat2.pImpl,"mat2",2, false);
    THCharTensor_addmm(result_->tensor, int8_t(0), result_->tensor, int8_t(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CPUCharType::_mm(const Tensor & self, const Tensor & mat2) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.size(0),mat2.size(1) });
    result.zero_();
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CPUCharTensor>(mat2.pImpl,"mat2",2, false);
    THCharTensor_addmm(result_->tensor, int8_t(0), result_->tensor, int8_t(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CPUCharType::bmm_out(Tensor & result, const Tensor & self, const Tensor & mat2) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.size(0),self.size(1),mat2.size(2) });
    result.zero_();
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CPUCharTensor>(mat2.pImpl,"mat2",2, false);
    THCharTensor_baddbmm(result_->tensor, int8_t(0), result_->tensor, int8_t(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CPUCharType::bmm(const Tensor & self, const Tensor & mat2) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.size(0),self.size(1),mat2.size(2) });
    result.zero_();
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CPUCharTensor>(mat2.pImpl,"mat2",2, false);
    THCharTensor_baddbmm(result_->tensor, int8_t(0), result_->tensor, int8_t(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CPUCharType::s_addbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toChar();
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toChar();
    auto batch1_ = checked_cast_tensor<CPUCharTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CPUCharTensor>(batch2.pImpl,"batch2",5, false);
    THCharTensor_addbmm(result_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor CPUCharType::s_addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toChar();
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toChar();
    auto batch1_ = checked_cast_tensor<CPUCharTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CPUCharTensor>(batch2.pImpl,"batch2",5, false);
    THCharTensor_addbmm(result_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor & CPUCharType::addbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toChar();
    auto alpha_ = alpha.toChar();
    auto batch1_ = checked_cast_tensor<CPUCharTensor>(batch1.pImpl,"batch1",5, false);
    auto batch2_ = checked_cast_tensor<CPUCharTensor>(batch2.pImpl,"batch2",6, false);
    THCharTensor_addbmm(self_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    return self;
}
Tensor & CPUCharType::s_baddbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toChar();
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toChar();
    auto batch1_ = checked_cast_tensor<CPUCharTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CPUCharTensor>(batch2.pImpl,"batch2",5, false);
    THCharTensor_baddbmm(result_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor CPUCharType::s_baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toChar();
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toChar();
    auto batch1_ = checked_cast_tensor<CPUCharTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CPUCharTensor>(batch2.pImpl,"batch2",5, false);
    THCharTensor_baddbmm(result_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor & CPUCharType::baddbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toChar();
    auto alpha_ = alpha.toChar();
    auto batch1_ = checked_cast_tensor<CPUCharTensor>(batch1.pImpl,"batch1",5, false);
    auto batch2_ = checked_cast_tensor<CPUCharTensor>(batch2.pImpl,"batch2",6, false);
    THCharTensor_baddbmm(self_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    return self;
}
Tensor & CPUCharType::s_addcmul_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toChar();
    auto tensor1_ = checked_cast_tensor<CPUCharTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CPUCharTensor>(tensor2.pImpl,"tensor2",4, false);
    THCharTensor_addcmul(result_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor CPUCharType::s_addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toChar();
    auto tensor1_ = checked_cast_tensor<CPUCharTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CPUCharTensor>(tensor2.pImpl,"tensor2",4, false);
    THCharTensor_addcmul(result_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor & CPUCharType::s_addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toChar();
    auto tensor1_ = checked_cast_tensor<CPUCharTensor>(tensor1.pImpl,"tensor1",4, false);
    auto tensor2_ = checked_cast_tensor<CPUCharTensor>(tensor2.pImpl,"tensor2",5, false);
    THCharTensor_addcmul(self_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    return self;
}
Tensor & CPUCharType::s_addcdiv_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toChar();
    auto tensor1_ = checked_cast_tensor<CPUCharTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CPUCharTensor>(tensor2.pImpl,"tensor2",4, false);
    THCharTensor_addcdiv(result_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor CPUCharType::s_addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toChar();
    auto tensor1_ = checked_cast_tensor<CPUCharTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CPUCharTensor>(tensor2.pImpl,"tensor2",4, false);
    THCharTensor_addcdiv(result_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor & CPUCharType::s_addcdiv_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toChar();
    auto tensor1_ = checked_cast_tensor<CPUCharTensor>(tensor1.pImpl,"tensor1",4, false);
    auto tensor2_ = checked_cast_tensor<CPUCharTensor>(tensor2.pImpl,"tensor2",5, false);
    THCharTensor_addcdiv(self_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    return self;
}
Tensor & CPUCharType::random_(Tensor & self, int64_t from, int64_t to, Generator * generator) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    THCharTensor_clampedRandom(self_->tensor, generator_->generator, from, to);
    return self;
}
Tensor & CPUCharType::random_(Tensor & self, int64_t to, Generator * generator) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    THCharTensor_cappedRandom(self_->tensor, generator_->generator, to);
    return self;
}
Tensor & CPUCharType::random_(Tensor & self, Generator * generator) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    THCharTensor_random(self_->tensor, generator_->generator);
    return self;
}
Tensor & CPUCharType::geometric_(Tensor & self, double p, Generator * generator) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    THCharTensor_geometric(self_->tensor, generator_->generator, p);
    return self;
}
Tensor CPUCharType::tensor(Storage & storage, int64_t storageOffset, IntList size, IntList stride) const {
    auto storage_ = checked_cast_storage<CPUCharStorage>(&storage,"storage",1);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    return Tensor((new CPUCharTensor(context, THCharTensor_newWithStorage(storage_->storage, storageOffset, size_, stride_)))->maybeScalar(size.size() == 0),false);
}
Tensor CPUCharType::tensor(IntList size) const {
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new CPUCharTensor(context, THCharTensor_newWithSize(size_, NULL)))->maybeScalar(size.size() == 0),false);
}
Tensor CPUCharType::tensor(IntList size, IntList stride) const {
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    return Tensor((new CPUCharTensor(context, THCharTensor_newWithSize(size_, stride_)))->maybeScalar(size.size() == 0),false);
}
Tensor CPUCharType::tensor() const {
    return Tensor((new CPUCharTensor(context, THCharTensor_new())),false);
}
Tensor CPUCharType::alias(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPUCharTensor(context, THCharTensor_newWithTensor(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor & CPUCharType::as_strided_out(Tensor & result, const Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    auto result_ = checked_cast_tensor<CPUCharTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    if (storage_offset == -1) {
      storage_offset = self_->tensor->storageOffset;
    }
    THCharTensor_setStorage(result_->tensor, self_->tensor->storage, storage_offset, size_, stride_);
    result_->maybeScalar(size.size() == 0);
    return result;
}
Tensor CPUCharType::as_strided(const Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    auto result_ = new CPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    if (storage_offset == -1) {
      storage_offset = self_->tensor->storageOffset;
    }
    THCharTensor_setStorage(result_->tensor, self_->tensor->storage, storage_offset, size_, stride_);
    result_->maybeScalar(size.size() == 0);
    return result;
}
Tensor & CPUCharType::as_strided_(Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    if (storage_offset == -1) {
      storage_offset = self_->tensor->storageOffset;
    }
    THCharTensor_setStorage(self_->tensor, self_->tensor->storage, storage_offset, size_, stride_);
    self_->maybeScalar(size.size() == 0);
    return self;
}
Tensor & CPUCharType::_cat_out(Tensor & self, TensorList tensors, int64_t dim) const {
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",0, false);
    auto tensors_ = tensor_list_checked_cast<CPUCharTensor, Tensor, THCharTensor>(tensors,"tensors",1);
    THCharTensor_catArray(self_->tensor, tensors_.data(), tensors_.size(), dim);
    return self;
}
Tensor CPUCharType::_cat(TensorList tensors, int64_t dim) const {
    auto self_ = new CPUCharTensor(context);
    auto self = Tensor(self_, false);
    auto tensors_ = tensor_list_checked_cast<CPUCharTensor, Tensor, THCharTensor>(tensors,"tensors",1);
    THCharTensor_catArray(self_->tensor, tensors_.data(), tensors_.size(), dim);
    return self;
}
Tensor CPUCharType::_sparse_mask(const Tensor & self, SparseTensor mask) const {
    auto result_ = new SparseCPUCharTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUCharTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<SparseCPUCharTensor>(mask.tref.pImpl,"mask",2,false);
    THCharTensor_sparseMask(result_->tensor, self_->tensor, mask_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUCharType::abs_out(Tensor & result, const Tensor & self) const {
    return  at::native::_abs_out_cpu(result, self);
}
Tensor & CPUCharType::ceil_out(Tensor & result, const Tensor & self) const {
    return  at::native::_ceil_out_cpu(result, self);
}
Tensor & CPUCharType::cos_out(Tensor & result, const Tensor & self) const {
    return  at::native::_cos_out_cpu(result, self);
}
Tensor CPUCharType::embedding_dense_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) const {
    return  at::native::embedding_backward_cpu(grad, indices, num_weights, padding_idx, scale_grad_by_freq);
}
Tensor & CPUCharType::embedding_renorm_(Tensor & self, const Tensor & indices, double max_norm, double norm_type) const {
    return  at::native::embedding_renorm_cpu_(self, indices, max_norm, norm_type);
}
std::tuple<Tensor,Tensor,Tensor> CPUCharType::embedding_bag(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse) const {
    return  at::native::embedding_bag_cpu(weight, indices, offsets, scale_grad_by_freq, mode, sparse);
}
Tensor CPUCharType::embedding_bag_dense_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode) const {
    return  at::native::embedding_bag_backward_cpu(grad, indices, offsets, offset2bag, bag_size, num_weights, scale_grad_by_freq, mode);
}
Tensor & CPUCharType::exp_out(Tensor & result, const Tensor & self) const {
    return  at::native::_exp_out_cpu(result, self);
}
Tensor & CPUCharType::eye_out(Tensor & result, int64_t n, int64_t m) const {
    return  at::native::eye_out_cpu(result, n, m);
}
Tensor & CPUCharType::floor_out(Tensor & result, const Tensor & self) const {
    return  at::native::_floor_out_cpu(result, self);
}
Tensor CPUCharType::_fft_with_size(const Tensor & self, int64_t signal_ndim, bool complex_input, bool complex_output, bool inverse, IntList checked_signal_sizes, bool normalized, bool onesided, IntList output_sizes) const {
    return  at::native::_fft_mkl(self, signal_ndim, complex_input, complex_output, inverse, checked_signal_sizes, normalized, onesided, output_sizes);
}
Tensor & CPUCharType::log_out(Tensor & result, const Tensor & self) const {
    return  at::native::_log_out_cpu(result, self);
}
std::tuple<Tensor,Tensor> CPUCharType::RoiPooling2d_forward(const Tensor & input, const Tensor & rois, int64_t pooledHeight, int64_t pooledWidth, double spatialScale) const {
    return  at::native::RoiPooling2d_forward_cpu(input, rois, pooledHeight, pooledWidth, spatialScale);
}
Tensor CPUCharType::RoiPooling2d_backward(const Tensor & input, const Tensor & rois, int64_t pooledHeight, int64_t pooledWidth, double spatialScale, const Tensor & gradOutput, const Tensor & argmaxes) const {
    return  at::native::RoiPooling2d_backward_cpu(input, rois, pooledHeight, pooledWidth, spatialScale, gradOutput, argmaxes);
}
Tensor & CPUCharType::round_out(Tensor & result, const Tensor & self) const {
    return  at::native::_round_out_cpu(result, self);
}
Tensor & CPUCharType::sin_out(Tensor & result, const Tensor & self) const {
    return  at::native::_sin_out_cpu(result, self);
}
Tensor & CPUCharType::sspaddmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return  at::native::_sspaddmm_out_only_sparse(result, self, mat1, mat2, beta, alpha);
}
Tensor CPUCharType::sum(const Tensor & self) const {
    return  at::native::_sum_cpu(self);
}
Tensor & CPUCharType::sum_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    return  at::native::_sum_out_cpu(result, self, dim, keepdim);
}
Tensor & CPUCharType::sqrt_out(Tensor & result, const Tensor & self) const {
    return  at::native::_sqrt_out_cpu(result, self);
}
Tensor CPUCharType::prod(const Tensor & self) const {
    return  at::native::_prod_cpu(self);
}
Tensor & CPUCharType::prod_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    return  at::native::_prod_out_cpu(result, self, dim, keepdim);
}
Tensor & CPUCharType::trunc_out(Tensor & result, const Tensor & self) const {
    return  at::native::_trunc_out_cpu(result, self);
}
std::tuple<Tensor,Tensor> CPUCharType::_unique(const Tensor & self, bool sorted, bool return_inverse) const {
    return  at::native::_unique_cpu(self, sorted, return_inverse);
}
Tensor CPUCharType::_s_where(const Tensor & condition, const Tensor & self, const Tensor & other) const {
    return  at::native::_s_where_cpu(condition, self, other);
}
Tensor CPUCharType::_standard_gamma_grad(const Tensor & self, const Tensor & output) const {
    return  at::native::_standard_gamma_grad_cpu(self, output);
}
Tensor CPUCharType::poisson(const Tensor & self, Generator * generator) const {
    return  at::native::_s_poisson_cpu(self, generator);
}

}
