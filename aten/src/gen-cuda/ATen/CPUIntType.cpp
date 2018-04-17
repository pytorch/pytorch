// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/CPUIntType.h"
#include "ATen/CPUIntStorage.h"
#include "ATen/CPUIntTensor.h"
#include "ATen/CPUGenerator.h"
#include "ATen/CPUByteTensor.h"
#include "ATen/CPUIntTensor.h"
#include "ATen/CPULongTensor.h"
#include "ATen/SparseCPUIntTensor.h"
#include "ATen/CPUIntTensor.h"
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

CPUIntType::CPUIntType(Context* context)
: Type(context, /*is_variable_or_undefined=*/false) {}
ScalarType CPUIntType::scalarType() const {
  return ScalarType::Int;
}
Backend CPUIntType::backend() const {
  return Backend::CPU;
}
bool CPUIntType::is_cuda() const { return backend() == kCUDA || backend() == kSparseCUDA; }
bool CPUIntType::is_sparse() const { return backend() == kSparseCPU || backend() == kSparseCUDA; }
bool CPUIntType::is_distributed() const { return false; }

std::unique_ptr<Storage> CPUIntType::storage() const {
  return std::unique_ptr<Storage>(new CPUIntStorage(context));
}
std::unique_ptr<Storage> CPUIntType::storage(size_t size) const {
  return std::unique_ptr<Storage>(new CPUIntStorage(context,size));
}
std::unique_ptr<Storage> CPUIntType::storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const {
    return std::unique_ptr<Storage>(
      new CPUIntStorage(context,data,size,deleter));
}
std::unique_ptr<Storage> CPUIntType::storageWithAllocator(int64_t size, std::unique_ptr<Allocator> allocator) const {
    return std::unique_ptr<Storage>(
        new CPUIntStorage(context, size, std::move(allocator)));
}
Tensor CPUIntType::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THIntTensor_retain( (THIntTensor*) th_pointer);
  return Tensor(new CPUIntTensor(context,(THIntTensor*)(th_pointer)), false);
}
std::unique_ptr<Storage> CPUIntType::unsafeStorageFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THIntStorage_retain( (THIntStorage*) th_pointer);
  return std::unique_ptr<Storage>(new CPUIntStorage(context, (THIntStorage*) th_pointer));
}
std::unique_ptr<Generator> CPUIntType::generator() const {
  return std::unique_ptr<Generator>(new CPUGenerator(context));
}

const char * CPUIntType::toString() const {
  return CPUIntType::typeString();
}
TypeID CPUIntType::ID() const {
  return TypeID::CPUInt;
}

std::size_t CPUIntType::elementSizeInBytes() const {
  return sizeof(int);
}

const char * CPUIntType::typeString() {
  return "CPUIntType";
}

/* example
Tensor * CPUIntType::add(Tensor & a, Tensor & b) {
  std::cout << "add CPUIntTensor\n";
  return &a;
}
*/

int64_t CPUIntType::storage_offset(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    return static_cast<int64_t>(THIntTensor_storageOffset(self_->tensor));
}
Tensor & CPUIntType::resize_(Tensor & self, IntList size) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    THIntTensor_resize(self_->tensor, size_, NULL);
    self_->maybeScalar(size.size() == 0);
    return self;
}
int64_t CPUIntType::numel(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    return static_cast<int64_t>(THIntTensor_nElement(self_->tensor));
}
Tensor & CPUIntType::set_(Tensor & self, Storage & storage) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto storage_ = checked_cast_storage<CPUIntStorage>(&storage,"storage",2);
    THIntTensor_setStorage(self_->tensor, storage_->storage, 0, THLongStorageView::makeFromLength(static_cast<int64_t>(storage.size())), NULL);
    self_->maybeScalar(false);
    return self;
}
Tensor & CPUIntType::set_(Tensor & self, Storage & sourceStorage, int64_t storage_offset, IntList size, IntList stride) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto sourceStorage_ = checked_cast_storage<CPUIntStorage>(&sourceStorage,"sourceStorage",2);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    THIntTensor_setStorage(self_->tensor, sourceStorage_->storage, storage_offset, size_, stride_);
    self_->maybeScalar(size.size() == 0);
    return self;
}
Tensor & CPUIntType::set_(Tensor & self, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto source_ = checked_cast_tensor<CPUIntTensor>(source.pImpl,"source",2, false);
    THIntTensor_set(self_->tensor, source_->tensor);
    self_->maybeScalar(source_->isScalar());
    return self;
}
Tensor & CPUIntType::set_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    THIntTensor_setStorage(self_->tensor, NULL, 0, NULL, NULL);
    self_->maybeScalar(false);
    return self;
}
Tensor & CPUIntType::fill_(Tensor & self, Scalar value) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toInt();
    THIntTensor_fill(self_->tensor, value_);
    return self;
}
Tensor & CPUIntType::fill_(Tensor & self, const Tensor & value) const {
    if (value.dim() == 0) {
        return static_cast<const Type*>(this)->fill_(self, Scalar(value));
    }
    AT_ERROR("fill_ only supports a 0-dimensional value tensor, but got tensor "
        "with %" PRId64 " dimension(s)", value.dim());
}
bool CPUIntType::is_contiguous(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    return THIntTensor_isContiguous(self_->tensor);
}
bool CPUIntType::is_set_to(const Tensor & self, const Tensor & tensor) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto tensor_ = checked_cast_tensor<CPUIntTensor>(tensor.pImpl,"tensor",2, false);
    return THIntTensor_isSetTo(self_->tensor, tensor_->tensor);
}
Tensor & CPUIntType::s_masked_fill_(Tensor & self, const Tensor & mask, Scalar value) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CPUByteTensor>(mask.pImpl,"mask",2, false);
    auto value_ = value.toInt();
    THIntTensor_maskedFill(self_->tensor, mask_->tensor, value_);
    return self;
}
Tensor & CPUIntType::s_masked_fill_(Tensor & self, const Tensor & mask, const Tensor & value) const {
    if (value.dim() == 0) {
        return static_cast<const Type*>(this)->masked_fill_(self, mask, Scalar(value));
    }
    AT_ERROR("masked_fill_ only supports a 0-dimensional value tensor, but got tensor "
        "with %" PRId64 " dimension(s)", value.dim());
}
Tensor & CPUIntType::s_masked_scatter_(Tensor & self, const Tensor & mask, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CPUByteTensor>(mask.pImpl,"mask",2, false);
    auto source_ = checked_cast_tensor<CPUIntTensor>(source.pImpl,"source",3, false);
    THIntTensor_maskedCopy(self_->tensor, mask_->tensor, source_->tensor);
    return self;
}
Tensor & CPUIntType::s_masked_select_out(Tensor & result, const Tensor & self, const Tensor & mask) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CPUByteTensor>(mask.pImpl,"mask",2, false);
    THIntTensor_maskedSelect(result_->tensor, self_->tensor, mask_->tensor);
    result_->maybeScalar(self_->isScalar() && mask_->isScalar());
    return result;
}
Tensor CPUIntType::s_masked_select(const Tensor & self, const Tensor & mask) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CPUByteTensor>(mask.pImpl,"mask",2, false);
    THIntTensor_maskedSelect(result_->tensor, self_->tensor, mask_->tensor);
    result_->maybeScalar(self_->isScalar() && mask_->isScalar());
    return result;
}
Tensor CPUIntType::transpose(const Tensor & self, int64_t dim0, int64_t dim1) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    dim0 = maybe_wrap_dim(dim0, self_);
    dim1 = maybe_wrap_dim(dim1, self_);
    return Tensor((new CPUIntTensor(context, THIntTensor_newTranspose(self_->tensor, dim0, dim1)))->maybeScalar(self_->isScalar()),false);
}
Tensor CPUIntType::t(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPUIntTensor(context, THIntTensor_newTranspose(self_->tensor, 0, 1)))->maybeScalar(self_->isScalar()),false);
}
Tensor & CPUIntType::nonzero_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    THIntTensor_nonzero(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUIntType::nonzero(const Tensor & self) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    THIntTensor_nonzero(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUIntType::contiguous(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPUIntTensor(context, THIntTensor_newContiguous(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor CPUIntType::clone(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPUIntTensor(context, THIntTensor_newClone(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor CPUIntType::view(const Tensor & self, IntList size) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new CPUIntTensor(context, THIntTensor_newView(self_->tensor, size_)))->maybeScalar(size.size() == 0),false);
}
Tensor & CPUIntType::resize_as_(Tensor & self, const Tensor & the_template) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto the_template_ = checked_cast_tensor<CPUIntTensor>(the_template.pImpl,"the_template",2, false);
    THIntTensor_resizeAs(self_->tensor, the_template_->tensor);
    self_->maybeScalar(the_template_->isScalar());
    return self;
}
Tensor & CPUIntType::index_select_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    THIntTensor_indexSelect(result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
Tensor CPUIntType::index_select(const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    THIntTensor_indexSelect(result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
Tensor & CPUIntType::_indexCopy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    auto source_ = checked_cast_tensor<CPUIntTensor>(source.pImpl,"source",4, false);
    THIntTensor_indexCopy(self_->tensor, dim, index_->tensor, source_->tensor);
    return self;
}
Tensor & CPUIntType::take_out(Tensor & result, const Tensor & self, const Tensor & index) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",2, false);
    THIntTensor_take(result_->tensor, self_->tensor, index_->tensor);
    result_->maybeScalar(index_->isScalar());
    return result;
}
Tensor CPUIntType::take(const Tensor & self, const Tensor & index) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",2, false);
    THIntTensor_take(result_->tensor, self_->tensor, index_->tensor);
    result_->maybeScalar(index_->isScalar());
    return result;
}
Tensor & CPUIntType::put_(Tensor & self, const Tensor & index, const Tensor & source, bool accumulate) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",2, false);
    auto source_ = checked_cast_tensor<CPUIntTensor>(source.pImpl,"source",3, false);
    THIntTensor_put(self_->tensor, index_->tensor, source_->tensor, accumulate);
    return self;
}
Tensor & CPUIntType::index_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    auto source_ = checked_cast_tensor<CPUIntTensor>(source.pImpl,"source",4, false);
    THIntTensor_indexAdd(self_->tensor, dim, index_->tensor, source_->tensor);
    return self;
}
Tensor & CPUIntType::index_fill_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    auto value_ = value.toInt();
    THIntTensor_indexFill(self_->tensor, dim, index_->tensor, value_);
    return self;
}
Tensor & CPUIntType::index_fill_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & value) const {
    if (value.dim() == 0) {
        return static_cast<const Type*>(this)->index_fill_(self, dim, index, Scalar(value));
    }
    AT_ERROR("index_fill_ only supports a 0-dimensional value tensor, but got tensor "
        "with %" PRId64 " dimension(s)", value.dim());
}
Tensor CPUIntType::unfold(const Tensor & self, int64_t dimension, int64_t size, int64_t step) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    dimension = maybe_wrap_dim(dimension, self_);
    THIntTensor_unfold(result_->tensor, self_->tensor, dimension, size, step);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUIntType::_range_out(Tensor & result, Scalar start, Scalar end, Scalar step) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto start_ = start.toLong();
    auto end_ = end.toLong();
    auto step_ = step.toLong();
    THIntTensor_range(result_->tensor, start_, end_, step_);
    return result;
}
Tensor CPUIntType::_range(Scalar start, Scalar end, Scalar step) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto start_ = start.toLong();
    auto end_ = end.toLong();
    auto step_ = step.toLong();
    THIntTensor_range(result_->tensor, start_, end_, step_);
    return result;
}
Tensor & CPUIntType::_arange_out(Tensor & result, Scalar start, Scalar end, Scalar step) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto start_ = start.toLong();
    auto end_ = end.toLong();
    auto step_ = step.toLong();
    THIntTensor_arange(result_->tensor, start_, end_, step_);
    return result;
}
Tensor CPUIntType::_arange(Scalar start, Scalar end, Scalar step) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto start_ = start.toLong();
    auto end_ = end.toLong();
    auto step_ = step.toLong();
    THIntTensor_arange(result_->tensor, start_, end_, step_);
    return result;
}
Tensor & CPUIntType::_arange_out(Tensor & result, Scalar end) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto end_ = end.toLong();
    THIntTensor_arange(result_->tensor, 0, end_, 1);
    return result;
}
Tensor CPUIntType::_arange(Scalar end) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto end_ = end.toLong();
    THIntTensor_arange(result_->tensor, 0, end_, 1);
    return result;
}
Tensor & CPUIntType::scatter_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
    if (src.dim() == 0) {
        return static_cast<const Type*>(this)->scatter_(self, dim, index, Scalar(src));
    }
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    auto src_ = checked_cast_tensor<CPUIntTensor>(src.pImpl,"src",4, false);
    THIntTensor_scatter(self_->tensor, dim, index_->tensor, src_->tensor);
    return self;
}
Tensor & CPUIntType::scatter_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    auto value_ = value.toInt();
    THIntTensor_scatterFill(self_->tensor, dim, index_->tensor, value_);
    return self;
}
Tensor & CPUIntType::scatter_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    auto src_ = checked_cast_tensor<CPUIntTensor>(src.pImpl,"src",4, false);
    THIntTensor_scatterAdd(self_->tensor, dim, index_->tensor, src_->tensor);
    return self;
}
Tensor & CPUIntType::gather_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    result.resize_(index.sizes());
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    THIntTensor_gather(result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
Tensor CPUIntType::gather(const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    result.resize_(index.sizes());
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    THIntTensor_gather(result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
void* CPUIntType::data_ptr(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    return THIntTensor_data(self_->tensor);
}
bool CPUIntType::equal(const Tensor & self, const Tensor & other) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",2, false);
    return THIntTensor_equal(self_->tensor, other_->tensor);
}
Tensor & CPUIntType::__and___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_bitand(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUIntType::__and__(const Tensor & self, Scalar other) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_bitand(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUIntType::s___and___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__and___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",2, false);
    THIntTensor_cbitand(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUIntType::s___and__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__and__(self, Scalar(other));
    }
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",2, false);
    THIntTensor_cbitand(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUIntType::__iand__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_bitand(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUIntType::s___iand__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__iand__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",3, false);
    THIntTensor_cbitand(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUIntType::__or___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_bitor(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUIntType::__or__(const Tensor & self, Scalar other) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_bitor(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUIntType::s___or___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__or___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",2, false);
    THIntTensor_cbitor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUIntType::s___or__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__or__(self, Scalar(other));
    }
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",2, false);
    THIntTensor_cbitor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUIntType::__ior__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_bitor(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUIntType::s___ior__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__ior__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",3, false);
    THIntTensor_cbitor(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUIntType::__xor___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_bitxor(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUIntType::__xor__(const Tensor & self, Scalar other) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_bitxor(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUIntType::s___xor___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__xor___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",2, false);
    THIntTensor_cbitxor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUIntType::s___xor__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__xor__(self, Scalar(other));
    }
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",2, false);
    THIntTensor_cbitxor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUIntType::__ixor__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_bitxor(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUIntType::s___ixor__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__ixor__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",3, false);
    THIntTensor_cbitxor(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUIntType::__lshift___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_lshift(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUIntType::__lshift__(const Tensor & self, Scalar other) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_lshift(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUIntType::s___lshift___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__lshift___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",2, false);
    THIntTensor_clshift(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUIntType::s___lshift__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__lshift__(self, Scalar(other));
    }
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",2, false);
    THIntTensor_clshift(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUIntType::__ilshift__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_lshift(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUIntType::s___ilshift__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__ilshift__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",3, false);
    THIntTensor_clshift(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUIntType::__rshift___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_rshift(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUIntType::__rshift__(const Tensor & self, Scalar other) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_rshift(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUIntType::s___rshift___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__rshift___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",2, false);
    THIntTensor_crshift(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUIntType::s___rshift__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__rshift__(self, Scalar(other));
    }
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",2, false);
    THIntTensor_crshift(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUIntType::__irshift__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_rshift(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUIntType::s___irshift__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__irshift__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",3, false);
    THIntTensor_crshift(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUIntType::lt_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_ltValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUIntType::lt(const Tensor & self, Scalar other) const {
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_ltValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUIntType::s_lt_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->lt_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",2, false);
    THIntTensor_ltTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUIntType::s_lt(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->lt(self, Scalar(other));
    }
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",2, false);
    THIntTensor_ltTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUIntType::lt_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_ltValueT(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUIntType::s_lt_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->lt_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",3, false);
    THIntTensor_ltTensorT(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUIntType::gt_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_gtValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUIntType::gt(const Tensor & self, Scalar other) const {
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_gtValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUIntType::s_gt_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->gt_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",2, false);
    THIntTensor_gtTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUIntType::s_gt(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->gt(self, Scalar(other));
    }
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",2, false);
    THIntTensor_gtTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUIntType::gt_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_gtValueT(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUIntType::s_gt_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->gt_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",3, false);
    THIntTensor_gtTensorT(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUIntType::le_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_leValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUIntType::le(const Tensor & self, Scalar other) const {
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_leValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUIntType::s_le_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->le_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",2, false);
    THIntTensor_leTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUIntType::s_le(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->le(self, Scalar(other));
    }
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",2, false);
    THIntTensor_leTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUIntType::le_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_leValueT(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUIntType::s_le_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->le_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",3, false);
    THIntTensor_leTensorT(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUIntType::ge_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_geValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUIntType::ge(const Tensor & self, Scalar other) const {
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_geValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUIntType::s_ge_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ge_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",2, false);
    THIntTensor_geTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUIntType::s_ge(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ge(self, Scalar(other));
    }
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",2, false);
    THIntTensor_geTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUIntType::ge_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_geValueT(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUIntType::s_ge_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ge_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",3, false);
    THIntTensor_geTensorT(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUIntType::eq_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_eqValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUIntType::eq(const Tensor & self, Scalar other) const {
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_eqValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUIntType::s_eq_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->eq_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",2, false);
    THIntTensor_eqTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUIntType::s_eq(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->eq(self, Scalar(other));
    }
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",2, false);
    THIntTensor_eqTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUIntType::eq_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_eqValueT(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUIntType::s_eq_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->eq_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",3, false);
    THIntTensor_eqTensorT(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUIntType::ne_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_neValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUIntType::ne(const Tensor & self, Scalar other) const {
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_neValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUIntType::s_ne_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ne_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",2, false);
    THIntTensor_neTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUIntType::s_ne(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ne(self, Scalar(other));
    }
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",2, false);
    THIntTensor_neTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUIntType::ne_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_neValueT(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUIntType::s_ne_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ne_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",3, false);
    THIntTensor_neTensorT(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
std::tuple<Tensor &,Tensor &> CPUIntType::min_out(Tensor & min, Tensor & min_indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto min_ = checked_cast_tensor<CPUIntTensor>(min.pImpl,"min",0, false);
    auto min_indices_ = checked_cast_tensor<CPULongTensor>(min_indices.pImpl,"min_indices",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THIntTensor_min(min_->tensor, min_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    min_->maybeScalar(maybe_scalar);
    min_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(min, min_indices);
}
std::tuple<Tensor,Tensor> CPUIntType::min(const Tensor & self, int64_t dim, bool keepdim) const {
    auto min_ = new CPUIntTensor(context);
    auto min = Tensor(min_, false);
    auto min_indices_ = new CPULongTensor(context);
    auto min_indices = Tensor(min_indices_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THIntTensor_min(min_->tensor, min_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    min_->maybeScalar(maybe_scalar);
    min_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(min, min_indices);
}
Tensor & CPUIntType::s_min_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",2, false);
    THIntTensor_cmin(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUIntType::s_min(const Tensor & self, const Tensor & other) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",2, false);
    THIntTensor_cmin(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUIntType::min(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<int>(THIntTensor_minall(self_->tensor)));
}
std::tuple<Tensor &,Tensor &> CPUIntType::max_out(Tensor & max, Tensor & max_indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto max_ = checked_cast_tensor<CPUIntTensor>(max.pImpl,"max",0, false);
    auto max_indices_ = checked_cast_tensor<CPULongTensor>(max_indices.pImpl,"max_indices",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THIntTensor_max(max_->tensor, max_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    max_->maybeScalar(maybe_scalar);
    max_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(max, max_indices);
}
std::tuple<Tensor,Tensor> CPUIntType::max(const Tensor & self, int64_t dim, bool keepdim) const {
    auto max_ = new CPUIntTensor(context);
    auto max = Tensor(max_, false);
    auto max_indices_ = new CPULongTensor(context);
    auto max_indices = Tensor(max_indices_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THIntTensor_max(max_->tensor, max_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    max_->maybeScalar(maybe_scalar);
    max_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(max, max_indices);
}
Tensor & CPUIntType::s_max_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",2, false);
    THIntTensor_cmax(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUIntType::s_max(const Tensor & self, const Tensor & other) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",2, false);
    THIntTensor_cmax(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUIntType::max(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<int>(THIntTensor_maxall(self_->tensor)));
}
std::tuple<Tensor &,Tensor &> CPUIntType::kthvalue_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool keepdim) const {
    auto values_ = checked_cast_tensor<CPUIntTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THIntTensor_kthvalue(values_->tensor, indices_->tensor, self_->tensor, k, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CPUIntType::kthvalue(const Tensor & self, int64_t k, int64_t dim, bool keepdim) const {
    auto values_ = new CPUIntTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CPULongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THIntTensor_kthvalue(values_->tensor, indices_->tensor, self_->tensor, k, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
std::tuple<Tensor &,Tensor &> CPUIntType::mode_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = checked_cast_tensor<CPUIntTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THIntTensor_mode(values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CPUIntType::mode(const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = new CPUIntTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CPULongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THIntTensor_mode(values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
std::tuple<Tensor &,Tensor &> CPUIntType::median_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = checked_cast_tensor<CPUIntTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THIntTensor_median(values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CPUIntType::median(const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = new CPUIntTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CPULongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THIntTensor_median(values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
Tensor CPUIntType::median(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<int>(THIntTensor_medianall(self_->tensor)));
}
std::tuple<Tensor &,Tensor &> CPUIntType::sort_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool descending) const {
    auto values_ = checked_cast_tensor<CPUIntTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THIntTensor_sort(values_->tensor, indices_->tensor, self_->tensor, dim, descending);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CPUIntType::sort(const Tensor & self, int64_t dim, bool descending) const {
    auto values_ = new CPUIntTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CPULongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THIntTensor_sort(values_->tensor, indices_->tensor, self_->tensor, dim, descending);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
std::tuple<Tensor &,Tensor &> CPUIntType::topk_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) const {
    auto values_ = checked_cast_tensor<CPUIntTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THIntTensor_topk(values_->tensor, indices_->tensor, self_->tensor, k, dim, largest, sorted);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CPUIntType::topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) const {
    auto values_ = new CPUIntTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CPULongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THIntTensor_topk(values_->tensor, indices_->tensor, self_->tensor, k, dim, largest, sorted);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
Tensor & CPUIntType::_abs_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    THIntTensor_abs(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUIntType::_abs(const Tensor & self) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    THIntTensor_abs(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUIntType::neg_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    THIntTensor_neg(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUIntType::neg(const Tensor & self) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    THIntTensor_neg(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUIntType::neg_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    THIntTensor_neg(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUIntType::pow_out(Tensor & result, const Tensor & self, Scalar exponent) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toInt();
    THIntTensor_pow(result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUIntType::pow(const Tensor & self, Scalar exponent) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toInt();
    THIntTensor_pow(result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUIntType::s_pow_out(Tensor & result, const Tensor & self, const Tensor & exponent) const {
    if (exponent.dim() == 0) {
        return static_cast<const Type*>(this)->pow_out(result, self, Scalar(exponent));
    }
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto exponent_ = checked_cast_tensor<CPUIntTensor>(exponent.pImpl,"exponent",2, false);
    THIntTensor_cpow(result_->tensor, self_->tensor, exponent_->tensor);
    result_->maybeScalar(self_->isScalar() && exponent_->isScalar());
    return result;
}
Tensor CPUIntType::s_pow(const Tensor & self, const Tensor & exponent) const {
    if (exponent.dim() == 0) {
        return static_cast<const Type*>(this)->pow(self, Scalar(exponent));
    }
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto exponent_ = checked_cast_tensor<CPUIntTensor>(exponent.pImpl,"exponent",2, false);
    THIntTensor_cpow(result_->tensor, self_->tensor, exponent_->tensor);
    result_->maybeScalar(self_->isScalar() && exponent_->isScalar());
    return result;
}
Tensor & CPUIntType::pow_out(Tensor & result, Scalar base, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto base_ = base.toInt();
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",2, false);
    THIntTensor_tpow(result_->tensor, base_, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUIntType::pow(Scalar base, const Tensor & self) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto base_ = base.toInt();
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",2, false);
    THIntTensor_tpow(result_->tensor, base_, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUIntType::pow_(Tensor & self, Scalar exponent) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toInt();
    THIntTensor_pow(self_->tensor, self_->tensor, exponent_);
    return self;
}
Tensor & CPUIntType::s_pow_(Tensor & self, const Tensor & exponent) const {
    if (exponent.dim() == 0) {
        return static_cast<const Type*>(this)->pow_(self, Scalar(exponent));
    }
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto exponent_ = checked_cast_tensor<CPUIntTensor>(exponent.pImpl,"exponent",3, false);
    THIntTensor_cpow(self_->tensor, self_->tensor, exponent_->tensor);
    return self;
}
Tensor & CPUIntType::zero_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    THIntTensor_zero(self_->tensor);
    return self;
}
Tensor CPUIntType::_sumall(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<int>(THIntTensor_sumall(self_->tensor)));
}
Tensor & CPUIntType::_sum_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THIntTensor_sum(result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CPUIntType::_sum(const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THIntTensor_sum(result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CPUIntType::_prodall(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<int>(THIntTensor_prodall(self_->tensor)));
}
Tensor & CPUIntType::_prod_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THIntTensor_prod(result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CPUIntType::_prod(const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THIntTensor_prod(result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor & CPUIntType::cumsum_out(Tensor & result, const Tensor & self, int64_t dim) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THIntTensor_cumsum(result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUIntType::cumsum(const Tensor & self, int64_t dim) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THIntTensor_cumsum(result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUIntType::cumprod_out(Tensor & result, const Tensor & self, int64_t dim) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THIntTensor_cumprod(result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUIntType::cumprod(const Tensor & self, int64_t dim) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THIntTensor_cumprod(result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUIntType::sign_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    THIntTensor_sign(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUIntType::sign(const Tensor & self) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    THIntTensor_sign(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUIntType::sign_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    THIntTensor_sign(self_->tensor, self_->tensor);
    return self;
}
Tensor CPUIntType::trace(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<int>(THIntTensor_trace(self_->tensor)));
}
Tensor & CPUIntType::add_out(Tensor & result, const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    auto alpha_ = alpha.toInt();
    THIntTensor_add_scaled(result_->tensor, self_->tensor, other_, alpha_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUIntType::add(const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    auto alpha_ = alpha.toInt();
    THIntTensor_add_scaled(result_->tensor, self_->tensor, other_, alpha_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUIntType::s_add_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if(other.type().is_sparse()) {
        return static_cast<const Type*>(this)->add_out(result, self, SparseTensor(other), alpha);
    }
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toInt();
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",3, false);
    THIntTensor_cadd(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUIntType::s_add(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if(other.type().is_sparse()) {
        return static_cast<const Type*>(this)->add(self, SparseTensor(other), alpha);
    }
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add(self, Scalar(other), alpha);
    }
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toInt();
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",3, false);
    THIntTensor_cadd(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUIntType::add_out(Tensor & result, const Tensor & self, SparseTensor other, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toInt();
    auto other_ = checked_cast_tensor<SparseCPUIntTensor>(other.tref.pImpl,"other",3,false);
    THSIntTensor_spcadd(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUIntType::add(const Tensor & self, SparseTensor other, Scalar alpha) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toInt();
    auto other_ = checked_cast_tensor<SparseCPUIntTensor>(other.tref.pImpl,"other",3,false);
    THSIntTensor_spcadd(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUIntType::add_(Tensor & self, Scalar other, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    auto alpha_ = alpha.toInt();
    THIntTensor_add_scaled(self_->tensor, self_->tensor, other_, alpha_);
    return self;
}
Tensor & CPUIntType::s_add_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if(other.type().is_sparse()) {
        return static_cast<const Type*>(this)->add_(self, SparseTensor(other), alpha);
    }
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toInt();
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",4, false);
    THIntTensor_cadd(self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & CPUIntType::add_(Tensor & self, SparseTensor other, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toInt();
    auto other_ = checked_cast_tensor<SparseCPUIntTensor>(other.tref.pImpl,"other",4,false);
    THSIntTensor_spcadd(self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & CPUIntType::sub_out(Tensor & result, const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    auto alpha_ = alpha.toInt();
    THIntTensor_sub_scaled(result_->tensor, self_->tensor, other_, alpha_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUIntType::sub(const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    auto alpha_ = alpha.toInt();
    THIntTensor_sub_scaled(result_->tensor, self_->tensor, other_, alpha_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUIntType::s_sub_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toInt();
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",3, false);
    THIntTensor_csub(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUIntType::s_sub(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub(self, Scalar(other), alpha);
    }
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toInt();
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",3, false);
    THIntTensor_csub(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUIntType::sub_(Tensor & self, Scalar other, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    auto alpha_ = alpha.toInt();
    THIntTensor_sub_scaled(self_->tensor, self_->tensor, other_, alpha_);
    return self;
}
Tensor & CPUIntType::s_sub_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toInt();
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",4, false);
    THIntTensor_csub(self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & CPUIntType::mul_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_mul(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUIntType::mul(const Tensor & self, Scalar other) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_mul(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUIntType::s_mul_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",2, false);
    THIntTensor_cmul(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUIntType::s_mul(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul(self, Scalar(other));
    }
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",2, false);
    THIntTensor_cmul(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUIntType::mul_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_mul(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUIntType::s_mul_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",3, false);
    THIntTensor_cmul(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUIntType::div_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_div(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUIntType::div(const Tensor & self, Scalar other) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_div(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUIntType::s_div_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->div_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",2, false);
    THIntTensor_cdiv(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUIntType::s_div(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->div(self, Scalar(other));
    }
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",2, false);
    THIntTensor_cdiv(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUIntType::div_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_div(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUIntType::s_div_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->div_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",3, false);
    THIntTensor_cdiv(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUIntType::fmod_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_fmod(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUIntType::fmod(const Tensor & self, Scalar other) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_fmod(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUIntType::s_fmod_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->fmod_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",2, false);
    THIntTensor_cfmod(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUIntType::s_fmod(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->fmod(self, Scalar(other));
    }
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",2, false);
    THIntTensor_cfmod(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUIntType::fmod_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_fmod(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUIntType::s_fmod_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->fmod_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",3, false);
    THIntTensor_cfmod(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUIntType::remainder_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_remainder(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUIntType::remainder(const Tensor & self, Scalar other) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_remainder(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUIntType::s_remainder_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->remainder_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",2, false);
    THIntTensor_cremainder(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUIntType::s_remainder(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->remainder(self, Scalar(other));
    }
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",2, false);
    THIntTensor_cremainder(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUIntType::remainder_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toInt();
    THIntTensor_remainder(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUIntType::s_remainder_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->remainder_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",3, false);
    THIntTensor_cremainder(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUIntType::clamp_out(Tensor & result, const Tensor & self, Scalar min, Scalar max) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toInt();
    auto max_ = max.toInt();
    THIntTensor_clamp(result_->tensor, self_->tensor, min_, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUIntType::clamp(const Tensor & self, Scalar min, Scalar max) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toInt();
    auto max_ = max.toInt();
    THIntTensor_clamp(result_->tensor, self_->tensor, min_, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUIntType::clamp_(Tensor & self, Scalar min, Scalar max) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toInt();
    auto max_ = max.toInt();
    THIntTensor_clamp(self_->tensor, self_->tensor, min_, max_);
    return self;
}
Tensor & CPUIntType::clamp_min_out(Tensor & result, const Tensor & self, Scalar min) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toInt();
    THIntTensor_cmaxValue(result_->tensor, self_->tensor, min_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUIntType::clamp_min(const Tensor & self, Scalar min) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toInt();
    THIntTensor_cmaxValue(result_->tensor, self_->tensor, min_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUIntType::clamp_min_(Tensor & self, Scalar min) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toInt();
    THIntTensor_cmaxValue(self_->tensor, self_->tensor, min_);
    return self;
}
Tensor & CPUIntType::clamp_max_out(Tensor & result, const Tensor & self, Scalar max) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto max_ = max.toInt();
    THIntTensor_cminValue(result_->tensor, self_->tensor, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUIntType::clamp_max(const Tensor & self, Scalar max) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto max_ = max.toInt();
    THIntTensor_cminValue(result_->tensor, self_->tensor, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUIntType::clamp_max_(Tensor & self, Scalar max) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto max_ = max.toInt();
    THIntTensor_cminValue(self_->tensor, self_->tensor, max_);
    return self;
}
Tensor CPUIntType::_dot(const Tensor & self, const Tensor & tensor) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto tensor_ = checked_cast_tensor<CPUIntTensor>(tensor.pImpl,"tensor",2, false);
    return scalarTensor(convert<int>(THIntTensor_dot(self_->tensor, tensor_->tensor)));
}
Tensor & CPUIntType::tril_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    THIntTensor_tril(result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUIntType::tril(const Tensor & self, int64_t diagonal) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    THIntTensor_tril(result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUIntType::tril_(Tensor & self, int64_t diagonal) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    THIntTensor_tril(self_->tensor, self_->tensor, diagonal);
    return self;
}
Tensor & CPUIntType::triu_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    THIntTensor_triu(result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUIntType::triu(const Tensor & self, int64_t diagonal) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    THIntTensor_triu(result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUIntType::triu_(Tensor & self, int64_t diagonal) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    THIntTensor_triu(self_->tensor, self_->tensor, diagonal);
    return self;
}
Tensor & CPUIntType::cross_out(Tensor & result, const Tensor & self, const Tensor & other, int64_t dim) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",2, false);
    THIntTensor_cross(result_->tensor, self_->tensor, other_->tensor, dim);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUIntType::cross(const Tensor & self, const Tensor & other, int64_t dim) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUIntTensor>(other.pImpl,"other",2, false);
    THIntTensor_cross(result_->tensor, self_->tensor, other_->tensor, dim);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUIntType::diag_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    if (self_->isScalar()) {
      throw std::runtime_error("Input must be 1-d or 2-d");
    }
    THIntTensor_diag(result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUIntType::diag(const Tensor & self, int64_t diagonal) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    if (self_->isScalar()) {
      throw std::runtime_error("Input must be 1-d or 2-d");
    }
    THIntTensor_diag(result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUIntType::s_addmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    if(mat1.type().is_sparse()) {
        return static_cast<const Type*>(this)->addmm_out(result, self, SparseTensor(mat1), mat2, beta, alpha);
    }
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toInt();
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toInt();
    auto mat1_ = checked_cast_tensor<CPUIntTensor>(mat1.pImpl,"mat1",4, false);
    auto mat2_ = checked_cast_tensor<CPUIntTensor>(mat2.pImpl,"mat2",5, false);
    THIntTensor_addmm(result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CPUIntType::s_addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    if(mat1.type().is_sparse()) {
        return static_cast<const Type*>(this)->addmm(self, SparseTensor(mat1), mat2, beta, alpha);
    }
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toInt();
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toInt();
    auto mat1_ = checked_cast_tensor<CPUIntTensor>(mat1.pImpl,"mat1",4, false);
    auto mat2_ = checked_cast_tensor<CPUIntTensor>(mat2.pImpl,"mat2",5, false);
    THIntTensor_addmm(result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CPUIntType::addmm_out(Tensor & result, const Tensor & self, SparseTensor mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toInt();
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toInt();
    auto mat1_ = checked_cast_tensor<SparseCPUIntTensor>(mat1.tref.pImpl,"mat1",4,false);
    auto mat2_ = checked_cast_tensor<CPUIntTensor>(mat2.pImpl,"mat2",5, false);
    THSIntTensor_spaddmm(result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CPUIntType::addmm(const Tensor & self, SparseTensor mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toInt();
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toInt();
    auto mat1_ = checked_cast_tensor<SparseCPUIntTensor>(mat1.tref.pImpl,"mat1",4,false);
    auto mat2_ = checked_cast_tensor<CPUIntTensor>(mat2.pImpl,"mat2",5, false);
    THSIntTensor_spaddmm(result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CPUIntType::addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    if(mat1.type().is_sparse()) {
        return static_cast<const Type*>(this)->addmm_(self, SparseTensor(mat1), mat2, beta, alpha);
    }
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toInt();
    auto alpha_ = alpha.toInt();
    auto mat1_ = checked_cast_tensor<CPUIntTensor>(mat1.pImpl,"mat1",5, false);
    auto mat2_ = checked_cast_tensor<CPUIntTensor>(mat2.pImpl,"mat2",6, false);
    THIntTensor_addmm(self_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    return self;
}
Tensor & CPUIntType::addmm_(Tensor & self, SparseTensor mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toInt();
    auto alpha_ = alpha.toInt();
    auto mat1_ = checked_cast_tensor<SparseCPUIntTensor>(mat1.tref.pImpl,"mat1",5,false);
    auto mat2_ = checked_cast_tensor<CPUIntTensor>(mat2.pImpl,"mat2",6, false);
    THSIntTensor_spaddmm(self_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    return self;
}
Tensor & CPUIntType::s__addmv_out(Tensor & result, const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toInt();
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toInt();
    auto mat_ = checked_cast_tensor<CPUIntTensor>(mat.pImpl,"mat",4, false);
    auto vec_ = checked_cast_tensor<CPUIntTensor>(vec.pImpl,"vec",5, false);
    THIntTensor_addmv(result_->tensor, beta_, self_->tensor, alpha_, mat_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && mat_->isScalar() && vec_->isScalar());
    return result;
}
Tensor CPUIntType::s__addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toInt();
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toInt();
    auto mat_ = checked_cast_tensor<CPUIntTensor>(mat.pImpl,"mat",4, false);
    auto vec_ = checked_cast_tensor<CPUIntTensor>(vec.pImpl,"vec",5, false);
    THIntTensor_addmv(result_->tensor, beta_, self_->tensor, alpha_, mat_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && mat_->isScalar() && vec_->isScalar());
    return result;
}
Tensor & CPUIntType::_addmv_(Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toInt();
    auto alpha_ = alpha.toInt();
    auto mat_ = checked_cast_tensor<CPUIntTensor>(mat.pImpl,"mat",5, false);
    auto vec_ = checked_cast_tensor<CPUIntTensor>(vec.pImpl,"vec",6, false);
    THIntTensor_addmv(self_->tensor, beta_, self_->tensor, alpha_, mat_->tensor, vec_->tensor);
    return self;
}
Tensor & CPUIntType::s__addr_out(Tensor & result, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toInt();
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toInt();
    auto vec1_ = checked_cast_tensor<CPUIntTensor>(vec1.pImpl,"vec1",4, false);
    auto vec2_ = checked_cast_tensor<CPUIntTensor>(vec2.pImpl,"vec2",5, false);
    THIntTensor_addr(result_->tensor, beta_, self_->tensor, alpha_, vec1_->tensor, vec2_->tensor);
    result_->maybeScalar(self_->isScalar() && vec1_->isScalar() && vec2_->isScalar());
    return result;
}
Tensor CPUIntType::s__addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toInt();
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toInt();
    auto vec1_ = checked_cast_tensor<CPUIntTensor>(vec1.pImpl,"vec1",4, false);
    auto vec2_ = checked_cast_tensor<CPUIntTensor>(vec2.pImpl,"vec2",5, false);
    THIntTensor_addr(result_->tensor, beta_, self_->tensor, alpha_, vec1_->tensor, vec2_->tensor);
    result_->maybeScalar(self_->isScalar() && vec1_->isScalar() && vec2_->isScalar());
    return result;
}
Tensor & CPUIntType::_addr_(Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toInt();
    auto alpha_ = alpha.toInt();
    auto vec1_ = checked_cast_tensor<CPUIntTensor>(vec1.pImpl,"vec1",5, false);
    auto vec2_ = checked_cast_tensor<CPUIntTensor>(vec2.pImpl,"vec2",6, false);
    THIntTensor_addr(self_->tensor, beta_, self_->tensor, alpha_, vec1_->tensor, vec2_->tensor);
    return self;
}
Tensor & CPUIntType::_ger_out(Tensor & result, const Tensor & self, const Tensor & vec2) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.dim() == 0 ? 1 : self.size(0),vec2.dim() == 0 ? 1 : vec2.size(0) });
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto vec2_ = checked_cast_tensor<CPUIntTensor>(vec2.pImpl,"vec2",2, false);
    THIntTensor_addr(result_->tensor, int(0), result_->tensor, int(1), self_->tensor, vec2_->tensor);
    result_->maybeScalar(false);
    return result;
}
Tensor CPUIntType::_ger(const Tensor & self, const Tensor & vec2) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.dim() == 0 ? 1 : self.size(0),vec2.dim() == 0 ? 1 : vec2.size(0) });
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto vec2_ = checked_cast_tensor<CPUIntTensor>(vec2.pImpl,"vec2",2, false);
    THIntTensor_addr(result_->tensor, int(0), result_->tensor, int(1), self_->tensor, vec2_->tensor);
    result_->maybeScalar(false);
    return result;
}
Tensor & CPUIntType::_mv_out(Tensor & result, const Tensor & self, const Tensor & vec) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.size(0) });
    result.zero_();
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto vec_ = checked_cast_tensor<CPUIntTensor>(vec.pImpl,"vec",2, false);
    THIntTensor_addmv(result_->tensor, int(0), result_->tensor, int(1), self_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && vec_->isScalar());
    return result;
}
Tensor CPUIntType::_mv(const Tensor & self, const Tensor & vec) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.size(0) });
    result.zero_();
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto vec_ = checked_cast_tensor<CPUIntTensor>(vec.pImpl,"vec",2, false);
    THIntTensor_addmv(result_->tensor, int(0), result_->tensor, int(1), self_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && vec_->isScalar());
    return result;
}
Tensor & CPUIntType::_mm_out(Tensor & result, const Tensor & self, const Tensor & mat2) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.size(0),mat2.size(1) });
    result.zero_();
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CPUIntTensor>(mat2.pImpl,"mat2",2, false);
    THIntTensor_addmm(result_->tensor, int(0), result_->tensor, int(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CPUIntType::_mm(const Tensor & self, const Tensor & mat2) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.size(0),mat2.size(1) });
    result.zero_();
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CPUIntTensor>(mat2.pImpl,"mat2",2, false);
    THIntTensor_addmm(result_->tensor, int(0), result_->tensor, int(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CPUIntType::bmm_out(Tensor & result, const Tensor & self, const Tensor & mat2) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.size(0),self.size(1),mat2.size(2) });
    result.zero_();
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CPUIntTensor>(mat2.pImpl,"mat2",2, false);
    THIntTensor_baddbmm(result_->tensor, int(0), result_->tensor, int(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CPUIntType::bmm(const Tensor & self, const Tensor & mat2) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.size(0),self.size(1),mat2.size(2) });
    result.zero_();
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CPUIntTensor>(mat2.pImpl,"mat2",2, false);
    THIntTensor_baddbmm(result_->tensor, int(0), result_->tensor, int(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CPUIntType::s_addbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toInt();
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toInt();
    auto batch1_ = checked_cast_tensor<CPUIntTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CPUIntTensor>(batch2.pImpl,"batch2",5, false);
    THIntTensor_addbmm(result_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor CPUIntType::s_addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toInt();
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toInt();
    auto batch1_ = checked_cast_tensor<CPUIntTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CPUIntTensor>(batch2.pImpl,"batch2",5, false);
    THIntTensor_addbmm(result_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor & CPUIntType::addbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toInt();
    auto alpha_ = alpha.toInt();
    auto batch1_ = checked_cast_tensor<CPUIntTensor>(batch1.pImpl,"batch1",5, false);
    auto batch2_ = checked_cast_tensor<CPUIntTensor>(batch2.pImpl,"batch2",6, false);
    THIntTensor_addbmm(self_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    return self;
}
Tensor & CPUIntType::s_baddbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toInt();
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toInt();
    auto batch1_ = checked_cast_tensor<CPUIntTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CPUIntTensor>(batch2.pImpl,"batch2",5, false);
    THIntTensor_baddbmm(result_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor CPUIntType::s_baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toInt();
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toInt();
    auto batch1_ = checked_cast_tensor<CPUIntTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CPUIntTensor>(batch2.pImpl,"batch2",5, false);
    THIntTensor_baddbmm(result_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor & CPUIntType::baddbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toInt();
    auto alpha_ = alpha.toInt();
    auto batch1_ = checked_cast_tensor<CPUIntTensor>(batch1.pImpl,"batch1",5, false);
    auto batch2_ = checked_cast_tensor<CPUIntTensor>(batch2.pImpl,"batch2",6, false);
    THIntTensor_baddbmm(self_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    return self;
}
Tensor & CPUIntType::s_addcmul_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toInt();
    auto tensor1_ = checked_cast_tensor<CPUIntTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CPUIntTensor>(tensor2.pImpl,"tensor2",4, false);
    THIntTensor_addcmul(result_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor CPUIntType::s_addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toInt();
    auto tensor1_ = checked_cast_tensor<CPUIntTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CPUIntTensor>(tensor2.pImpl,"tensor2",4, false);
    THIntTensor_addcmul(result_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor & CPUIntType::s_addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toInt();
    auto tensor1_ = checked_cast_tensor<CPUIntTensor>(tensor1.pImpl,"tensor1",4, false);
    auto tensor2_ = checked_cast_tensor<CPUIntTensor>(tensor2.pImpl,"tensor2",5, false);
    THIntTensor_addcmul(self_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    return self;
}
Tensor & CPUIntType::s_addcdiv_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toInt();
    auto tensor1_ = checked_cast_tensor<CPUIntTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CPUIntTensor>(tensor2.pImpl,"tensor2",4, false);
    THIntTensor_addcdiv(result_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor CPUIntType::s_addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toInt();
    auto tensor1_ = checked_cast_tensor<CPUIntTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CPUIntTensor>(tensor2.pImpl,"tensor2",4, false);
    THIntTensor_addcdiv(result_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor & CPUIntType::s_addcdiv_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toInt();
    auto tensor1_ = checked_cast_tensor<CPUIntTensor>(tensor1.pImpl,"tensor1",4, false);
    auto tensor2_ = checked_cast_tensor<CPUIntTensor>(tensor2.pImpl,"tensor2",5, false);
    THIntTensor_addcdiv(self_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    return self;
}
Tensor & CPUIntType::random_(Tensor & self, int64_t from, int64_t to, Generator * generator) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    THIntTensor_clampedRandom(self_->tensor, generator_->generator, from, to);
    return self;
}
Tensor & CPUIntType::random_(Tensor & self, int64_t to, Generator * generator) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    THIntTensor_cappedRandom(self_->tensor, generator_->generator, to);
    return self;
}
Tensor & CPUIntType::random_(Tensor & self, Generator * generator) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    THIntTensor_random(self_->tensor, generator_->generator);
    return self;
}
Tensor & CPUIntType::geometric_(Tensor & self, double p, Generator * generator) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    THIntTensor_geometric(self_->tensor, generator_->generator, p);
    return self;
}
Tensor CPUIntType::tensor(Storage & storage, int64_t storageOffset, IntList size, IntList stride) const {
    auto storage_ = checked_cast_storage<CPUIntStorage>(&storage,"storage",1);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    return Tensor((new CPUIntTensor(context, THIntTensor_newWithStorage(storage_->storage, storageOffset, size_, stride_)))->maybeScalar(size.size() == 0),false);
}
Tensor CPUIntType::tensor(IntList size) const {
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new CPUIntTensor(context, THIntTensor_newWithSize(size_, NULL)))->maybeScalar(size.size() == 0),false);
}
Tensor CPUIntType::tensor(IntList size, IntList stride) const {
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    return Tensor((new CPUIntTensor(context, THIntTensor_newWithSize(size_, stride_)))->maybeScalar(size.size() == 0),false);
}
Tensor CPUIntType::tensor() const {
    return Tensor((new CPUIntTensor(context, THIntTensor_new())),false);
}
Tensor CPUIntType::alias(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPUIntTensor(context, THIntTensor_newWithTensor(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor & CPUIntType::as_strided_out(Tensor & result, const Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    auto result_ = checked_cast_tensor<CPUIntTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    if (storage_offset == -1) {
      storage_offset = self_->tensor->storageOffset;
    }
    THIntTensor_setStorage(result_->tensor, self_->tensor->storage, storage_offset, size_, stride_);
    result_->maybeScalar(size.size() == 0);
    return result;
}
Tensor CPUIntType::as_strided(const Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    auto result_ = new CPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    if (storage_offset == -1) {
      storage_offset = self_->tensor->storageOffset;
    }
    THIntTensor_setStorage(result_->tensor, self_->tensor->storage, storage_offset, size_, stride_);
    result_->maybeScalar(size.size() == 0);
    return result;
}
Tensor & CPUIntType::as_strided_(Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    if (storage_offset == -1) {
      storage_offset = self_->tensor->storageOffset;
    }
    THIntTensor_setStorage(self_->tensor, self_->tensor->storage, storage_offset, size_, stride_);
    self_->maybeScalar(size.size() == 0);
    return self;
}
Tensor & CPUIntType::_cat_out(Tensor & self, TensorList tensors, int64_t dim) const {
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",0, false);
    auto tensors_ = tensor_list_checked_cast<CPUIntTensor, Tensor, THIntTensor>(tensors,"tensors",1);
    THIntTensor_catArray(self_->tensor, tensors_.data(), tensors_.size(), dim);
    return self;
}
Tensor CPUIntType::_cat(TensorList tensors, int64_t dim) const {
    auto self_ = new CPUIntTensor(context);
    auto self = Tensor(self_, false);
    auto tensors_ = tensor_list_checked_cast<CPUIntTensor, Tensor, THIntTensor>(tensors,"tensors",1);
    THIntTensor_catArray(self_->tensor, tensors_.data(), tensors_.size(), dim);
    return self;
}
Tensor CPUIntType::_sparse_mask(const Tensor & self, SparseTensor mask) const {
    auto result_ = new SparseCPUIntTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUIntTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<SparseCPUIntTensor>(mask.tref.pImpl,"mask",2,false);
    THIntTensor_sparseMask(result_->tensor, self_->tensor, mask_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUIntType::abs_out(Tensor & result, const Tensor & self) const {
    return  at::native::_abs_out_cpu(result, self);
}
Tensor & CPUIntType::ceil_out(Tensor & result, const Tensor & self) const {
    return  at::native::_ceil_out_cpu(result, self);
}
Tensor & CPUIntType::cos_out(Tensor & result, const Tensor & self) const {
    return  at::native::_cos_out_cpu(result, self);
}
Tensor CPUIntType::embedding_dense_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) const {
    return  at::native::embedding_backward_cpu(grad, indices, num_weights, padding_idx, scale_grad_by_freq);
}
Tensor & CPUIntType::embedding_renorm_(Tensor & self, const Tensor & indices, double max_norm, double norm_type) const {
    return  at::native::embedding_renorm_cpu_(self, indices, max_norm, norm_type);
}
std::tuple<Tensor,Tensor,Tensor> CPUIntType::embedding_bag(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse) const {
    return  at::native::embedding_bag_cpu(weight, indices, offsets, scale_grad_by_freq, mode, sparse);
}
Tensor CPUIntType::embedding_bag_dense_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode) const {
    return  at::native::embedding_bag_backward_cpu(grad, indices, offsets, offset2bag, bag_size, num_weights, scale_grad_by_freq, mode);
}
Tensor & CPUIntType::exp_out(Tensor & result, const Tensor & self) const {
    return  at::native::_exp_out_cpu(result, self);
}
Tensor & CPUIntType::eye_out(Tensor & result, int64_t n, int64_t m) const {
    return  at::native::eye_out_cpu(result, n, m);
}
Tensor & CPUIntType::floor_out(Tensor & result, const Tensor & self) const {
    return  at::native::_floor_out_cpu(result, self);
}
Tensor CPUIntType::_fft_with_size(const Tensor & self, int64_t signal_ndim, bool complex_input, bool complex_output, bool inverse, IntList checked_signal_sizes, bool normalized, bool onesided, IntList output_sizes) const {
    return  at::native::_fft_mkl(self, signal_ndim, complex_input, complex_output, inverse, checked_signal_sizes, normalized, onesided, output_sizes);
}
Tensor & CPUIntType::log_out(Tensor & result, const Tensor & self) const {
    return  at::native::_log_out_cpu(result, self);
}
std::tuple<Tensor,Tensor> CPUIntType::RoiPooling2d_forward(const Tensor & input, const Tensor & rois, int64_t pooledHeight, int64_t pooledWidth, double spatialScale) const {
    return  at::native::RoiPooling2d_forward_cpu(input, rois, pooledHeight, pooledWidth, spatialScale);
}
Tensor CPUIntType::RoiPooling2d_backward(const Tensor & input, const Tensor & rois, int64_t pooledHeight, int64_t pooledWidth, double spatialScale, const Tensor & gradOutput, const Tensor & argmaxes) const {
    return  at::native::RoiPooling2d_backward_cpu(input, rois, pooledHeight, pooledWidth, spatialScale, gradOutput, argmaxes);
}
Tensor & CPUIntType::round_out(Tensor & result, const Tensor & self) const {
    return  at::native::_round_out_cpu(result, self);
}
Tensor & CPUIntType::sin_out(Tensor & result, const Tensor & self) const {
    return  at::native::_sin_out_cpu(result, self);
}
Tensor & CPUIntType::sspaddmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return  at::native::_sspaddmm_out_only_sparse(result, self, mat1, mat2, beta, alpha);
}
Tensor CPUIntType::sum(const Tensor & self) const {
    return  at::native::_sum_cpu(self);
}
Tensor & CPUIntType::sum_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    return  at::native::_sum_out_cpu(result, self, dim, keepdim);
}
Tensor & CPUIntType::sqrt_out(Tensor & result, const Tensor & self) const {
    return  at::native::_sqrt_out_cpu(result, self);
}
Tensor CPUIntType::prod(const Tensor & self) const {
    return  at::native::_prod_cpu(self);
}
Tensor & CPUIntType::prod_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    return  at::native::_prod_out_cpu(result, self, dim, keepdim);
}
Tensor & CPUIntType::trunc_out(Tensor & result, const Tensor & self) const {
    return  at::native::_trunc_out_cpu(result, self);
}
std::tuple<Tensor,Tensor> CPUIntType::_unique(const Tensor & self, bool sorted, bool return_inverse) const {
    return  at::native::_unique_cpu(self, sorted, return_inverse);
}
Tensor CPUIntType::_s_where(const Tensor & condition, const Tensor & self, const Tensor & other) const {
    return  at::native::_s_where_cpu(condition, self, other);
}
Tensor CPUIntType::_standard_gamma_grad(const Tensor & self, const Tensor & output) const {
    return  at::native::_standard_gamma_grad_cpu(self, output);
}
Tensor CPUIntType::poisson(const Tensor & self, Generator * generator) const {
    return  at::native::_s_poisson_cpu(self, generator);
}

}
