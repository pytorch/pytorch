// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/CPULongType.h"
#include "ATen/CPULongStorage.h"
#include "ATen/CPULongTensor.h"
#include "ATen/CPUGenerator.h"
#include "ATen/CPUByteTensor.h"
#include "ATen/CPUIntTensor.h"
#include "ATen/CPULongTensor.h"
#include "ATen/SparseCPULongTensor.h"
#include "ATen/CPULongTensor.h"
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

CPULongType::CPULongType(Context* context)
: Type(context, /*is_variable_or_undefined=*/false) {}
ScalarType CPULongType::scalarType() const {
  return ScalarType::Long;
}
Backend CPULongType::backend() const {
  return Backend::CPU;
}
bool CPULongType::is_cuda() const { return backend() == kCUDA || backend() == kSparseCUDA; }
bool CPULongType::is_sparse() const { return backend() == kSparseCPU || backend() == kSparseCUDA; }
bool CPULongType::is_distributed() const { return false; }

std::unique_ptr<Storage> CPULongType::storage() const {
  return std::unique_ptr<Storage>(new CPULongStorage(context));
}
std::unique_ptr<Storage> CPULongType::storage(size_t size) const {
  return std::unique_ptr<Storage>(new CPULongStorage(context,size));
}
std::unique_ptr<Storage> CPULongType::storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const {
    return std::unique_ptr<Storage>(
      new CPULongStorage(context,data,size,deleter));
}
std::unique_ptr<Storage> CPULongType::storageWithAllocator(int64_t size, std::unique_ptr<Allocator> allocator) const {
    return std::unique_ptr<Storage>(
        new CPULongStorage(context, size, std::move(allocator)));
}
Tensor CPULongType::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THLongTensor_retain( (THLongTensor*) th_pointer);
  return Tensor(new CPULongTensor(context,(THLongTensor*)(th_pointer)), false);
}
std::unique_ptr<Storage> CPULongType::unsafeStorageFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THLongStorage_retain( (THLongStorage*) th_pointer);
  return std::unique_ptr<Storage>(new CPULongStorage(context, (THLongStorage*) th_pointer));
}
std::unique_ptr<Generator> CPULongType::generator() const {
  return std::unique_ptr<Generator>(new CPUGenerator(context));
}

const char * CPULongType::toString() const {
  return CPULongType::typeString();
}
TypeID CPULongType::ID() const {
  return TypeID::CPULong;
}

std::size_t CPULongType::elementSizeInBytes() const {
  return sizeof(int64_t);
}

const char * CPULongType::typeString() {
  return "CPULongType";
}

/* example
Tensor * CPULongType::add(Tensor & a, Tensor & b) {
  std::cout << "add CPULongTensor\n";
  return &a;
}
*/

int64_t CPULongType::storage_offset(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    return static_cast<int64_t>(THLongTensor_storageOffset(self_->tensor));
}
Tensor & CPULongType::resize_(Tensor & self, IntList size) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    THLongTensor_resize(self_->tensor, size_, NULL);
    self_->maybeScalar(size.size() == 0);
    return self;
}
int64_t CPULongType::numel(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    return static_cast<int64_t>(THLongTensor_nElement(self_->tensor));
}
Tensor & CPULongType::set_(Tensor & self, Storage & storage) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto storage_ = checked_cast_storage<CPULongStorage>(&storage,"storage",2);
    THLongTensor_setStorage(self_->tensor, storage_->storage, 0, THLongStorageView::makeFromLength(static_cast<int64_t>(storage.size())), NULL);
    self_->maybeScalar(false);
    return self;
}
Tensor & CPULongType::set_(Tensor & self, Storage & sourceStorage, int64_t storage_offset, IntList size, IntList stride) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto sourceStorage_ = checked_cast_storage<CPULongStorage>(&sourceStorage,"sourceStorage",2);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    THLongTensor_setStorage(self_->tensor, sourceStorage_->storage, storage_offset, size_, stride_);
    self_->maybeScalar(size.size() == 0);
    return self;
}
Tensor & CPULongType::set_(Tensor & self, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto source_ = checked_cast_tensor<CPULongTensor>(source.pImpl,"source",2, false);
    THLongTensor_set(self_->tensor, source_->tensor);
    self_->maybeScalar(source_->isScalar());
    return self;
}
Tensor & CPULongType::set_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    THLongTensor_setStorage(self_->tensor, NULL, 0, NULL, NULL);
    self_->maybeScalar(false);
    return self;
}
Tensor & CPULongType::fill_(Tensor & self, Scalar value) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toLong();
    THLongTensor_fill(self_->tensor, value_);
    return self;
}
Tensor & CPULongType::fill_(Tensor & self, const Tensor & value) const {
    if (value.dim() == 0) {
        return static_cast<const Type*>(this)->fill_(self, Scalar(value));
    }
    AT_ERROR("fill_ only supports a 0-dimensional value tensor, but got tensor "
        "with %" PRId64 " dimension(s)", value.dim());
}
bool CPULongType::is_contiguous(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    return THLongTensor_isContiguous(self_->tensor);
}
bool CPULongType::is_set_to(const Tensor & self, const Tensor & tensor) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto tensor_ = checked_cast_tensor<CPULongTensor>(tensor.pImpl,"tensor",2, false);
    return THLongTensor_isSetTo(self_->tensor, tensor_->tensor);
}
Tensor & CPULongType::s_masked_fill_(Tensor & self, const Tensor & mask, Scalar value) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CPUByteTensor>(mask.pImpl,"mask",2, false);
    auto value_ = value.toLong();
    THLongTensor_maskedFill(self_->tensor, mask_->tensor, value_);
    return self;
}
Tensor & CPULongType::s_masked_fill_(Tensor & self, const Tensor & mask, const Tensor & value) const {
    if (value.dim() == 0) {
        return static_cast<const Type*>(this)->masked_fill_(self, mask, Scalar(value));
    }
    AT_ERROR("masked_fill_ only supports a 0-dimensional value tensor, but got tensor "
        "with %" PRId64 " dimension(s)", value.dim());
}
Tensor & CPULongType::s_masked_scatter_(Tensor & self, const Tensor & mask, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CPUByteTensor>(mask.pImpl,"mask",2, false);
    auto source_ = checked_cast_tensor<CPULongTensor>(source.pImpl,"source",3, false);
    THLongTensor_maskedCopy(self_->tensor, mask_->tensor, source_->tensor);
    return self;
}
Tensor & CPULongType::s_masked_select_out(Tensor & result, const Tensor & self, const Tensor & mask) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CPUByteTensor>(mask.pImpl,"mask",2, false);
    THLongTensor_maskedSelect(result_->tensor, self_->tensor, mask_->tensor);
    result_->maybeScalar(self_->isScalar() && mask_->isScalar());
    return result;
}
Tensor CPULongType::s_masked_select(const Tensor & self, const Tensor & mask) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CPUByteTensor>(mask.pImpl,"mask",2, false);
    THLongTensor_maskedSelect(result_->tensor, self_->tensor, mask_->tensor);
    result_->maybeScalar(self_->isScalar() && mask_->isScalar());
    return result;
}
Tensor CPULongType::transpose(const Tensor & self, int64_t dim0, int64_t dim1) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    dim0 = maybe_wrap_dim(dim0, self_);
    dim1 = maybe_wrap_dim(dim1, self_);
    return Tensor((new CPULongTensor(context, THLongTensor_newTranspose(self_->tensor, dim0, dim1)))->maybeScalar(self_->isScalar()),false);
}
Tensor CPULongType::t(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPULongTensor(context, THLongTensor_newTranspose(self_->tensor, 0, 1)))->maybeScalar(self_->isScalar()),false);
}
Tensor & CPULongType::nonzero_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    THLongTensor_nonzero(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPULongType::nonzero(const Tensor & self) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    THLongTensor_nonzero(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPULongType::contiguous(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPULongTensor(context, THLongTensor_newContiguous(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor CPULongType::clone(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPULongTensor(context, THLongTensor_newClone(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor CPULongType::view(const Tensor & self, IntList size) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new CPULongTensor(context, THLongTensor_newView(self_->tensor, size_)))->maybeScalar(size.size() == 0),false);
}
Tensor & CPULongType::resize_as_(Tensor & self, const Tensor & the_template) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto the_template_ = checked_cast_tensor<CPULongTensor>(the_template.pImpl,"the_template",2, false);
    THLongTensor_resizeAs(self_->tensor, the_template_->tensor);
    self_->maybeScalar(the_template_->isScalar());
    return self;
}
Tensor & CPULongType::index_select_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    THLongTensor_indexSelect(result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
Tensor CPULongType::index_select(const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    THLongTensor_indexSelect(result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
Tensor & CPULongType::_indexCopy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    auto source_ = checked_cast_tensor<CPULongTensor>(source.pImpl,"source",4, false);
    THLongTensor_indexCopy(self_->tensor, dim, index_->tensor, source_->tensor);
    return self;
}
Tensor & CPULongType::take_out(Tensor & result, const Tensor & self, const Tensor & index) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",2, false);
    THLongTensor_take(result_->tensor, self_->tensor, index_->tensor);
    result_->maybeScalar(index_->isScalar());
    return result;
}
Tensor CPULongType::take(const Tensor & self, const Tensor & index) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",2, false);
    THLongTensor_take(result_->tensor, self_->tensor, index_->tensor);
    result_->maybeScalar(index_->isScalar());
    return result;
}
Tensor & CPULongType::put_(Tensor & self, const Tensor & index, const Tensor & source, bool accumulate) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",2, false);
    auto source_ = checked_cast_tensor<CPULongTensor>(source.pImpl,"source",3, false);
    THLongTensor_put(self_->tensor, index_->tensor, source_->tensor, accumulate);
    return self;
}
Tensor & CPULongType::index_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    auto source_ = checked_cast_tensor<CPULongTensor>(source.pImpl,"source",4, false);
    THLongTensor_indexAdd(self_->tensor, dim, index_->tensor, source_->tensor);
    return self;
}
Tensor & CPULongType::index_fill_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    auto value_ = value.toLong();
    THLongTensor_indexFill(self_->tensor, dim, index_->tensor, value_);
    return self;
}
Tensor & CPULongType::index_fill_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & value) const {
    if (value.dim() == 0) {
        return static_cast<const Type*>(this)->index_fill_(self, dim, index, Scalar(value));
    }
    AT_ERROR("index_fill_ only supports a 0-dimensional value tensor, but got tensor "
        "with %" PRId64 " dimension(s)", value.dim());
}
Tensor CPULongType::unfold(const Tensor & self, int64_t dimension, int64_t size, int64_t step) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    dimension = maybe_wrap_dim(dimension, self_);
    THLongTensor_unfold(result_->tensor, self_->tensor, dimension, size, step);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPULongType::_range_out(Tensor & result, Scalar start, Scalar end, Scalar step) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto start_ = start.toLong();
    auto end_ = end.toLong();
    auto step_ = step.toLong();
    THLongTensor_range(result_->tensor, start_, end_, step_);
    return result;
}
Tensor CPULongType::_range(Scalar start, Scalar end, Scalar step) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto start_ = start.toLong();
    auto end_ = end.toLong();
    auto step_ = step.toLong();
    THLongTensor_range(result_->tensor, start_, end_, step_);
    return result;
}
Tensor & CPULongType::_arange_out(Tensor & result, Scalar start, Scalar end, Scalar step) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto start_ = start.toLong();
    auto end_ = end.toLong();
    auto step_ = step.toLong();
    THLongTensor_arange(result_->tensor, start_, end_, step_);
    return result;
}
Tensor CPULongType::_arange(Scalar start, Scalar end, Scalar step) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto start_ = start.toLong();
    auto end_ = end.toLong();
    auto step_ = step.toLong();
    THLongTensor_arange(result_->tensor, start_, end_, step_);
    return result;
}
Tensor & CPULongType::_arange_out(Tensor & result, Scalar end) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto end_ = end.toLong();
    THLongTensor_arange(result_->tensor, 0, end_, 1);
    return result;
}
Tensor CPULongType::_arange(Scalar end) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto end_ = end.toLong();
    THLongTensor_arange(result_->tensor, 0, end_, 1);
    return result;
}
Tensor & CPULongType::scatter_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
    if (src.dim() == 0) {
        return static_cast<const Type*>(this)->scatter_(self, dim, index, Scalar(src));
    }
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    auto src_ = checked_cast_tensor<CPULongTensor>(src.pImpl,"src",4, false);
    THLongTensor_scatter(self_->tensor, dim, index_->tensor, src_->tensor);
    return self;
}
Tensor & CPULongType::scatter_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    auto value_ = value.toLong();
    THLongTensor_scatterFill(self_->tensor, dim, index_->tensor, value_);
    return self;
}
Tensor & CPULongType::scatter_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    auto src_ = checked_cast_tensor<CPULongTensor>(src.pImpl,"src",4, false);
    THLongTensor_scatterAdd(self_->tensor, dim, index_->tensor, src_->tensor);
    return self;
}
Tensor & CPULongType::gather_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    result.resize_(index.sizes());
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    THLongTensor_gather(result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
Tensor CPULongType::gather(const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    result.resize_(index.sizes());
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    THLongTensor_gather(result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
void* CPULongType::data_ptr(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    return THLongTensor_data(self_->tensor);
}
bool CPULongType::equal(const Tensor & self, const Tensor & other) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",2, false);
    return THLongTensor_equal(self_->tensor, other_->tensor);
}
Tensor & CPULongType::__and___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_bitand(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPULongType::__and__(const Tensor & self, Scalar other) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_bitand(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPULongType::s___and___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__and___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",2, false);
    THLongTensor_cbitand(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPULongType::s___and__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__and__(self, Scalar(other));
    }
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",2, false);
    THLongTensor_cbitand(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPULongType::__iand__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_bitand(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPULongType::s___iand__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__iand__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",3, false);
    THLongTensor_cbitand(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPULongType::__or___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_bitor(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPULongType::__or__(const Tensor & self, Scalar other) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_bitor(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPULongType::s___or___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__or___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",2, false);
    THLongTensor_cbitor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPULongType::s___or__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__or__(self, Scalar(other));
    }
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",2, false);
    THLongTensor_cbitor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPULongType::__ior__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_bitor(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPULongType::s___ior__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__ior__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",3, false);
    THLongTensor_cbitor(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPULongType::__xor___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_bitxor(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPULongType::__xor__(const Tensor & self, Scalar other) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_bitxor(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPULongType::s___xor___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__xor___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",2, false);
    THLongTensor_cbitxor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPULongType::s___xor__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__xor__(self, Scalar(other));
    }
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",2, false);
    THLongTensor_cbitxor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPULongType::__ixor__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_bitxor(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPULongType::s___ixor__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__ixor__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",3, false);
    THLongTensor_cbitxor(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPULongType::__lshift___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_lshift(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPULongType::__lshift__(const Tensor & self, Scalar other) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_lshift(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPULongType::s___lshift___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__lshift___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",2, false);
    THLongTensor_clshift(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPULongType::s___lshift__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__lshift__(self, Scalar(other));
    }
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",2, false);
    THLongTensor_clshift(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPULongType::__ilshift__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_lshift(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPULongType::s___ilshift__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__ilshift__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",3, false);
    THLongTensor_clshift(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPULongType::__rshift___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_rshift(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPULongType::__rshift__(const Tensor & self, Scalar other) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_rshift(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPULongType::s___rshift___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__rshift___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",2, false);
    THLongTensor_crshift(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPULongType::s___rshift__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__rshift__(self, Scalar(other));
    }
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",2, false);
    THLongTensor_crshift(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPULongType::__irshift__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_rshift(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPULongType::s___irshift__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__irshift__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",3, false);
    THLongTensor_crshift(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPULongType::lt_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_ltValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPULongType::lt(const Tensor & self, Scalar other) const {
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_ltValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPULongType::s_lt_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->lt_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",2, false);
    THLongTensor_ltTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPULongType::s_lt(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->lt(self, Scalar(other));
    }
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",2, false);
    THLongTensor_ltTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPULongType::lt_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_ltValueT(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPULongType::s_lt_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->lt_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",3, false);
    THLongTensor_ltTensorT(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPULongType::gt_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_gtValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPULongType::gt(const Tensor & self, Scalar other) const {
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_gtValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPULongType::s_gt_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->gt_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",2, false);
    THLongTensor_gtTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPULongType::s_gt(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->gt(self, Scalar(other));
    }
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",2, false);
    THLongTensor_gtTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPULongType::gt_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_gtValueT(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPULongType::s_gt_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->gt_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",3, false);
    THLongTensor_gtTensorT(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPULongType::le_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_leValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPULongType::le(const Tensor & self, Scalar other) const {
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_leValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPULongType::s_le_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->le_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",2, false);
    THLongTensor_leTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPULongType::s_le(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->le(self, Scalar(other));
    }
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",2, false);
    THLongTensor_leTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPULongType::le_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_leValueT(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPULongType::s_le_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->le_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",3, false);
    THLongTensor_leTensorT(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPULongType::ge_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_geValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPULongType::ge(const Tensor & self, Scalar other) const {
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_geValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPULongType::s_ge_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ge_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",2, false);
    THLongTensor_geTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPULongType::s_ge(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ge(self, Scalar(other));
    }
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",2, false);
    THLongTensor_geTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPULongType::ge_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_geValueT(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPULongType::s_ge_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ge_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",3, false);
    THLongTensor_geTensorT(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPULongType::eq_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_eqValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPULongType::eq(const Tensor & self, Scalar other) const {
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_eqValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPULongType::s_eq_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->eq_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",2, false);
    THLongTensor_eqTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPULongType::s_eq(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->eq(self, Scalar(other));
    }
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",2, false);
    THLongTensor_eqTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPULongType::eq_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_eqValueT(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPULongType::s_eq_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->eq_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",3, false);
    THLongTensor_eqTensorT(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPULongType::ne_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_neValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPULongType::ne(const Tensor & self, Scalar other) const {
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_neValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPULongType::s_ne_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ne_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",2, false);
    THLongTensor_neTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPULongType::s_ne(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ne(self, Scalar(other));
    }
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",2, false);
    THLongTensor_neTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPULongType::ne_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_neValueT(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPULongType::s_ne_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ne_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",3, false);
    THLongTensor_neTensorT(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
std::tuple<Tensor &,Tensor &> CPULongType::min_out(Tensor & min, Tensor & min_indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto min_ = checked_cast_tensor<CPULongTensor>(min.pImpl,"min",0, false);
    auto min_indices_ = checked_cast_tensor<CPULongTensor>(min_indices.pImpl,"min_indices",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THLongTensor_min(min_->tensor, min_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    min_->maybeScalar(maybe_scalar);
    min_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(min, min_indices);
}
std::tuple<Tensor,Tensor> CPULongType::min(const Tensor & self, int64_t dim, bool keepdim) const {
    auto min_ = new CPULongTensor(context);
    auto min = Tensor(min_, false);
    auto min_indices_ = new CPULongTensor(context);
    auto min_indices = Tensor(min_indices_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THLongTensor_min(min_->tensor, min_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    min_->maybeScalar(maybe_scalar);
    min_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(min, min_indices);
}
Tensor & CPULongType::s_min_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",2, false);
    THLongTensor_cmin(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPULongType::s_min(const Tensor & self, const Tensor & other) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",2, false);
    THLongTensor_cmin(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPULongType::min(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<int64_t>(THLongTensor_minall(self_->tensor)));
}
std::tuple<Tensor &,Tensor &> CPULongType::max_out(Tensor & max, Tensor & max_indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto max_ = checked_cast_tensor<CPULongTensor>(max.pImpl,"max",0, false);
    auto max_indices_ = checked_cast_tensor<CPULongTensor>(max_indices.pImpl,"max_indices",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THLongTensor_max(max_->tensor, max_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    max_->maybeScalar(maybe_scalar);
    max_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(max, max_indices);
}
std::tuple<Tensor,Tensor> CPULongType::max(const Tensor & self, int64_t dim, bool keepdim) const {
    auto max_ = new CPULongTensor(context);
    auto max = Tensor(max_, false);
    auto max_indices_ = new CPULongTensor(context);
    auto max_indices = Tensor(max_indices_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THLongTensor_max(max_->tensor, max_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    max_->maybeScalar(maybe_scalar);
    max_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(max, max_indices);
}
Tensor & CPULongType::s_max_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",2, false);
    THLongTensor_cmax(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPULongType::s_max(const Tensor & self, const Tensor & other) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",2, false);
    THLongTensor_cmax(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPULongType::max(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<int64_t>(THLongTensor_maxall(self_->tensor)));
}
std::tuple<Tensor &,Tensor &> CPULongType::kthvalue_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool keepdim) const {
    auto values_ = checked_cast_tensor<CPULongTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THLongTensor_kthvalue(values_->tensor, indices_->tensor, self_->tensor, k, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CPULongType::kthvalue(const Tensor & self, int64_t k, int64_t dim, bool keepdim) const {
    auto values_ = new CPULongTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CPULongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THLongTensor_kthvalue(values_->tensor, indices_->tensor, self_->tensor, k, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
std::tuple<Tensor &,Tensor &> CPULongType::mode_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = checked_cast_tensor<CPULongTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THLongTensor_mode(values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CPULongType::mode(const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = new CPULongTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CPULongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THLongTensor_mode(values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
std::tuple<Tensor &,Tensor &> CPULongType::median_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = checked_cast_tensor<CPULongTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THLongTensor_median(values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CPULongType::median(const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = new CPULongTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CPULongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THLongTensor_median(values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
Tensor CPULongType::median(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<int64_t>(THLongTensor_medianall(self_->tensor)));
}
std::tuple<Tensor &,Tensor &> CPULongType::sort_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool descending) const {
    auto values_ = checked_cast_tensor<CPULongTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THLongTensor_sort(values_->tensor, indices_->tensor, self_->tensor, dim, descending);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CPULongType::sort(const Tensor & self, int64_t dim, bool descending) const {
    auto values_ = new CPULongTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CPULongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THLongTensor_sort(values_->tensor, indices_->tensor, self_->tensor, dim, descending);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
std::tuple<Tensor &,Tensor &> CPULongType::topk_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) const {
    auto values_ = checked_cast_tensor<CPULongTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THLongTensor_topk(values_->tensor, indices_->tensor, self_->tensor, k, dim, largest, sorted);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CPULongType::topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) const {
    auto values_ = new CPULongTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CPULongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THLongTensor_topk(values_->tensor, indices_->tensor, self_->tensor, k, dim, largest, sorted);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
Tensor & CPULongType::_abs_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    THLongTensor_abs(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPULongType::_abs(const Tensor & self) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    THLongTensor_abs(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPULongType::neg_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    THLongTensor_neg(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPULongType::neg(const Tensor & self) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    THLongTensor_neg(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPULongType::neg_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    THLongTensor_neg(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPULongType::pow_out(Tensor & result, const Tensor & self, Scalar exponent) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toLong();
    THLongTensor_pow(result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPULongType::pow(const Tensor & self, Scalar exponent) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toLong();
    THLongTensor_pow(result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPULongType::s_pow_out(Tensor & result, const Tensor & self, const Tensor & exponent) const {
    if (exponent.dim() == 0) {
        return static_cast<const Type*>(this)->pow_out(result, self, Scalar(exponent));
    }
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto exponent_ = checked_cast_tensor<CPULongTensor>(exponent.pImpl,"exponent",2, false);
    THLongTensor_cpow(result_->tensor, self_->tensor, exponent_->tensor);
    result_->maybeScalar(self_->isScalar() && exponent_->isScalar());
    return result;
}
Tensor CPULongType::s_pow(const Tensor & self, const Tensor & exponent) const {
    if (exponent.dim() == 0) {
        return static_cast<const Type*>(this)->pow(self, Scalar(exponent));
    }
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto exponent_ = checked_cast_tensor<CPULongTensor>(exponent.pImpl,"exponent",2, false);
    THLongTensor_cpow(result_->tensor, self_->tensor, exponent_->tensor);
    result_->maybeScalar(self_->isScalar() && exponent_->isScalar());
    return result;
}
Tensor & CPULongType::pow_out(Tensor & result, Scalar base, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto base_ = base.toLong();
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",2, false);
    THLongTensor_tpow(result_->tensor, base_, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPULongType::pow(Scalar base, const Tensor & self) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto base_ = base.toLong();
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",2, false);
    THLongTensor_tpow(result_->tensor, base_, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPULongType::pow_(Tensor & self, Scalar exponent) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toLong();
    THLongTensor_pow(self_->tensor, self_->tensor, exponent_);
    return self;
}
Tensor & CPULongType::s_pow_(Tensor & self, const Tensor & exponent) const {
    if (exponent.dim() == 0) {
        return static_cast<const Type*>(this)->pow_(self, Scalar(exponent));
    }
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto exponent_ = checked_cast_tensor<CPULongTensor>(exponent.pImpl,"exponent",3, false);
    THLongTensor_cpow(self_->tensor, self_->tensor, exponent_->tensor);
    return self;
}
Tensor & CPULongType::zero_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    THLongTensor_zero(self_->tensor);
    return self;
}
Tensor CPULongType::_sumall(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<int64_t>(THLongTensor_sumall(self_->tensor)));
}
Tensor & CPULongType::_sum_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THLongTensor_sum(result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CPULongType::_sum(const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THLongTensor_sum(result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CPULongType::_prodall(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<int64_t>(THLongTensor_prodall(self_->tensor)));
}
Tensor & CPULongType::_prod_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THLongTensor_prod(result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CPULongType::_prod(const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THLongTensor_prod(result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor & CPULongType::cumsum_out(Tensor & result, const Tensor & self, int64_t dim) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THLongTensor_cumsum(result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPULongType::cumsum(const Tensor & self, int64_t dim) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THLongTensor_cumsum(result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPULongType::cumprod_out(Tensor & result, const Tensor & self, int64_t dim) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THLongTensor_cumprod(result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPULongType::cumprod(const Tensor & self, int64_t dim) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THLongTensor_cumprod(result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPULongType::sign_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    THLongTensor_sign(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPULongType::sign(const Tensor & self) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    THLongTensor_sign(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPULongType::sign_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    THLongTensor_sign(self_->tensor, self_->tensor);
    return self;
}
Tensor CPULongType::trace(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<int64_t>(THLongTensor_trace(self_->tensor)));
}
Tensor & CPULongType::add_out(Tensor & result, const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    auto alpha_ = alpha.toLong();
    THLongTensor_add_scaled(result_->tensor, self_->tensor, other_, alpha_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPULongType::add(const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    auto alpha_ = alpha.toLong();
    THLongTensor_add_scaled(result_->tensor, self_->tensor, other_, alpha_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPULongType::s_add_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if(other.type().is_sparse()) {
        return static_cast<const Type*>(this)->add_out(result, self, SparseTensor(other), alpha);
    }
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toLong();
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",3, false);
    THLongTensor_cadd(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPULongType::s_add(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if(other.type().is_sparse()) {
        return static_cast<const Type*>(this)->add(self, SparseTensor(other), alpha);
    }
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add(self, Scalar(other), alpha);
    }
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toLong();
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",3, false);
    THLongTensor_cadd(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPULongType::add_out(Tensor & result, const Tensor & self, SparseTensor other, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toLong();
    auto other_ = checked_cast_tensor<SparseCPULongTensor>(other.tref.pImpl,"other",3,false);
    THSLongTensor_spcadd(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPULongType::add(const Tensor & self, SparseTensor other, Scalar alpha) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toLong();
    auto other_ = checked_cast_tensor<SparseCPULongTensor>(other.tref.pImpl,"other",3,false);
    THSLongTensor_spcadd(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPULongType::add_(Tensor & self, Scalar other, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    auto alpha_ = alpha.toLong();
    THLongTensor_add_scaled(self_->tensor, self_->tensor, other_, alpha_);
    return self;
}
Tensor & CPULongType::s_add_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if(other.type().is_sparse()) {
        return static_cast<const Type*>(this)->add_(self, SparseTensor(other), alpha);
    }
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toLong();
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",4, false);
    THLongTensor_cadd(self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & CPULongType::add_(Tensor & self, SparseTensor other, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toLong();
    auto other_ = checked_cast_tensor<SparseCPULongTensor>(other.tref.pImpl,"other",4,false);
    THSLongTensor_spcadd(self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & CPULongType::sub_out(Tensor & result, const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    auto alpha_ = alpha.toLong();
    THLongTensor_sub_scaled(result_->tensor, self_->tensor, other_, alpha_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPULongType::sub(const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    auto alpha_ = alpha.toLong();
    THLongTensor_sub_scaled(result_->tensor, self_->tensor, other_, alpha_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPULongType::s_sub_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toLong();
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",3, false);
    THLongTensor_csub(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPULongType::s_sub(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub(self, Scalar(other), alpha);
    }
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toLong();
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",3, false);
    THLongTensor_csub(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPULongType::sub_(Tensor & self, Scalar other, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    auto alpha_ = alpha.toLong();
    THLongTensor_sub_scaled(self_->tensor, self_->tensor, other_, alpha_);
    return self;
}
Tensor & CPULongType::s_sub_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toLong();
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",4, false);
    THLongTensor_csub(self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & CPULongType::mul_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_mul(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPULongType::mul(const Tensor & self, Scalar other) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_mul(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPULongType::s_mul_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",2, false);
    THLongTensor_cmul(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPULongType::s_mul(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul(self, Scalar(other));
    }
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",2, false);
    THLongTensor_cmul(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPULongType::mul_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_mul(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPULongType::s_mul_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",3, false);
    THLongTensor_cmul(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPULongType::div_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_div(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPULongType::div(const Tensor & self, Scalar other) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_div(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPULongType::s_div_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->div_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",2, false);
    THLongTensor_cdiv(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPULongType::s_div(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->div(self, Scalar(other));
    }
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",2, false);
    THLongTensor_cdiv(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPULongType::div_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_div(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPULongType::s_div_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->div_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",3, false);
    THLongTensor_cdiv(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPULongType::fmod_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_fmod(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPULongType::fmod(const Tensor & self, Scalar other) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_fmod(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPULongType::s_fmod_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->fmod_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",2, false);
    THLongTensor_cfmod(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPULongType::s_fmod(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->fmod(self, Scalar(other));
    }
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",2, false);
    THLongTensor_cfmod(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPULongType::fmod_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_fmod(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPULongType::s_fmod_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->fmod_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",3, false);
    THLongTensor_cfmod(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPULongType::remainder_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_remainder(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPULongType::remainder(const Tensor & self, Scalar other) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_remainder(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPULongType::s_remainder_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->remainder_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",2, false);
    THLongTensor_cremainder(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPULongType::s_remainder(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->remainder(self, Scalar(other));
    }
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",2, false);
    THLongTensor_cremainder(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPULongType::remainder_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toLong();
    THLongTensor_remainder(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPULongType::s_remainder_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->remainder_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",3, false);
    THLongTensor_cremainder(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPULongType::clamp_out(Tensor & result, const Tensor & self, Scalar min, Scalar max) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toLong();
    auto max_ = max.toLong();
    THLongTensor_clamp(result_->tensor, self_->tensor, min_, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPULongType::clamp(const Tensor & self, Scalar min, Scalar max) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toLong();
    auto max_ = max.toLong();
    THLongTensor_clamp(result_->tensor, self_->tensor, min_, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPULongType::clamp_(Tensor & self, Scalar min, Scalar max) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toLong();
    auto max_ = max.toLong();
    THLongTensor_clamp(self_->tensor, self_->tensor, min_, max_);
    return self;
}
Tensor & CPULongType::clamp_min_out(Tensor & result, const Tensor & self, Scalar min) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toLong();
    THLongTensor_cmaxValue(result_->tensor, self_->tensor, min_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPULongType::clamp_min(const Tensor & self, Scalar min) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toLong();
    THLongTensor_cmaxValue(result_->tensor, self_->tensor, min_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPULongType::clamp_min_(Tensor & self, Scalar min) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toLong();
    THLongTensor_cmaxValue(self_->tensor, self_->tensor, min_);
    return self;
}
Tensor & CPULongType::clamp_max_out(Tensor & result, const Tensor & self, Scalar max) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto max_ = max.toLong();
    THLongTensor_cminValue(result_->tensor, self_->tensor, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPULongType::clamp_max(const Tensor & self, Scalar max) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto max_ = max.toLong();
    THLongTensor_cminValue(result_->tensor, self_->tensor, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPULongType::clamp_max_(Tensor & self, Scalar max) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto max_ = max.toLong();
    THLongTensor_cminValue(self_->tensor, self_->tensor, max_);
    return self;
}
Tensor CPULongType::_dot(const Tensor & self, const Tensor & tensor) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto tensor_ = checked_cast_tensor<CPULongTensor>(tensor.pImpl,"tensor",2, false);
    return scalarTensor(convert<int64_t>(THLongTensor_dot(self_->tensor, tensor_->tensor)));
}
Tensor & CPULongType::tril_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    THLongTensor_tril(result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPULongType::tril(const Tensor & self, int64_t diagonal) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    THLongTensor_tril(result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPULongType::tril_(Tensor & self, int64_t diagonal) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    THLongTensor_tril(self_->tensor, self_->tensor, diagonal);
    return self;
}
Tensor & CPULongType::triu_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    THLongTensor_triu(result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPULongType::triu(const Tensor & self, int64_t diagonal) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    THLongTensor_triu(result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPULongType::triu_(Tensor & self, int64_t diagonal) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    THLongTensor_triu(self_->tensor, self_->tensor, diagonal);
    return self;
}
Tensor & CPULongType::cross_out(Tensor & result, const Tensor & self, const Tensor & other, int64_t dim) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",2, false);
    THLongTensor_cross(result_->tensor, self_->tensor, other_->tensor, dim);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPULongType::cross(const Tensor & self, const Tensor & other, int64_t dim) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPULongTensor>(other.pImpl,"other",2, false);
    THLongTensor_cross(result_->tensor, self_->tensor, other_->tensor, dim);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPULongType::diag_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    if (self_->isScalar()) {
      throw std::runtime_error("Input must be 1-d or 2-d");
    }
    THLongTensor_diag(result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPULongType::diag(const Tensor & self, int64_t diagonal) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    if (self_->isScalar()) {
      throw std::runtime_error("Input must be 1-d or 2-d");
    }
    THLongTensor_diag(result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPULongType::s_addmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    if(mat1.type().is_sparse()) {
        return static_cast<const Type*>(this)->addmm_out(result, self, SparseTensor(mat1), mat2, beta, alpha);
    }
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toLong();
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toLong();
    auto mat1_ = checked_cast_tensor<CPULongTensor>(mat1.pImpl,"mat1",4, false);
    auto mat2_ = checked_cast_tensor<CPULongTensor>(mat2.pImpl,"mat2",5, false);
    THLongTensor_addmm(result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CPULongType::s_addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    if(mat1.type().is_sparse()) {
        return static_cast<const Type*>(this)->addmm(self, SparseTensor(mat1), mat2, beta, alpha);
    }
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toLong();
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toLong();
    auto mat1_ = checked_cast_tensor<CPULongTensor>(mat1.pImpl,"mat1",4, false);
    auto mat2_ = checked_cast_tensor<CPULongTensor>(mat2.pImpl,"mat2",5, false);
    THLongTensor_addmm(result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CPULongType::addmm_out(Tensor & result, const Tensor & self, SparseTensor mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toLong();
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toLong();
    auto mat1_ = checked_cast_tensor<SparseCPULongTensor>(mat1.tref.pImpl,"mat1",4,false);
    auto mat2_ = checked_cast_tensor<CPULongTensor>(mat2.pImpl,"mat2",5, false);
    THSLongTensor_spaddmm(result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CPULongType::addmm(const Tensor & self, SparseTensor mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toLong();
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toLong();
    auto mat1_ = checked_cast_tensor<SparseCPULongTensor>(mat1.tref.pImpl,"mat1",4,false);
    auto mat2_ = checked_cast_tensor<CPULongTensor>(mat2.pImpl,"mat2",5, false);
    THSLongTensor_spaddmm(result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CPULongType::addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    if(mat1.type().is_sparse()) {
        return static_cast<const Type*>(this)->addmm_(self, SparseTensor(mat1), mat2, beta, alpha);
    }
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toLong();
    auto alpha_ = alpha.toLong();
    auto mat1_ = checked_cast_tensor<CPULongTensor>(mat1.pImpl,"mat1",5, false);
    auto mat2_ = checked_cast_tensor<CPULongTensor>(mat2.pImpl,"mat2",6, false);
    THLongTensor_addmm(self_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    return self;
}
Tensor & CPULongType::addmm_(Tensor & self, SparseTensor mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toLong();
    auto alpha_ = alpha.toLong();
    auto mat1_ = checked_cast_tensor<SparseCPULongTensor>(mat1.tref.pImpl,"mat1",5,false);
    auto mat2_ = checked_cast_tensor<CPULongTensor>(mat2.pImpl,"mat2",6, false);
    THSLongTensor_spaddmm(self_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    return self;
}
Tensor & CPULongType::s__addmv_out(Tensor & result, const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toLong();
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toLong();
    auto mat_ = checked_cast_tensor<CPULongTensor>(mat.pImpl,"mat",4, false);
    auto vec_ = checked_cast_tensor<CPULongTensor>(vec.pImpl,"vec",5, false);
    THLongTensor_addmv(result_->tensor, beta_, self_->tensor, alpha_, mat_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && mat_->isScalar() && vec_->isScalar());
    return result;
}
Tensor CPULongType::s__addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toLong();
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toLong();
    auto mat_ = checked_cast_tensor<CPULongTensor>(mat.pImpl,"mat",4, false);
    auto vec_ = checked_cast_tensor<CPULongTensor>(vec.pImpl,"vec",5, false);
    THLongTensor_addmv(result_->tensor, beta_, self_->tensor, alpha_, mat_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && mat_->isScalar() && vec_->isScalar());
    return result;
}
Tensor & CPULongType::_addmv_(Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toLong();
    auto alpha_ = alpha.toLong();
    auto mat_ = checked_cast_tensor<CPULongTensor>(mat.pImpl,"mat",5, false);
    auto vec_ = checked_cast_tensor<CPULongTensor>(vec.pImpl,"vec",6, false);
    THLongTensor_addmv(self_->tensor, beta_, self_->tensor, alpha_, mat_->tensor, vec_->tensor);
    return self;
}
Tensor & CPULongType::s__addr_out(Tensor & result, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toLong();
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toLong();
    auto vec1_ = checked_cast_tensor<CPULongTensor>(vec1.pImpl,"vec1",4, false);
    auto vec2_ = checked_cast_tensor<CPULongTensor>(vec2.pImpl,"vec2",5, false);
    THLongTensor_addr(result_->tensor, beta_, self_->tensor, alpha_, vec1_->tensor, vec2_->tensor);
    result_->maybeScalar(self_->isScalar() && vec1_->isScalar() && vec2_->isScalar());
    return result;
}
Tensor CPULongType::s__addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toLong();
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toLong();
    auto vec1_ = checked_cast_tensor<CPULongTensor>(vec1.pImpl,"vec1",4, false);
    auto vec2_ = checked_cast_tensor<CPULongTensor>(vec2.pImpl,"vec2",5, false);
    THLongTensor_addr(result_->tensor, beta_, self_->tensor, alpha_, vec1_->tensor, vec2_->tensor);
    result_->maybeScalar(self_->isScalar() && vec1_->isScalar() && vec2_->isScalar());
    return result;
}
Tensor & CPULongType::_addr_(Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toLong();
    auto alpha_ = alpha.toLong();
    auto vec1_ = checked_cast_tensor<CPULongTensor>(vec1.pImpl,"vec1",5, false);
    auto vec2_ = checked_cast_tensor<CPULongTensor>(vec2.pImpl,"vec2",6, false);
    THLongTensor_addr(self_->tensor, beta_, self_->tensor, alpha_, vec1_->tensor, vec2_->tensor);
    return self;
}
Tensor & CPULongType::_ger_out(Tensor & result, const Tensor & self, const Tensor & vec2) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.dim() == 0 ? 1 : self.size(0),vec2.dim() == 0 ? 1 : vec2.size(0) });
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto vec2_ = checked_cast_tensor<CPULongTensor>(vec2.pImpl,"vec2",2, false);
    THLongTensor_addr(result_->tensor, int64_t(0), result_->tensor, int64_t(1), self_->tensor, vec2_->tensor);
    result_->maybeScalar(false);
    return result;
}
Tensor CPULongType::_ger(const Tensor & self, const Tensor & vec2) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.dim() == 0 ? 1 : self.size(0),vec2.dim() == 0 ? 1 : vec2.size(0) });
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto vec2_ = checked_cast_tensor<CPULongTensor>(vec2.pImpl,"vec2",2, false);
    THLongTensor_addr(result_->tensor, int64_t(0), result_->tensor, int64_t(1), self_->tensor, vec2_->tensor);
    result_->maybeScalar(false);
    return result;
}
Tensor & CPULongType::_mv_out(Tensor & result, const Tensor & self, const Tensor & vec) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.size(0) });
    result.zero_();
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto vec_ = checked_cast_tensor<CPULongTensor>(vec.pImpl,"vec",2, false);
    THLongTensor_addmv(result_->tensor, int64_t(0), result_->tensor, int64_t(1), self_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && vec_->isScalar());
    return result;
}
Tensor CPULongType::_mv(const Tensor & self, const Tensor & vec) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.size(0) });
    result.zero_();
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto vec_ = checked_cast_tensor<CPULongTensor>(vec.pImpl,"vec",2, false);
    THLongTensor_addmv(result_->tensor, int64_t(0), result_->tensor, int64_t(1), self_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && vec_->isScalar());
    return result;
}
Tensor & CPULongType::_mm_out(Tensor & result, const Tensor & self, const Tensor & mat2) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.size(0),mat2.size(1) });
    result.zero_();
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CPULongTensor>(mat2.pImpl,"mat2",2, false);
    THLongTensor_addmm(result_->tensor, int64_t(0), result_->tensor, int64_t(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CPULongType::_mm(const Tensor & self, const Tensor & mat2) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.size(0),mat2.size(1) });
    result.zero_();
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CPULongTensor>(mat2.pImpl,"mat2",2, false);
    THLongTensor_addmm(result_->tensor, int64_t(0), result_->tensor, int64_t(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CPULongType::bmm_out(Tensor & result, const Tensor & self, const Tensor & mat2) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.size(0),self.size(1),mat2.size(2) });
    result.zero_();
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CPULongTensor>(mat2.pImpl,"mat2",2, false);
    THLongTensor_baddbmm(result_->tensor, int64_t(0), result_->tensor, int64_t(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CPULongType::bmm(const Tensor & self, const Tensor & mat2) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.size(0),self.size(1),mat2.size(2) });
    result.zero_();
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CPULongTensor>(mat2.pImpl,"mat2",2, false);
    THLongTensor_baddbmm(result_->tensor, int64_t(0), result_->tensor, int64_t(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CPULongType::s_addbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toLong();
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toLong();
    auto batch1_ = checked_cast_tensor<CPULongTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CPULongTensor>(batch2.pImpl,"batch2",5, false);
    THLongTensor_addbmm(result_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor CPULongType::s_addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toLong();
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toLong();
    auto batch1_ = checked_cast_tensor<CPULongTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CPULongTensor>(batch2.pImpl,"batch2",5, false);
    THLongTensor_addbmm(result_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor & CPULongType::addbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toLong();
    auto alpha_ = alpha.toLong();
    auto batch1_ = checked_cast_tensor<CPULongTensor>(batch1.pImpl,"batch1",5, false);
    auto batch2_ = checked_cast_tensor<CPULongTensor>(batch2.pImpl,"batch2",6, false);
    THLongTensor_addbmm(self_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    return self;
}
Tensor & CPULongType::s_baddbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toLong();
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toLong();
    auto batch1_ = checked_cast_tensor<CPULongTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CPULongTensor>(batch2.pImpl,"batch2",5, false);
    THLongTensor_baddbmm(result_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor CPULongType::s_baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toLong();
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toLong();
    auto batch1_ = checked_cast_tensor<CPULongTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CPULongTensor>(batch2.pImpl,"batch2",5, false);
    THLongTensor_baddbmm(result_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor & CPULongType::baddbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toLong();
    auto alpha_ = alpha.toLong();
    auto batch1_ = checked_cast_tensor<CPULongTensor>(batch1.pImpl,"batch1",5, false);
    auto batch2_ = checked_cast_tensor<CPULongTensor>(batch2.pImpl,"batch2",6, false);
    THLongTensor_baddbmm(self_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    return self;
}
Tensor & CPULongType::s_addcmul_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toLong();
    auto tensor1_ = checked_cast_tensor<CPULongTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CPULongTensor>(tensor2.pImpl,"tensor2",4, false);
    THLongTensor_addcmul(result_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor CPULongType::s_addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toLong();
    auto tensor1_ = checked_cast_tensor<CPULongTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CPULongTensor>(tensor2.pImpl,"tensor2",4, false);
    THLongTensor_addcmul(result_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor & CPULongType::s_addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toLong();
    auto tensor1_ = checked_cast_tensor<CPULongTensor>(tensor1.pImpl,"tensor1",4, false);
    auto tensor2_ = checked_cast_tensor<CPULongTensor>(tensor2.pImpl,"tensor2",5, false);
    THLongTensor_addcmul(self_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    return self;
}
Tensor & CPULongType::s_addcdiv_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toLong();
    auto tensor1_ = checked_cast_tensor<CPULongTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CPULongTensor>(tensor2.pImpl,"tensor2",4, false);
    THLongTensor_addcdiv(result_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor CPULongType::s_addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toLong();
    auto tensor1_ = checked_cast_tensor<CPULongTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CPULongTensor>(tensor2.pImpl,"tensor2",4, false);
    THLongTensor_addcdiv(result_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor & CPULongType::s_addcdiv_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toLong();
    auto tensor1_ = checked_cast_tensor<CPULongTensor>(tensor1.pImpl,"tensor1",4, false);
    auto tensor2_ = checked_cast_tensor<CPULongTensor>(tensor2.pImpl,"tensor2",5, false);
    THLongTensor_addcdiv(self_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    return self;
}
Tensor & CPULongType::random_(Tensor & self, int64_t from, int64_t to, Generator * generator) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    THLongTensor_clampedRandom(self_->tensor, generator_->generator, from, to);
    return self;
}
Tensor & CPULongType::random_(Tensor & self, int64_t to, Generator * generator) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    THLongTensor_cappedRandom(self_->tensor, generator_->generator, to);
    return self;
}
Tensor & CPULongType::random_(Tensor & self, Generator * generator) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    THLongTensor_random(self_->tensor, generator_->generator);
    return self;
}
Tensor & CPULongType::geometric_(Tensor & self, double p, Generator * generator) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    THLongTensor_geometric(self_->tensor, generator_->generator, p);
    return self;
}
Tensor CPULongType::tensor(Storage & storage, int64_t storageOffset, IntList size, IntList stride) const {
    auto storage_ = checked_cast_storage<CPULongStorage>(&storage,"storage",1);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    return Tensor((new CPULongTensor(context, THLongTensor_newWithStorage(storage_->storage, storageOffset, size_, stride_)))->maybeScalar(size.size() == 0),false);
}
Tensor CPULongType::tensor(IntList size) const {
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new CPULongTensor(context, THLongTensor_newWithSize(size_, NULL)))->maybeScalar(size.size() == 0),false);
}
Tensor CPULongType::tensor(IntList size, IntList stride) const {
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    return Tensor((new CPULongTensor(context, THLongTensor_newWithSize(size_, stride_)))->maybeScalar(size.size() == 0),false);
}
Tensor CPULongType::tensor() const {
    return Tensor((new CPULongTensor(context, THLongTensor_new())),false);
}
Tensor CPULongType::alias(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPULongTensor(context, THLongTensor_newWithTensor(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor & CPULongType::as_strided_out(Tensor & result, const Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    if (storage_offset == -1) {
      storage_offset = self_->tensor->storageOffset;
    }
    THLongTensor_setStorage(result_->tensor, self_->tensor->storage, storage_offset, size_, stride_);
    result_->maybeScalar(size.size() == 0);
    return result;
}
Tensor CPULongType::as_strided(const Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    if (storage_offset == -1) {
      storage_offset = self_->tensor->storageOffset;
    }
    THLongTensor_setStorage(result_->tensor, self_->tensor->storage, storage_offset, size_, stride_);
    result_->maybeScalar(size.size() == 0);
    return result;
}
Tensor & CPULongType::as_strided_(Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    if (storage_offset == -1) {
      storage_offset = self_->tensor->storageOffset;
    }
    THLongTensor_setStorage(self_->tensor, self_->tensor->storage, storage_offset, size_, stride_);
    self_->maybeScalar(size.size() == 0);
    return self;
}
Tensor & CPULongType::_cat_out(Tensor & self, TensorList tensors, int64_t dim) const {
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",0, false);
    auto tensors_ = tensor_list_checked_cast<CPULongTensor, Tensor, THLongTensor>(tensors,"tensors",1);
    THLongTensor_catArray(self_->tensor, tensors_.data(), tensors_.size(), dim);
    return self;
}
Tensor CPULongType::_cat(TensorList tensors, int64_t dim) const {
    auto self_ = new CPULongTensor(context);
    auto self = Tensor(self_, false);
    auto tensors_ = tensor_list_checked_cast<CPULongTensor, Tensor, THLongTensor>(tensors,"tensors",1);
    THLongTensor_catArray(self_->tensor, tensors_.data(), tensors_.size(), dim);
    return self;
}
Tensor CPULongType::_sparse_mask(const Tensor & self, SparseTensor mask) const {
    auto result_ = new SparseCPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPULongTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<SparseCPULongTensor>(mask.tref.pImpl,"mask",2,false);
    THLongTensor_sparseMask(result_->tensor, self_->tensor, mask_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPULongType::abs_out(Tensor & result, const Tensor & self) const {
    return  at::native::_abs_out_cpu(result, self);
}
Tensor & CPULongType::ceil_out(Tensor & result, const Tensor & self) const {
    return  at::native::_ceil_out_cpu(result, self);
}
Tensor & CPULongType::cos_out(Tensor & result, const Tensor & self) const {
    return  at::native::_cos_out_cpu(result, self);
}
Tensor CPULongType::embedding_dense_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) const {
    return  at::native::embedding_backward_cpu(grad, indices, num_weights, padding_idx, scale_grad_by_freq);
}
Tensor & CPULongType::embedding_renorm_(Tensor & self, const Tensor & indices, double max_norm, double norm_type) const {
    return  at::native::embedding_renorm_cpu_(self, indices, max_norm, norm_type);
}
std::tuple<Tensor,Tensor,Tensor> CPULongType::embedding_bag(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse) const {
    return  at::native::embedding_bag_cpu(weight, indices, offsets, scale_grad_by_freq, mode, sparse);
}
Tensor CPULongType::embedding_bag_dense_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode) const {
    return  at::native::embedding_bag_backward_cpu(grad, indices, offsets, offset2bag, bag_size, num_weights, scale_grad_by_freq, mode);
}
Tensor & CPULongType::exp_out(Tensor & result, const Tensor & self) const {
    return  at::native::_exp_out_cpu(result, self);
}
Tensor & CPULongType::eye_out(Tensor & result, int64_t n, int64_t m) const {
    return  at::native::eye_out_cpu(result, n, m);
}
Tensor & CPULongType::floor_out(Tensor & result, const Tensor & self) const {
    return  at::native::_floor_out_cpu(result, self);
}
Tensor CPULongType::_fft_with_size(const Tensor & self, int64_t signal_ndim, bool complex_input, bool complex_output, bool inverse, IntList checked_signal_sizes, bool normalized, bool onesided, IntList output_sizes) const {
    return  at::native::_fft_mkl(self, signal_ndim, complex_input, complex_output, inverse, checked_signal_sizes, normalized, onesided, output_sizes);
}
Tensor & CPULongType::log_out(Tensor & result, const Tensor & self) const {
    return  at::native::_log_out_cpu(result, self);
}
std::tuple<Tensor,Tensor> CPULongType::RoiPooling2d_forward(const Tensor & input, const Tensor & rois, int64_t pooledHeight, int64_t pooledWidth, double spatialScale) const {
    return  at::native::RoiPooling2d_forward_cpu(input, rois, pooledHeight, pooledWidth, spatialScale);
}
Tensor CPULongType::RoiPooling2d_backward(const Tensor & input, const Tensor & rois, int64_t pooledHeight, int64_t pooledWidth, double spatialScale, const Tensor & gradOutput, const Tensor & argmaxes) const {
    return  at::native::RoiPooling2d_backward_cpu(input, rois, pooledHeight, pooledWidth, spatialScale, gradOutput, argmaxes);
}
Tensor & CPULongType::round_out(Tensor & result, const Tensor & self) const {
    return  at::native::_round_out_cpu(result, self);
}
Tensor & CPULongType::sin_out(Tensor & result, const Tensor & self) const {
    return  at::native::_sin_out_cpu(result, self);
}
Tensor & CPULongType::sspaddmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return  at::native::_sspaddmm_out_only_sparse(result, self, mat1, mat2, beta, alpha);
}
Tensor CPULongType::sum(const Tensor & self) const {
    return  at::native::_sum_cpu(self);
}
Tensor & CPULongType::sum_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    return  at::native::_sum_out_cpu(result, self, dim, keepdim);
}
Tensor & CPULongType::sqrt_out(Tensor & result, const Tensor & self) const {
    return  at::native::_sqrt_out_cpu(result, self);
}
Tensor CPULongType::prod(const Tensor & self) const {
    return  at::native::_prod_cpu(self);
}
Tensor & CPULongType::prod_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    return  at::native::_prod_out_cpu(result, self, dim, keepdim);
}
Tensor & CPULongType::trunc_out(Tensor & result, const Tensor & self) const {
    return  at::native::_trunc_out_cpu(result, self);
}
std::tuple<Tensor,Tensor> CPULongType::_unique(const Tensor & self, bool sorted, bool return_inverse) const {
    return  at::native::_unique_cpu(self, sorted, return_inverse);
}
Tensor CPULongType::_s_where(const Tensor & condition, const Tensor & self, const Tensor & other) const {
    return  at::native::_s_where_cpu(condition, self, other);
}
Tensor CPULongType::_standard_gamma_grad(const Tensor & self, const Tensor & output) const {
    return  at::native::_standard_gamma_grad_cpu(self, output);
}
Tensor CPULongType::poisson(const Tensor & self, Generator * generator) const {
    return  at::native::_s_poisson_cpu(self, generator);
}

}
