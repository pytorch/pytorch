// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/CPUDoubleType.h"
#include "ATen/CPUDoubleStorage.h"
#include "ATen/CPUDoubleTensor.h"
#include "ATen/CPUGenerator.h"
#include "ATen/CPUByteTensor.h"
#include "ATen/CPUIntTensor.h"
#include "ATen/CPULongTensor.h"
#include "ATen/SparseCPUDoubleTensor.h"
#include "ATen/CPUDoubleTensor.h"
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

CPUDoubleType::CPUDoubleType(Context* context)
: Type(context, /*is_variable_or_undefined=*/false) {}
ScalarType CPUDoubleType::scalarType() const {
  return ScalarType::Double;
}
Backend CPUDoubleType::backend() const {
  return Backend::CPU;
}
bool CPUDoubleType::is_cuda() const { return backend() == kCUDA || backend() == kSparseCUDA; }
bool CPUDoubleType::is_sparse() const { return backend() == kSparseCPU || backend() == kSparseCUDA; }
bool CPUDoubleType::is_distributed() const { return false; }

std::unique_ptr<Storage> CPUDoubleType::storage() const {
  return std::unique_ptr<Storage>(new CPUDoubleStorage(context));
}
std::unique_ptr<Storage> CPUDoubleType::storage(size_t size) const {
  return std::unique_ptr<Storage>(new CPUDoubleStorage(context,size));
}
std::unique_ptr<Storage> CPUDoubleType::storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const {
    return std::unique_ptr<Storage>(
      new CPUDoubleStorage(context,data,size,deleter));
}
std::unique_ptr<Storage> CPUDoubleType::storageWithAllocator(int64_t size, std::unique_ptr<Allocator> allocator) const {
    return std::unique_ptr<Storage>(
        new CPUDoubleStorage(context, size, std::move(allocator)));
}
Tensor CPUDoubleType::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THDoubleTensor_retain( (THDoubleTensor*) th_pointer);
  return Tensor(new CPUDoubleTensor(context,(THDoubleTensor*)(th_pointer)), false);
}
std::unique_ptr<Storage> CPUDoubleType::unsafeStorageFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THDoubleStorage_retain( (THDoubleStorage*) th_pointer);
  return std::unique_ptr<Storage>(new CPUDoubleStorage(context, (THDoubleStorage*) th_pointer));
}
std::unique_ptr<Generator> CPUDoubleType::generator() const {
  return std::unique_ptr<Generator>(new CPUGenerator(context));
}

const char * CPUDoubleType::toString() const {
  return CPUDoubleType::typeString();
}
TypeID CPUDoubleType::ID() const {
  return TypeID::CPUDouble;
}

std::size_t CPUDoubleType::elementSizeInBytes() const {
  return sizeof(double);
}

const char * CPUDoubleType::typeString() {
  return "CPUDoubleType";
}

/* example
Tensor * CPUDoubleType::add(Tensor & a, Tensor & b) {
  std::cout << "add CPUDoubleTensor\n";
  return &a;
}
*/

int64_t CPUDoubleType::storage_offset(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    return static_cast<int64_t>(THDoubleTensor_storageOffset(self_->tensor));
}
Tensor & CPUDoubleType::resize_(Tensor & self, IntList size) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    THDoubleTensor_resize(self_->tensor, size_, NULL);
    self_->maybeScalar(size.size() == 0);
    return self;
}
int64_t CPUDoubleType::numel(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    return static_cast<int64_t>(THDoubleTensor_nElement(self_->tensor));
}
Tensor & CPUDoubleType::set_(Tensor & self, Storage & storage) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto storage_ = checked_cast_storage<CPUDoubleStorage>(&storage,"storage",2);
    THDoubleTensor_setStorage(self_->tensor, storage_->storage, 0, THLongStorageView::makeFromLength(static_cast<int64_t>(storage.size())), NULL);
    self_->maybeScalar(false);
    return self;
}
Tensor & CPUDoubleType::set_(Tensor & self, Storage & sourceStorage, int64_t storage_offset, IntList size, IntList stride) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto sourceStorage_ = checked_cast_storage<CPUDoubleStorage>(&sourceStorage,"sourceStorage",2);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    THDoubleTensor_setStorage(self_->tensor, sourceStorage_->storage, storage_offset, size_, stride_);
    self_->maybeScalar(size.size() == 0);
    return self;
}
Tensor & CPUDoubleType::set_(Tensor & self, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto source_ = checked_cast_tensor<CPUDoubleTensor>(source.pImpl,"source",2, false);
    THDoubleTensor_set(self_->tensor, source_->tensor);
    self_->maybeScalar(source_->isScalar());
    return self;
}
Tensor & CPUDoubleType::set_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_setStorage(self_->tensor, NULL, 0, NULL, NULL);
    self_->maybeScalar(false);
    return self;
}
Tensor & CPUDoubleType::fill_(Tensor & self, Scalar value) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toDouble();
    THDoubleTensor_fill(self_->tensor, value_);
    return self;
}
Tensor & CPUDoubleType::fill_(Tensor & self, const Tensor & value) const {
    if (value.dim() == 0) {
        return static_cast<const Type*>(this)->fill_(self, Scalar(value));
    }
    AT_ERROR("fill_ only supports a 0-dimensional value tensor, but got tensor "
        "with %" PRId64 " dimension(s)", value.dim());
}
bool CPUDoubleType::is_contiguous(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    return THDoubleTensor_isContiguous(self_->tensor);
}
bool CPUDoubleType::is_set_to(const Tensor & self, const Tensor & tensor) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto tensor_ = checked_cast_tensor<CPUDoubleTensor>(tensor.pImpl,"tensor",2, false);
    return THDoubleTensor_isSetTo(self_->tensor, tensor_->tensor);
}
Tensor & CPUDoubleType::s_masked_fill_(Tensor & self, const Tensor & mask, Scalar value) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CPUByteTensor>(mask.pImpl,"mask",2, false);
    auto value_ = value.toDouble();
    THDoubleTensor_maskedFill(self_->tensor, mask_->tensor, value_);
    return self;
}
Tensor & CPUDoubleType::s_masked_fill_(Tensor & self, const Tensor & mask, const Tensor & value) const {
    if (value.dim() == 0) {
        return static_cast<const Type*>(this)->masked_fill_(self, mask, Scalar(value));
    }
    AT_ERROR("masked_fill_ only supports a 0-dimensional value tensor, but got tensor "
        "with %" PRId64 " dimension(s)", value.dim());
}
Tensor & CPUDoubleType::s_masked_scatter_(Tensor & self, const Tensor & mask, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CPUByteTensor>(mask.pImpl,"mask",2, false);
    auto source_ = checked_cast_tensor<CPUDoubleTensor>(source.pImpl,"source",3, false);
    THDoubleTensor_maskedCopy(self_->tensor, mask_->tensor, source_->tensor);
    return self;
}
Tensor & CPUDoubleType::s_masked_select_out(Tensor & result, const Tensor & self, const Tensor & mask) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CPUByteTensor>(mask.pImpl,"mask",2, false);
    THDoubleTensor_maskedSelect(result_->tensor, self_->tensor, mask_->tensor);
    result_->maybeScalar(self_->isScalar() && mask_->isScalar());
    return result;
}
Tensor CPUDoubleType::s_masked_select(const Tensor & self, const Tensor & mask) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<CPUByteTensor>(mask.pImpl,"mask",2, false);
    THDoubleTensor_maskedSelect(result_->tensor, self_->tensor, mask_->tensor);
    result_->maybeScalar(self_->isScalar() && mask_->isScalar());
    return result;
}
Tensor CPUDoubleType::transpose(const Tensor & self, int64_t dim0, int64_t dim1) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim0 = maybe_wrap_dim(dim0, self_);
    dim1 = maybe_wrap_dim(dim1, self_);
    return Tensor((new CPUDoubleTensor(context, THDoubleTensor_newTranspose(self_->tensor, dim0, dim1)))->maybeScalar(self_->isScalar()),false);
}
Tensor CPUDoubleType::t(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPUDoubleTensor(context, THDoubleTensor_newTranspose(self_->tensor, 0, 1)))->maybeScalar(self_->isScalar()),false);
}
Tensor & CPUDoubleType::nonzero_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_nonzero(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::nonzero(const Tensor & self) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_nonzero(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::contiguous(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPUDoubleTensor(context, THDoubleTensor_newContiguous(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor CPUDoubleType::clone(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPUDoubleTensor(context, THDoubleTensor_newClone(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor CPUDoubleType::view(const Tensor & self, IntList size) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new CPUDoubleTensor(context, THDoubleTensor_newView(self_->tensor, size_)))->maybeScalar(size.size() == 0),false);
}
Tensor & CPUDoubleType::resize_as_(Tensor & self, const Tensor & the_template) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto the_template_ = checked_cast_tensor<CPUDoubleTensor>(the_template.pImpl,"the_template",2, false);
    THDoubleTensor_resizeAs(self_->tensor, the_template_->tensor);
    self_->maybeScalar(the_template_->isScalar());
    return self;
}
Tensor & CPUDoubleType::index_select_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    THDoubleTensor_indexSelect(result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
Tensor CPUDoubleType::index_select(const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    THDoubleTensor_indexSelect(result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
Tensor & CPUDoubleType::_indexCopy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    auto source_ = checked_cast_tensor<CPUDoubleTensor>(source.pImpl,"source",4, false);
    THDoubleTensor_indexCopy(self_->tensor, dim, index_->tensor, source_->tensor);
    return self;
}
Tensor & CPUDoubleType::take_out(Tensor & result, const Tensor & self, const Tensor & index) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",2, false);
    THDoubleTensor_take(result_->tensor, self_->tensor, index_->tensor);
    result_->maybeScalar(index_->isScalar());
    return result;
}
Tensor CPUDoubleType::take(const Tensor & self, const Tensor & index) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",2, false);
    THDoubleTensor_take(result_->tensor, self_->tensor, index_->tensor);
    result_->maybeScalar(index_->isScalar());
    return result;
}
Tensor & CPUDoubleType::put_(Tensor & self, const Tensor & index, const Tensor & source, bool accumulate) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",2, false);
    auto source_ = checked_cast_tensor<CPUDoubleTensor>(source.pImpl,"source",3, false);
    THDoubleTensor_put(self_->tensor, index_->tensor, source_->tensor, accumulate);
    return self;
}
Tensor & CPUDoubleType::index_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    auto source_ = checked_cast_tensor<CPUDoubleTensor>(source.pImpl,"source",4, false);
    THDoubleTensor_indexAdd(self_->tensor, dim, index_->tensor, source_->tensor);
    return self;
}
Tensor & CPUDoubleType::index_fill_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    auto value_ = value.toDouble();
    THDoubleTensor_indexFill(self_->tensor, dim, index_->tensor, value_);
    return self;
}
Tensor & CPUDoubleType::index_fill_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & value) const {
    if (value.dim() == 0) {
        return static_cast<const Type*>(this)->index_fill_(self, dim, index, Scalar(value));
    }
    AT_ERROR("index_fill_ only supports a 0-dimensional value tensor, but got tensor "
        "with %" PRId64 " dimension(s)", value.dim());
}
Tensor CPUDoubleType::unfold(const Tensor & self, int64_t dimension, int64_t size, int64_t step) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dimension = maybe_wrap_dim(dimension, self_);
    THDoubleTensor_unfold(result_->tensor, self_->tensor, dimension, size, step);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::_range_out(Tensor & result, Scalar start, Scalar end, Scalar step) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto start_ = start.toDouble();
    auto end_ = end.toDouble();
    auto step_ = step.toDouble();
    THDoubleTensor_range(result_->tensor, start_, end_, step_);
    return result;
}
Tensor CPUDoubleType::_range(Scalar start, Scalar end, Scalar step) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto start_ = start.toDouble();
    auto end_ = end.toDouble();
    auto step_ = step.toDouble();
    THDoubleTensor_range(result_->tensor, start_, end_, step_);
    return result;
}
Tensor & CPUDoubleType::_arange_out(Tensor & result, Scalar start, Scalar end, Scalar step) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto start_ = start.toDouble();
    auto end_ = end.toDouble();
    auto step_ = step.toDouble();
    THDoubleTensor_arange(result_->tensor, start_, end_, step_);
    return result;
}
Tensor CPUDoubleType::_arange(Scalar start, Scalar end, Scalar step) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto start_ = start.toDouble();
    auto end_ = end.toDouble();
    auto step_ = step.toDouble();
    THDoubleTensor_arange(result_->tensor, start_, end_, step_);
    return result;
}
Tensor & CPUDoubleType::_arange_out(Tensor & result, Scalar end) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto end_ = end.toDouble();
    THDoubleTensor_arange(result_->tensor, 0, end_, 1);
    return result;
}
Tensor CPUDoubleType::_arange(Scalar end) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto end_ = end.toDouble();
    THDoubleTensor_arange(result_->tensor, 0, end_, 1);
    return result;
}
Tensor & CPUDoubleType::scatter_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
    if (src.dim() == 0) {
        return static_cast<const Type*>(this)->scatter_(self, dim, index, Scalar(src));
    }
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    auto src_ = checked_cast_tensor<CPUDoubleTensor>(src.pImpl,"src",4, false);
    THDoubleTensor_scatter(self_->tensor, dim, index_->tensor, src_->tensor);
    return self;
}
Tensor & CPUDoubleType::scatter_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    auto value_ = value.toDouble();
    THDoubleTensor_scatterFill(self_->tensor, dim, index_->tensor, value_);
    return self;
}
Tensor & CPUDoubleType::scatter_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    auto src_ = checked_cast_tensor<CPUDoubleTensor>(src.pImpl,"src",4, false);
    THDoubleTensor_scatterAdd(self_->tensor, dim, index_->tensor, src_->tensor);
    return self;
}
Tensor & CPUDoubleType::gather_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    result.resize_(index.sizes());
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    THDoubleTensor_gather(result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
Tensor CPUDoubleType::gather(const Tensor & self, int64_t dim, const Tensor & index) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    result.resize_(index.sizes());
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto index_ = checked_cast_tensor<CPULongTensor>(index.pImpl,"index",3, false);
    THDoubleTensor_gather(result_->tensor, self_->tensor, dim, index_->tensor);
    result_->maybeScalar(self_->isScalar() && index_->isScalar());
    return result;
}
void* CPUDoubleType::data_ptr(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    return THDoubleTensor_data(self_->tensor);
}
bool CPUDoubleType::equal(const Tensor & self, const Tensor & other) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",2, false);
    return THDoubleTensor_equal(self_->tensor, other_->tensor);
}
Tensor & CPUDoubleType::__and___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_bitand(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::__and__(const Tensor & self, Scalar other) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_bitand(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::s___and___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__and___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",2, false);
    THDoubleTensor_cbitand(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUDoubleType::s___and__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__and__(self, Scalar(other));
    }
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",2, false);
    THDoubleTensor_cbitand(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUDoubleType::__iand__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_bitand(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUDoubleType::s___iand__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__iand__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",3, false);
    THDoubleTensor_cbitand(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUDoubleType::__or___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_bitor(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::__or__(const Tensor & self, Scalar other) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_bitor(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::s___or___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__or___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",2, false);
    THDoubleTensor_cbitor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUDoubleType::s___or__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__or__(self, Scalar(other));
    }
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",2, false);
    THDoubleTensor_cbitor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUDoubleType::__ior__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_bitor(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUDoubleType::s___ior__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__ior__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",3, false);
    THDoubleTensor_cbitor(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUDoubleType::__xor___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_bitxor(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::__xor__(const Tensor & self, Scalar other) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_bitxor(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::s___xor___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__xor___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",2, false);
    THDoubleTensor_cbitxor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUDoubleType::s___xor__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__xor__(self, Scalar(other));
    }
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",2, false);
    THDoubleTensor_cbitxor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUDoubleType::__ixor__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_bitxor(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUDoubleType::s___ixor__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__ixor__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",3, false);
    THDoubleTensor_cbitxor(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUDoubleType::__lshift___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_lshift(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::__lshift__(const Tensor & self, Scalar other) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_lshift(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::s___lshift___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__lshift___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",2, false);
    THDoubleTensor_clshift(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUDoubleType::s___lshift__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__lshift__(self, Scalar(other));
    }
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",2, false);
    THDoubleTensor_clshift(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUDoubleType::__ilshift__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_lshift(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUDoubleType::s___ilshift__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__ilshift__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",3, false);
    THDoubleTensor_clshift(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUDoubleType::__rshift___out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_rshift(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::__rshift__(const Tensor & self, Scalar other) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_rshift(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::s___rshift___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__rshift___out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",2, false);
    THDoubleTensor_crshift(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUDoubleType::s___rshift__(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__rshift__(self, Scalar(other));
    }
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",2, false);
    THDoubleTensor_crshift(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUDoubleType::__irshift__(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_rshift(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUDoubleType::s___irshift__(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->__irshift__(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",3, false);
    THDoubleTensor_crshift(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUDoubleType::lt_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_ltValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::lt(const Tensor & self, Scalar other) const {
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_ltValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::s_lt_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->lt_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",2, false);
    THDoubleTensor_ltTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUDoubleType::s_lt(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->lt(self, Scalar(other));
    }
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",2, false);
    THDoubleTensor_ltTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUDoubleType::lt_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_ltValueT(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUDoubleType::s_lt_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->lt_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",3, false);
    THDoubleTensor_ltTensorT(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUDoubleType::gt_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_gtValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::gt(const Tensor & self, Scalar other) const {
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_gtValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::s_gt_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->gt_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",2, false);
    THDoubleTensor_gtTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUDoubleType::s_gt(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->gt(self, Scalar(other));
    }
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",2, false);
    THDoubleTensor_gtTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUDoubleType::gt_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_gtValueT(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUDoubleType::s_gt_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->gt_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",3, false);
    THDoubleTensor_gtTensorT(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUDoubleType::le_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_leValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::le(const Tensor & self, Scalar other) const {
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_leValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::s_le_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->le_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",2, false);
    THDoubleTensor_leTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUDoubleType::s_le(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->le(self, Scalar(other));
    }
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",2, false);
    THDoubleTensor_leTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUDoubleType::le_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_leValueT(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUDoubleType::s_le_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->le_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",3, false);
    THDoubleTensor_leTensorT(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUDoubleType::ge_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_geValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::ge(const Tensor & self, Scalar other) const {
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_geValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::s_ge_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ge_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",2, false);
    THDoubleTensor_geTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUDoubleType::s_ge(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ge(self, Scalar(other));
    }
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",2, false);
    THDoubleTensor_geTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUDoubleType::ge_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_geValueT(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUDoubleType::s_ge_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ge_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",3, false);
    THDoubleTensor_geTensorT(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUDoubleType::eq_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_eqValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::eq(const Tensor & self, Scalar other) const {
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_eqValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::s_eq_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->eq_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",2, false);
    THDoubleTensor_eqTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUDoubleType::s_eq(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->eq(self, Scalar(other));
    }
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",2, false);
    THDoubleTensor_eqTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUDoubleType::eq_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_eqValueT(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUDoubleType::s_eq_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->eq_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",3, false);
    THDoubleTensor_eqTensorT(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUDoubleType::ne_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_neValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::ne(const Tensor & self, Scalar other) const {
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_neValue(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::s_ne_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ne_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUByteTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",2, false);
    THDoubleTensor_neTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUDoubleType::s_ne(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ne(self, Scalar(other));
    }
    auto result_ = new CPUByteTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",2, false);
    THDoubleTensor_neTensor(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUDoubleType::ne_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_neValueT(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUDoubleType::s_ne_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->ne_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",3, false);
    THDoubleTensor_neTensorT(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
std::tuple<Tensor &,Tensor &> CPUDoubleType::min_out(Tensor & min, Tensor & min_indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto min_ = checked_cast_tensor<CPUDoubleTensor>(min.pImpl,"min",0, false);
    auto min_indices_ = checked_cast_tensor<CPULongTensor>(min_indices.pImpl,"min_indices",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THDoubleTensor_min(min_->tensor, min_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    min_->maybeScalar(maybe_scalar);
    min_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(min, min_indices);
}
std::tuple<Tensor,Tensor> CPUDoubleType::min(const Tensor & self, int64_t dim, bool keepdim) const {
    auto min_ = new CPUDoubleTensor(context);
    auto min = Tensor(min_, false);
    auto min_indices_ = new CPULongTensor(context);
    auto min_indices = Tensor(min_indices_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THDoubleTensor_min(min_->tensor, min_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    min_->maybeScalar(maybe_scalar);
    min_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(min, min_indices);
}
Tensor & CPUDoubleType::s_min_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",2, false);
    THDoubleTensor_cmin(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUDoubleType::s_min(const Tensor & self, const Tensor & other) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",2, false);
    THDoubleTensor_cmin(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUDoubleType::min(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<double>(THDoubleTensor_minall(self_->tensor)));
}
std::tuple<Tensor &,Tensor &> CPUDoubleType::max_out(Tensor & max, Tensor & max_indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto max_ = checked_cast_tensor<CPUDoubleTensor>(max.pImpl,"max",0, false);
    auto max_indices_ = checked_cast_tensor<CPULongTensor>(max_indices.pImpl,"max_indices",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THDoubleTensor_max(max_->tensor, max_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    max_->maybeScalar(maybe_scalar);
    max_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(max, max_indices);
}
std::tuple<Tensor,Tensor> CPUDoubleType::max(const Tensor & self, int64_t dim, bool keepdim) const {
    auto max_ = new CPUDoubleTensor(context);
    auto max = Tensor(max_, false);
    auto max_indices_ = new CPULongTensor(context);
    auto max_indices = Tensor(max_indices_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THDoubleTensor_max(max_->tensor, max_indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    max_->maybeScalar(maybe_scalar);
    max_indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(max, max_indices);
}
Tensor & CPUDoubleType::s_max_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",2, false);
    THDoubleTensor_cmax(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUDoubleType::s_max(const Tensor & self, const Tensor & other) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",2, false);
    THDoubleTensor_cmax(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUDoubleType::max(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<double>(THDoubleTensor_maxall(self_->tensor)));
}
std::tuple<Tensor &,Tensor &> CPUDoubleType::kthvalue_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool keepdim) const {
    auto values_ = checked_cast_tensor<CPUDoubleTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THDoubleTensor_kthvalue(values_->tensor, indices_->tensor, self_->tensor, k, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CPUDoubleType::kthvalue(const Tensor & self, int64_t k, int64_t dim, bool keepdim) const {
    auto values_ = new CPUDoubleTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CPULongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THDoubleTensor_kthvalue(values_->tensor, indices_->tensor, self_->tensor, k, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
std::tuple<Tensor &,Tensor &> CPUDoubleType::mode_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = checked_cast_tensor<CPUDoubleTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THDoubleTensor_mode(values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CPUDoubleType::mode(const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = new CPUDoubleTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CPULongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THDoubleTensor_mode(values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
std::tuple<Tensor &,Tensor &> CPUDoubleType::median_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = checked_cast_tensor<CPUDoubleTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THDoubleTensor_median(values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CPUDoubleType::median(const Tensor & self, int64_t dim, bool keepdim) const {
    auto values_ = new CPUDoubleTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CPULongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THDoubleTensor_median(values_->tensor, indices_->tensor, self_->tensor, dim, keepdim);
    bool maybe_scalar = self_->isScalar() || (keepdim == false && self_->dim() == 1);
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
Tensor CPUDoubleType::median(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<double>(THDoubleTensor_medianall(self_->tensor)));
}
std::tuple<Tensor &,Tensor &> CPUDoubleType::sort_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool descending) const {
    auto values_ = checked_cast_tensor<CPUDoubleTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THDoubleTensor_sort(values_->tensor, indices_->tensor, self_->tensor, dim, descending);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CPUDoubleType::sort(const Tensor & self, int64_t dim, bool descending) const {
    auto values_ = new CPUDoubleTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CPULongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THDoubleTensor_sort(values_->tensor, indices_->tensor, self_->tensor, dim, descending);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
std::tuple<Tensor &,Tensor &> CPUDoubleType::topk_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) const {
    auto values_ = checked_cast_tensor<CPUDoubleTensor>(values.pImpl,"values",0, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THDoubleTensor_topk(values_->tensor, indices_->tensor, self_->tensor, k, dim, largest, sorted);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> CPUDoubleType::topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) const {
    auto values_ = new CPUDoubleTensor(context);
    auto values = Tensor(values_, false);
    auto indices_ = new CPULongTensor(context);
    auto indices = Tensor(indices_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THDoubleTensor_topk(values_->tensor, indices_->tensor, self_->tensor, k, dim, largest, sorted);
    bool maybe_scalar = self_->isScalar();
    values_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(values, indices);
}
Tensor & CPUDoubleType::_abs_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_abs(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::_abs(const Tensor & self) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_abs(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::sigmoid_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_sigmoid(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUDoubleType::sigmoid_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_sigmoid(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::sigmoid(const Tensor & self) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_sigmoid(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::_log_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_log(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::_log(const Tensor & self) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_log(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::log10_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_log10(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUDoubleType::log10_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_log10(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::log10(const Tensor & self) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_log10(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::log1p_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_log1p(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUDoubleType::log1p_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_log1p(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::log1p(const Tensor & self) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_log1p(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::log2_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_log2(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUDoubleType::log2_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_log2(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::log2(const Tensor & self) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_log2(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::lgamma_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_lgamma(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::lgamma(const Tensor & self) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_lgamma(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::lgamma_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_lgamma(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUDoubleType::digamma_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_digamma(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::digamma(const Tensor & self) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_digamma(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::digamma_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_digamma(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUDoubleType::polygamma_out(Tensor & result, int64_t n, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    THDoubleTensor_polygamma(result_->tensor, n, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::polygamma(int64_t n, const Tensor & self) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    THDoubleTensor_polygamma(result_->tensor, n, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::polygamma_(Tensor & self, int64_t n) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_polygamma(self_->tensor, n, self_->tensor);
    return self;
}
Tensor & CPUDoubleType::_exp_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_exp(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::_exp(const Tensor & self) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_exp(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::expm1_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_expm1(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUDoubleType::expm1_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_expm1(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::expm1(const Tensor & self) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_expm1(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::_cos_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_cos(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::_cos(const Tensor & self) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_cos(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::acos_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_acos(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUDoubleType::acos_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_acos(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::acos(const Tensor & self) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_acos(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::cosh_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_cosh(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUDoubleType::cosh_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_cosh(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::cosh(const Tensor & self) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_cosh(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::_sin_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_sin(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::_sin(const Tensor & self) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_sin(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::asin_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_asin(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUDoubleType::asin_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_asin(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::asin(const Tensor & self) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_asin(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::sinh_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_sinh(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUDoubleType::sinh_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_sinh(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::sinh(const Tensor & self) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_sinh(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::tan_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_tan(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUDoubleType::tan_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_tan(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::tan(const Tensor & self) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_tan(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::atan_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_atan(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUDoubleType::atan_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_atan(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::atan(const Tensor & self) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_atan(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::tanh_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_tanh(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUDoubleType::tanh_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_tanh(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::tanh(const Tensor & self) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_tanh(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::erf_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_erf(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUDoubleType::erf_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_erf(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::erf(const Tensor & self) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_erf(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::erfinv_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_erfinv(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUDoubleType::erfinv_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_erfinv(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::erfinv(const Tensor & self) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_erfinv(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::_sqrt_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_sqrt(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::_sqrt(const Tensor & self) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_sqrt(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::rsqrt_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_rsqrt(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUDoubleType::rsqrt_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_rsqrt(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::rsqrt(const Tensor & self) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_rsqrt(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::_ceil_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_ceil(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::_ceil(const Tensor & self) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_ceil(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::_floor_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_floor(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::_floor(const Tensor & self) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_floor(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::_round_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_round(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::_round(const Tensor & self) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_round(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::_trunc_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_trunc(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::_trunc(const Tensor & self) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_trunc(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::frac_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_frac(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUDoubleType::frac_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_frac(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::frac(const Tensor & self) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_frac(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::mean_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THDoubleTensor_mean(result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CPUDoubleType::mean(const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THDoubleTensor_mean(result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CPUDoubleType::mean(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<double>(THDoubleTensor_meanall(self_->tensor)));
}
Tensor & CPUDoubleType::var_out(Tensor & result, const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THDoubleTensor_var(result_->tensor, self_->tensor, dim, (unbiased) ? 0 : 1, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CPUDoubleType::var(const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THDoubleTensor_var(result_->tensor, self_->tensor, dim, (unbiased) ? 0 : 1, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CPUDoubleType::var(const Tensor & self, bool unbiased) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<double>(THDoubleTensor_varall(self_->tensor, (unbiased) ? 0 : 1)));
}
Tensor & CPUDoubleType::std_out(Tensor & result, const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THDoubleTensor_std(result_->tensor, self_->tensor, dim, (unbiased) ? 0 : 1, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CPUDoubleType::std(const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THDoubleTensor_std(result_->tensor, self_->tensor, dim, (unbiased) ? 0 : 1, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CPUDoubleType::std(const Tensor & self, bool unbiased) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<double>(THDoubleTensor_stdall(self_->tensor, (unbiased) ? 0 : 1)));
}
Tensor & CPUDoubleType::norm_out(Tensor & result, const Tensor & self, Scalar p, int64_t dim, bool keepdim) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto p_ = p.toDouble();
    dim = maybe_wrap_dim(dim, self_);
    THDoubleTensor_norm(result_->tensor, self_->tensor, p_, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CPUDoubleType::norm(const Tensor & self, Scalar p, int64_t dim, bool keepdim) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto p_ = p.toDouble();
    dim = maybe_wrap_dim(dim, self_);
    THDoubleTensor_norm(result_->tensor, self_->tensor, p_, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CPUDoubleType::norm(const Tensor & self, Scalar p) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto p_ = p.toDouble();
    // norm(value) for a sparse tensor returns a DENSE 0-dim tensor
    if (self.is_sparse()) {
      auto result = THDoubleTensor_normall( self_->tensor, convert<double>(p_));
      return toBackend(toDense(backend())).tensor({}).fill_(result);
    }
    // aten_custom_call is followed by the generated call to normall
    return scalarTensor(convert<double>(THDoubleTensor_normall(self_->tensor, p_)));
}
Tensor & CPUDoubleType::renorm_out(Tensor & result, const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto p_ = p.toDouble();
    dim = maybe_wrap_dim(dim, self_);
    auto maxnorm_ = maxnorm.toDouble();
    THDoubleTensor_renorm(result_->tensor, self_->tensor, p_, dim, maxnorm_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::renorm(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto p_ = p.toDouble();
    dim = maybe_wrap_dim(dim, self_);
    auto maxnorm_ = maxnorm.toDouble();
    THDoubleTensor_renorm(result_->tensor, self_->tensor, p_, dim, maxnorm_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::renorm_(Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto p_ = p.toDouble();
    dim = maybe_wrap_dim(dim, self_);
    auto maxnorm_ = maxnorm.toDouble();
    THDoubleTensor_renorm(self_->tensor, self_->tensor, p_, dim, maxnorm_);
    return self;
}
Tensor CPUDoubleType::s_dist(const Tensor & self, const Tensor & other, Scalar p) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",2, false);
    auto p_ = p.toDouble();
    return scalarTensor(convert<double>(THDoubleTensor_dist(self_->tensor, other_->tensor, p_)));
}
Tensor & CPUDoubleType::reciprocal_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_cinv(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::reciprocal(const Tensor & self) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_cinv(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::reciprocal_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_cinv(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUDoubleType::neg_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_neg(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::neg(const Tensor & self) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_neg(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::neg_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_neg(self_->tensor, self_->tensor);
    return self;
}
Tensor & CPUDoubleType::s_atan2_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",2, false);
    THDoubleTensor_atan2(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUDoubleType::s_atan2(const Tensor & self, const Tensor & other) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",2, false);
    THDoubleTensor_atan2(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUDoubleType::s_atan2_(Tensor & self, const Tensor & other) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",3, false);
    THDoubleTensor_atan2(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUDoubleType::pow_out(Tensor & result, const Tensor & self, Scalar exponent) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toDouble();
    THDoubleTensor_pow(result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::pow(const Tensor & self, Scalar exponent) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toDouble();
    THDoubleTensor_pow(result_->tensor, self_->tensor, exponent_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::s_pow_out(Tensor & result, const Tensor & self, const Tensor & exponent) const {
    if (exponent.dim() == 0) {
        return static_cast<const Type*>(this)->pow_out(result, self, Scalar(exponent));
    }
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto exponent_ = checked_cast_tensor<CPUDoubleTensor>(exponent.pImpl,"exponent",2, false);
    THDoubleTensor_cpow(result_->tensor, self_->tensor, exponent_->tensor);
    result_->maybeScalar(self_->isScalar() && exponent_->isScalar());
    return result;
}
Tensor CPUDoubleType::s_pow(const Tensor & self, const Tensor & exponent) const {
    if (exponent.dim() == 0) {
        return static_cast<const Type*>(this)->pow(self, Scalar(exponent));
    }
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto exponent_ = checked_cast_tensor<CPUDoubleTensor>(exponent.pImpl,"exponent",2, false);
    THDoubleTensor_cpow(result_->tensor, self_->tensor, exponent_->tensor);
    result_->maybeScalar(self_->isScalar() && exponent_->isScalar());
    return result;
}
Tensor & CPUDoubleType::pow_out(Tensor & result, Scalar base, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto base_ = base.toDouble();
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    THDoubleTensor_tpow(result_->tensor, base_, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::pow(Scalar base, const Tensor & self) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto base_ = base.toDouble();
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    THDoubleTensor_tpow(result_->tensor, base_, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::pow_(Tensor & self, Scalar exponent) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto exponent_ = exponent.toDouble();
    THDoubleTensor_pow(self_->tensor, self_->tensor, exponent_);
    return self;
}
Tensor & CPUDoubleType::s_pow_(Tensor & self, const Tensor & exponent) const {
    if (exponent.dim() == 0) {
        return static_cast<const Type*>(this)->pow_(self, Scalar(exponent));
    }
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto exponent_ = checked_cast_tensor<CPUDoubleTensor>(exponent.pImpl,"exponent",3, false);
    THDoubleTensor_cpow(self_->tensor, self_->tensor, exponent_->tensor);
    return self;
}
Tensor & CPUDoubleType::s_lerp_out(Tensor & result, const Tensor & self, const Tensor & end, Scalar weight) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto end_ = checked_cast_tensor<CPUDoubleTensor>(end.pImpl,"end",2, false);
    auto weight_ = weight.toDouble();
    THDoubleTensor_lerp(result_->tensor, self_->tensor, end_->tensor, weight_);
    result_->maybeScalar(self_->isScalar() && end_->isScalar());
    return result;
}
Tensor CPUDoubleType::s_lerp(const Tensor & self, const Tensor & end, Scalar weight) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto end_ = checked_cast_tensor<CPUDoubleTensor>(end.pImpl,"end",2, false);
    auto weight_ = weight.toDouble();
    THDoubleTensor_lerp(result_->tensor, self_->tensor, end_->tensor, weight_);
    result_->maybeScalar(self_->isScalar() && end_->isScalar());
    return result;
}
Tensor & CPUDoubleType::s_lerp_(Tensor & self, const Tensor & end, Scalar weight) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto end_ = checked_cast_tensor<CPUDoubleTensor>(end.pImpl,"end",3, false);
    auto weight_ = weight.toDouble();
    THDoubleTensor_lerp(self_->tensor, self_->tensor, end_->tensor, weight_);
    return self;
}
Tensor & CPUDoubleType::_linspace_out(Tensor & result, Scalar start, Scalar end, int64_t steps) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto start_ = start.toDouble();
    auto end_ = end.toDouble();
    THDoubleTensor_linspace(result_->tensor, start_, end_, steps);
    return result;
}
Tensor CPUDoubleType::_linspace(Scalar start, Scalar end, int64_t steps) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto start_ = start.toDouble();
    auto end_ = end.toDouble();
    THDoubleTensor_linspace(result_->tensor, start_, end_, steps);
    return result;
}
Tensor & CPUDoubleType::_logspace_out(Tensor & result, Scalar start, Scalar end, int64_t steps) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto start_ = start.toDouble();
    auto end_ = end.toDouble();
    THDoubleTensor_logspace(result_->tensor, start_, end_, steps);
    return result;
}
Tensor CPUDoubleType::_logspace(Scalar start, Scalar end, int64_t steps) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto start_ = start.toDouble();
    auto end_ = end.toDouble();
    THDoubleTensor_logspace(result_->tensor, start_, end_, steps);
    return result;
}
Tensor & CPUDoubleType::histc_out(Tensor & result, const Tensor & self, int64_t bins, Scalar min, Scalar max) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toDouble();
    auto max_ = max.toDouble();
    THDoubleTensor_histc(result_->tensor, self_->tensor, bins, min_, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::histc(const Tensor & self, int64_t bins, Scalar min, Scalar max) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toDouble();
    auto max_ = max.toDouble();
    THDoubleTensor_histc(result_->tensor, self_->tensor, bins, min_, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::zero_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_zero(self_->tensor);
    return self;
}
Tensor CPUDoubleType::_sumall(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<double>(THDoubleTensor_sumall(self_->tensor)));
}
Tensor & CPUDoubleType::_sum_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THDoubleTensor_sum(result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CPUDoubleType::_sum(const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THDoubleTensor_sum(result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CPUDoubleType::_prodall(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<double>(THDoubleTensor_prodall(self_->tensor)));
}
Tensor & CPUDoubleType::_prod_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THDoubleTensor_prod(result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor CPUDoubleType::_prod(const Tensor & self, int64_t dim, bool keepdim) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THDoubleTensor_prod(result_->tensor, self_->tensor, dim, keepdim);
    result_->maybeScalar(self_->isScalar() || (keepdim == false && self_->dim() == 1));
    return result;
}
Tensor & CPUDoubleType::cumsum_out(Tensor & result, const Tensor & self, int64_t dim) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THDoubleTensor_cumsum(result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::cumsum(const Tensor & self, int64_t dim) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THDoubleTensor_cumsum(result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::cumprod_out(Tensor & result, const Tensor & self, int64_t dim) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THDoubleTensor_cumprod(result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::cumprod(const Tensor & self, int64_t dim) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    THDoubleTensor_cumprod(result_->tensor, self_->tensor, dim);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::sign_out(Tensor & result, const Tensor & self) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_sign(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::sign(const Tensor & self) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_sign(result_->tensor, self_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::sign_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_sign(self_->tensor, self_->tensor);
    return self;
}
Tensor CPUDoubleType::trace(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    return scalarTensor(convert<double>(THDoubleTensor_trace(self_->tensor)));
}
Tensor & CPUDoubleType::add_out(Tensor & result, const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    auto alpha_ = alpha.toDouble();
    THDoubleTensor_add_scaled(result_->tensor, self_->tensor, other_, alpha_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::add(const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    auto alpha_ = alpha.toDouble();
    THDoubleTensor_add_scaled(result_->tensor, self_->tensor, other_, alpha_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::s_add_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if(other.type().is_sparse()) {
        return static_cast<const Type*>(this)->add_out(result, self, SparseTensor(other), alpha);
    }
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toDouble();
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",3, false);
    THDoubleTensor_cadd(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUDoubleType::s_add(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if(other.type().is_sparse()) {
        return static_cast<const Type*>(this)->add(self, SparseTensor(other), alpha);
    }
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add(self, Scalar(other), alpha);
    }
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toDouble();
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",3, false);
    THDoubleTensor_cadd(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUDoubleType::add_out(Tensor & result, const Tensor & self, SparseTensor other, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toDouble();
    auto other_ = checked_cast_tensor<SparseCPUDoubleTensor>(other.tref.pImpl,"other",3,false);
    THSDoubleTensor_spcadd(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::add(const Tensor & self, SparseTensor other, Scalar alpha) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toDouble();
    auto other_ = checked_cast_tensor<SparseCPUDoubleTensor>(other.tref.pImpl,"other",3,false);
    THSDoubleTensor_spcadd(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::add_(Tensor & self, Scalar other, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    auto alpha_ = alpha.toDouble();
    THDoubleTensor_add_scaled(self_->tensor, self_->tensor, other_, alpha_);
    return self;
}
Tensor & CPUDoubleType::s_add_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if(other.type().is_sparse()) {
        return static_cast<const Type*>(this)->add_(self, SparseTensor(other), alpha);
    }
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->add_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toDouble();
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",4, false);
    THDoubleTensor_cadd(self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & CPUDoubleType::add_(Tensor & self, SparseTensor other, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toDouble();
    auto other_ = checked_cast_tensor<SparseCPUDoubleTensor>(other.tref.pImpl,"other",4,false);
    THSDoubleTensor_spcadd(self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & CPUDoubleType::sub_out(Tensor & result, const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    auto alpha_ = alpha.toDouble();
    THDoubleTensor_sub_scaled(result_->tensor, self_->tensor, other_, alpha_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::sub(const Tensor & self, Scalar other, Scalar alpha) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    auto alpha_ = alpha.toDouble();
    THDoubleTensor_sub_scaled(result_->tensor, self_->tensor, other_, alpha_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::s_sub_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_out(result, self, Scalar(other), alpha);
    }
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toDouble();
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",3, false);
    THDoubleTensor_csub(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUDoubleType::s_sub(const Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub(self, Scalar(other), alpha);
    }
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toDouble();
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",3, false);
    THDoubleTensor_csub(result_->tensor, self_->tensor, alpha_, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUDoubleType::sub_(Tensor & self, Scalar other, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    auto alpha_ = alpha.toDouble();
    THDoubleTensor_sub_scaled(self_->tensor, self_->tensor, other_, alpha_);
    return self;
}
Tensor & CPUDoubleType::s_sub_(Tensor & self, const Tensor & other, Scalar alpha) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->sub_(self, Scalar(other), alpha);
    }
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toDouble();
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",4, false);
    THDoubleTensor_csub(self_->tensor, self_->tensor, alpha_, other_->tensor);
    return self;
}
Tensor & CPUDoubleType::mul_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_mul(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::mul(const Tensor & self, Scalar other) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_mul(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::s_mul_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",2, false);
    THDoubleTensor_cmul(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUDoubleType::s_mul(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul(self, Scalar(other));
    }
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",2, false);
    THDoubleTensor_cmul(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUDoubleType::mul_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_mul(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUDoubleType::s_mul_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->mul_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",3, false);
    THDoubleTensor_cmul(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUDoubleType::div_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_div(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::div(const Tensor & self, Scalar other) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_div(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::s_div_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->div_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",2, false);
    THDoubleTensor_cdiv(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUDoubleType::s_div(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->div(self, Scalar(other));
    }
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",2, false);
    THDoubleTensor_cdiv(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUDoubleType::div_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_div(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUDoubleType::s_div_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->div_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",3, false);
    THDoubleTensor_cdiv(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUDoubleType::fmod_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_fmod(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::fmod(const Tensor & self, Scalar other) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_fmod(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::s_fmod_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->fmod_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",2, false);
    THDoubleTensor_cfmod(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUDoubleType::s_fmod(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->fmod(self, Scalar(other));
    }
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",2, false);
    THDoubleTensor_cfmod(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUDoubleType::fmod_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_fmod(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUDoubleType::s_fmod_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->fmod_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",3, false);
    THDoubleTensor_cfmod(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUDoubleType::remainder_out(Tensor & result, const Tensor & self, Scalar other) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_remainder(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::remainder(const Tensor & self, Scalar other) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_remainder(result_->tensor, self_->tensor, other_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::s_remainder_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->remainder_out(result, self, Scalar(other));
    }
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",2, false);
    THDoubleTensor_cremainder(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUDoubleType::s_remainder(const Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->remainder(self, Scalar(other));
    }
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",2, false);
    THDoubleTensor_cremainder(result_->tensor, self_->tensor, other_->tensor);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUDoubleType::remainder_(Tensor & self, Scalar other) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = other.toDouble();
    THDoubleTensor_remainder(self_->tensor, self_->tensor, other_);
    return self;
}
Tensor & CPUDoubleType::s_remainder_(Tensor & self, const Tensor & other) const {
    if (other.dim() == 0) {
        return static_cast<const Type*>(this)->remainder_(self, Scalar(other));
    }
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",3, false);
    THDoubleTensor_cremainder(self_->tensor, self_->tensor, other_->tensor);
    return self;
}
Tensor & CPUDoubleType::clamp_out(Tensor & result, const Tensor & self, Scalar min, Scalar max) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toDouble();
    auto max_ = max.toDouble();
    THDoubleTensor_clamp(result_->tensor, self_->tensor, min_, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::clamp(const Tensor & self, Scalar min, Scalar max) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toDouble();
    auto max_ = max.toDouble();
    THDoubleTensor_clamp(result_->tensor, self_->tensor, min_, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::clamp_(Tensor & self, Scalar min, Scalar max) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toDouble();
    auto max_ = max.toDouble();
    THDoubleTensor_clamp(self_->tensor, self_->tensor, min_, max_);
    return self;
}
Tensor & CPUDoubleType::clamp_min_out(Tensor & result, const Tensor & self, Scalar min) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toDouble();
    THDoubleTensor_cmaxValue(result_->tensor, self_->tensor, min_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::clamp_min(const Tensor & self, Scalar min) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toDouble();
    THDoubleTensor_cmaxValue(result_->tensor, self_->tensor, min_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::clamp_min_(Tensor & self, Scalar min) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto min_ = min.toDouble();
    THDoubleTensor_cmaxValue(self_->tensor, self_->tensor, min_);
    return self;
}
Tensor & CPUDoubleType::clamp_max_out(Tensor & result, const Tensor & self, Scalar max) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto max_ = max.toDouble();
    THDoubleTensor_cminValue(result_->tensor, self_->tensor, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::clamp_max(const Tensor & self, Scalar max) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto max_ = max.toDouble();
    THDoubleTensor_cminValue(result_->tensor, self_->tensor, max_);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::clamp_max_(Tensor & self, Scalar max) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto max_ = max.toDouble();
    THDoubleTensor_cminValue(self_->tensor, self_->tensor, max_);
    return self;
}
Tensor CPUDoubleType::_dot(const Tensor & self, const Tensor & tensor) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto tensor_ = checked_cast_tensor<CPUDoubleTensor>(tensor.pImpl,"tensor",2, false);
    return scalarTensor(convert<double>(THDoubleTensor_dot(self_->tensor, tensor_->tensor)));
}
Tensor & CPUDoubleType::tril_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_tril(result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::tril(const Tensor & self, int64_t diagonal) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_tril(result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::tril_(Tensor & self, int64_t diagonal) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_tril(self_->tensor, self_->tensor, diagonal);
    return self;
}
Tensor & CPUDoubleType::triu_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_triu(result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::triu(const Tensor & self, int64_t diagonal) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_triu(result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::triu_(Tensor & self, int64_t diagonal) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_triu(self_->tensor, self_->tensor, diagonal);
    return self;
}
Tensor & CPUDoubleType::cross_out(Tensor & result, const Tensor & self, const Tensor & other, int64_t dim) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",2, false);
    THDoubleTensor_cross(result_->tensor, self_->tensor, other_->tensor, dim);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor CPUDoubleType::cross(const Tensor & self, const Tensor & other, int64_t dim) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto other_ = checked_cast_tensor<CPUDoubleTensor>(other.pImpl,"other",2, false);
    THDoubleTensor_cross(result_->tensor, self_->tensor, other_->tensor, dim);
    result_->maybeScalar(self_->isScalar() && other_->isScalar());
    return result;
}
Tensor & CPUDoubleType::diag_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    if (self_->isScalar()) {
      throw std::runtime_error("Input must be 1-d or 2-d");
    }
    THDoubleTensor_diag(result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::diag(const Tensor & self, int64_t diagonal) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    if (self_->isScalar()) {
      throw std::runtime_error("Input must be 1-d or 2-d");
    }
    THDoubleTensor_diag(result_->tensor, self_->tensor, diagonal);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::s_addmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    if(mat1.type().is_sparse()) {
        return static_cast<const Type*>(this)->addmm_out(result, self, SparseTensor(mat1), mat2, beta, alpha);
    }
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toDouble();
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toDouble();
    auto mat1_ = checked_cast_tensor<CPUDoubleTensor>(mat1.pImpl,"mat1",4, false);
    auto mat2_ = checked_cast_tensor<CPUDoubleTensor>(mat2.pImpl,"mat2",5, false);
    THDoubleTensor_addmm(result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CPUDoubleType::s_addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    if(mat1.type().is_sparse()) {
        return static_cast<const Type*>(this)->addmm(self, SparseTensor(mat1), mat2, beta, alpha);
    }
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toDouble();
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toDouble();
    auto mat1_ = checked_cast_tensor<CPUDoubleTensor>(mat1.pImpl,"mat1",4, false);
    auto mat2_ = checked_cast_tensor<CPUDoubleTensor>(mat2.pImpl,"mat2",5, false);
    THDoubleTensor_addmm(result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat1_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CPUDoubleType::addmm_out(Tensor & result, const Tensor & self, SparseTensor mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toDouble();
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toDouble();
    auto mat1_ = checked_cast_tensor<SparseCPUDoubleTensor>(mat1.tref.pImpl,"mat1",4,false);
    auto mat2_ = checked_cast_tensor<CPUDoubleTensor>(mat2.pImpl,"mat2",5, false);
    THSDoubleTensor_spaddmm(result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CPUDoubleType::addmm(const Tensor & self, SparseTensor mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toDouble();
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toDouble();
    auto mat1_ = checked_cast_tensor<SparseCPUDoubleTensor>(mat1.tref.pImpl,"mat1",4,false);
    auto mat2_ = checked_cast_tensor<CPUDoubleTensor>(mat2.pImpl,"mat2",5, false);
    THSDoubleTensor_spaddmm(result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CPUDoubleType::addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    if(mat1.type().is_sparse()) {
        return static_cast<const Type*>(this)->addmm_(self, SparseTensor(mat1), mat2, beta, alpha);
    }
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toDouble();
    auto alpha_ = alpha.toDouble();
    auto mat1_ = checked_cast_tensor<CPUDoubleTensor>(mat1.pImpl,"mat1",5, false);
    auto mat2_ = checked_cast_tensor<CPUDoubleTensor>(mat2.pImpl,"mat2",6, false);
    THDoubleTensor_addmm(self_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    return self;
}
Tensor & CPUDoubleType::addmm_(Tensor & self, SparseTensor mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toDouble();
    auto alpha_ = alpha.toDouble();
    auto mat1_ = checked_cast_tensor<SparseCPUDoubleTensor>(mat1.tref.pImpl,"mat1",5,false);
    auto mat2_ = checked_cast_tensor<CPUDoubleTensor>(mat2.pImpl,"mat2",6, false);
    THSDoubleTensor_spaddmm(self_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
    return self;
}
Tensor & CPUDoubleType::s__addmv_out(Tensor & result, const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toDouble();
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toDouble();
    auto mat_ = checked_cast_tensor<CPUDoubleTensor>(mat.pImpl,"mat",4, false);
    auto vec_ = checked_cast_tensor<CPUDoubleTensor>(vec.pImpl,"vec",5, false);
    THDoubleTensor_addmv(result_->tensor, beta_, self_->tensor, alpha_, mat_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && mat_->isScalar() && vec_->isScalar());
    return result;
}
Tensor CPUDoubleType::s__addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toDouble();
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toDouble();
    auto mat_ = checked_cast_tensor<CPUDoubleTensor>(mat.pImpl,"mat",4, false);
    auto vec_ = checked_cast_tensor<CPUDoubleTensor>(vec.pImpl,"vec",5, false);
    THDoubleTensor_addmv(result_->tensor, beta_, self_->tensor, alpha_, mat_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && mat_->isScalar() && vec_->isScalar());
    return result;
}
Tensor & CPUDoubleType::_addmv_(Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toDouble();
    auto alpha_ = alpha.toDouble();
    auto mat_ = checked_cast_tensor<CPUDoubleTensor>(mat.pImpl,"mat",5, false);
    auto vec_ = checked_cast_tensor<CPUDoubleTensor>(vec.pImpl,"vec",6, false);
    THDoubleTensor_addmv(self_->tensor, beta_, self_->tensor, alpha_, mat_->tensor, vec_->tensor);
    return self;
}
Tensor & CPUDoubleType::s__addr_out(Tensor & result, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toDouble();
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toDouble();
    auto vec1_ = checked_cast_tensor<CPUDoubleTensor>(vec1.pImpl,"vec1",4, false);
    auto vec2_ = checked_cast_tensor<CPUDoubleTensor>(vec2.pImpl,"vec2",5, false);
    THDoubleTensor_addr(result_->tensor, beta_, self_->tensor, alpha_, vec1_->tensor, vec2_->tensor);
    result_->maybeScalar(self_->isScalar() && vec1_->isScalar() && vec2_->isScalar());
    return result;
}
Tensor CPUDoubleType::s__addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toDouble();
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toDouble();
    auto vec1_ = checked_cast_tensor<CPUDoubleTensor>(vec1.pImpl,"vec1",4, false);
    auto vec2_ = checked_cast_tensor<CPUDoubleTensor>(vec2.pImpl,"vec2",5, false);
    THDoubleTensor_addr(result_->tensor, beta_, self_->tensor, alpha_, vec1_->tensor, vec2_->tensor);
    result_->maybeScalar(self_->isScalar() && vec1_->isScalar() && vec2_->isScalar());
    return result;
}
Tensor & CPUDoubleType::_addr_(Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toDouble();
    auto alpha_ = alpha.toDouble();
    auto vec1_ = checked_cast_tensor<CPUDoubleTensor>(vec1.pImpl,"vec1",5, false);
    auto vec2_ = checked_cast_tensor<CPUDoubleTensor>(vec2.pImpl,"vec2",6, false);
    THDoubleTensor_addr(self_->tensor, beta_, self_->tensor, alpha_, vec1_->tensor, vec2_->tensor);
    return self;
}
Tensor & CPUDoubleType::_ger_out(Tensor & result, const Tensor & self, const Tensor & vec2) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.dim() == 0 ? 1 : self.size(0),vec2.dim() == 0 ? 1 : vec2.size(0) });
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto vec2_ = checked_cast_tensor<CPUDoubleTensor>(vec2.pImpl,"vec2",2, false);
    THDoubleTensor_addr(result_->tensor, double(0), result_->tensor, double(1), self_->tensor, vec2_->tensor);
    result_->maybeScalar(false);
    return result;
}
Tensor CPUDoubleType::_ger(const Tensor & self, const Tensor & vec2) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.dim() == 0 ? 1 : self.size(0),vec2.dim() == 0 ? 1 : vec2.size(0) });
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto vec2_ = checked_cast_tensor<CPUDoubleTensor>(vec2.pImpl,"vec2",2, false);
    THDoubleTensor_addr(result_->tensor, double(0), result_->tensor, double(1), self_->tensor, vec2_->tensor);
    result_->maybeScalar(false);
    return result;
}
Tensor & CPUDoubleType::_mv_out(Tensor & result, const Tensor & self, const Tensor & vec) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.size(0) });
    result.zero_();
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto vec_ = checked_cast_tensor<CPUDoubleTensor>(vec.pImpl,"vec",2, false);
    THDoubleTensor_addmv(result_->tensor, double(0), result_->tensor, double(1), self_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && vec_->isScalar());
    return result;
}
Tensor CPUDoubleType::_mv(const Tensor & self, const Tensor & vec) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.size(0) });
    result.zero_();
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto vec_ = checked_cast_tensor<CPUDoubleTensor>(vec.pImpl,"vec",2, false);
    THDoubleTensor_addmv(result_->tensor, double(0), result_->tensor, double(1), self_->tensor, vec_->tensor);
    result_->maybeScalar(self_->isScalar() && vec_->isScalar());
    return result;
}
Tensor & CPUDoubleType::_mm_out(Tensor & result, const Tensor & self, const Tensor & mat2) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.size(0),mat2.size(1) });
    result.zero_();
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CPUDoubleTensor>(mat2.pImpl,"mat2",2, false);
    THDoubleTensor_addmm(result_->tensor, double(0), result_->tensor, double(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CPUDoubleType::_mm(const Tensor & self, const Tensor & mat2) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.size(0),mat2.size(1) });
    result.zero_();
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CPUDoubleTensor>(mat2.pImpl,"mat2",2, false);
    THDoubleTensor_addmm(result_->tensor, double(0), result_->tensor, double(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CPUDoubleType::bmm_out(Tensor & result, const Tensor & self, const Tensor & mat2) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    result.resize_({ self.size(0),self.size(1),mat2.size(2) });
    result.zero_();
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CPUDoubleTensor>(mat2.pImpl,"mat2",2, false);
    THDoubleTensor_baddbmm(result_->tensor, double(0), result_->tensor, double(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor CPUDoubleType::bmm(const Tensor & self, const Tensor & mat2) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    result.resize_({ self.size(0),self.size(1),mat2.size(2) });
    result.zero_();
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto mat2_ = checked_cast_tensor<CPUDoubleTensor>(mat2.pImpl,"mat2",2, false);
    THDoubleTensor_baddbmm(result_->tensor, double(0), result_->tensor, double(1), self_->tensor, mat2_->tensor);
    result_->maybeScalar(self_->isScalar() && mat2_->isScalar());
    return result;
}
Tensor & CPUDoubleType::s_addbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toDouble();
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toDouble();
    auto batch1_ = checked_cast_tensor<CPUDoubleTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CPUDoubleTensor>(batch2.pImpl,"batch2",5, false);
    THDoubleTensor_addbmm(result_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor CPUDoubleType::s_addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toDouble();
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toDouble();
    auto batch1_ = checked_cast_tensor<CPUDoubleTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CPUDoubleTensor>(batch2.pImpl,"batch2",5, false);
    THDoubleTensor_addbmm(result_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor & CPUDoubleType::addbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toDouble();
    auto alpha_ = alpha.toDouble();
    auto batch1_ = checked_cast_tensor<CPUDoubleTensor>(batch1.pImpl,"batch1",5, false);
    auto batch2_ = checked_cast_tensor<CPUDoubleTensor>(batch2.pImpl,"batch2",6, false);
    THDoubleTensor_addbmm(self_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    return self;
}
Tensor & CPUDoubleType::s_baddbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto beta_ = beta.toDouble();
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toDouble();
    auto batch1_ = checked_cast_tensor<CPUDoubleTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CPUDoubleTensor>(batch2.pImpl,"batch2",5, false);
    THDoubleTensor_baddbmm(result_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor CPUDoubleType::s_baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto beta_ = beta.toDouble();
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto alpha_ = alpha.toDouble();
    auto batch1_ = checked_cast_tensor<CPUDoubleTensor>(batch1.pImpl,"batch1",4, false);
    auto batch2_ = checked_cast_tensor<CPUDoubleTensor>(batch2.pImpl,"batch2",5, false);
    THDoubleTensor_baddbmm(result_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    result_->maybeScalar(self_->isScalar() && batch1_->isScalar() && batch2_->isScalar());
    return result;
}
Tensor & CPUDoubleType::baddbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toDouble();
    auto alpha_ = alpha.toDouble();
    auto batch1_ = checked_cast_tensor<CPUDoubleTensor>(batch1.pImpl,"batch1",5, false);
    auto batch2_ = checked_cast_tensor<CPUDoubleTensor>(batch2.pImpl,"batch2",6, false);
    THDoubleTensor_baddbmm(self_->tensor, beta_, self_->tensor, alpha_, batch1_->tensor, batch2_->tensor);
    return self;
}
Tensor & CPUDoubleType::s_addcmul_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toDouble();
    auto tensor1_ = checked_cast_tensor<CPUDoubleTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CPUDoubleTensor>(tensor2.pImpl,"tensor2",4, false);
    THDoubleTensor_addcmul(result_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor CPUDoubleType::s_addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toDouble();
    auto tensor1_ = checked_cast_tensor<CPUDoubleTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CPUDoubleTensor>(tensor2.pImpl,"tensor2",4, false);
    THDoubleTensor_addcmul(result_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor & CPUDoubleType::s_addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toDouble();
    auto tensor1_ = checked_cast_tensor<CPUDoubleTensor>(tensor1.pImpl,"tensor1",4, false);
    auto tensor2_ = checked_cast_tensor<CPUDoubleTensor>(tensor2.pImpl,"tensor2",5, false);
    THDoubleTensor_addcmul(self_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    return self;
}
Tensor & CPUDoubleType::s_addcdiv_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toDouble();
    auto tensor1_ = checked_cast_tensor<CPUDoubleTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CPUDoubleTensor>(tensor2.pImpl,"tensor2",4, false);
    THDoubleTensor_addcdiv(result_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor CPUDoubleType::s_addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toDouble();
    auto tensor1_ = checked_cast_tensor<CPUDoubleTensor>(tensor1.pImpl,"tensor1",3, false);
    auto tensor2_ = checked_cast_tensor<CPUDoubleTensor>(tensor2.pImpl,"tensor2",4, false);
    THDoubleTensor_addcdiv(result_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    result_->maybeScalar(self_->isScalar() && tensor1_->isScalar() && tensor2_->isScalar());
    return result;
}
Tensor & CPUDoubleType::s_addcdiv_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto value_ = value.toDouble();
    auto tensor1_ = checked_cast_tensor<CPUDoubleTensor>(tensor1.pImpl,"tensor1",4, false);
    auto tensor2_ = checked_cast_tensor<CPUDoubleTensor>(tensor2.pImpl,"tensor2",5, false);
    THDoubleTensor_addcdiv(self_->tensor, self_->tensor, value_, tensor1_->tensor, tensor2_->tensor);
    return self;
}
std::tuple<Tensor &,Tensor &> CPUDoubleType::gesv_out(Tensor & solution, Tensor & lu, const Tensor & self, const Tensor & A) const {
    auto solution_ = checked_cast_tensor<CPUDoubleTensor>(solution.pImpl,"solution",0, false);
    auto lu_ = checked_cast_tensor<CPUDoubleTensor>(lu.pImpl,"lu",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto A_ = checked_cast_tensor<CPUDoubleTensor>(A.pImpl,"A",2, false);
    THDoubleTensor_gesv(solution_->tensor, lu_->tensor, self_->tensor, A_->tensor);
    bool maybe_scalar = self_->isScalar() && A_->isScalar();
    solution_->maybeScalar(maybe_scalar);
    lu_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(solution, lu);
}
std::tuple<Tensor,Tensor> CPUDoubleType::gesv(const Tensor & self, const Tensor & A) const {
    auto solution_ = new CPUDoubleTensor(context);
    auto solution = Tensor(solution_, false);
    auto lu_ = new CPUDoubleTensor(context);
    auto lu = Tensor(lu_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto A_ = checked_cast_tensor<CPUDoubleTensor>(A.pImpl,"A",2, false);
    THDoubleTensor_gesv(solution_->tensor, lu_->tensor, self_->tensor, A_->tensor);
    bool maybe_scalar = self_->isScalar() && A_->isScalar();
    solution_->maybeScalar(maybe_scalar);
    lu_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(solution, lu);
}
std::tuple<Tensor &,Tensor &> CPUDoubleType::gels_out(Tensor & res1, Tensor & res2, const Tensor & self, const Tensor & A) const {
    auto res1_ = checked_cast_tensor<CPUDoubleTensor>(res1.pImpl,"res1",0, false);
    auto res2_ = checked_cast_tensor<CPUDoubleTensor>(res2.pImpl,"res2",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto A_ = checked_cast_tensor<CPUDoubleTensor>(A.pImpl,"A",2, false);
    THDoubleTensor_gels(res1_->tensor, res2_->tensor, self_->tensor, A_->tensor);
    bool maybe_scalar = self_->isScalar() && A_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(res1, res2);
}
std::tuple<Tensor,Tensor> CPUDoubleType::gels(const Tensor & self, const Tensor & A) const {
    auto res1_ = new CPUDoubleTensor(context);
    auto res1 = Tensor(res1_, false);
    auto res2_ = new CPUDoubleTensor(context);
    auto res2 = Tensor(res2_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto A_ = checked_cast_tensor<CPUDoubleTensor>(A.pImpl,"A",2, false);
    THDoubleTensor_gels(res1_->tensor, res2_->tensor, self_->tensor, A_->tensor);
    bool maybe_scalar = self_->isScalar() && A_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(res1, res2);
}
std::tuple<Tensor &,Tensor &> CPUDoubleType::trtrs_out(Tensor & res1, Tensor & res2, const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) const {
    auto res1_ = checked_cast_tensor<CPUDoubleTensor>(res1.pImpl,"res1",0, false);
    auto res2_ = checked_cast_tensor<CPUDoubleTensor>(res2.pImpl,"res2",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto A_ = checked_cast_tensor<CPUDoubleTensor>(A.pImpl,"A",2, false);
    THDoubleTensor_trtrs(res1_->tensor, res2_->tensor, self_->tensor, A_->tensor, (upper) ? "U" : "L", (transpose) ? "T" : "N", (unitriangular) ? "U" : "N");
    bool maybe_scalar = self_->isScalar() && A_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(res1, res2);
}
std::tuple<Tensor,Tensor> CPUDoubleType::trtrs(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) const {
    auto res1_ = new CPUDoubleTensor(context);
    auto res1 = Tensor(res1_, false);
    auto res2_ = new CPUDoubleTensor(context);
    auto res2 = Tensor(res2_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto A_ = checked_cast_tensor<CPUDoubleTensor>(A.pImpl,"A",2, false);
    THDoubleTensor_trtrs(res1_->tensor, res2_->tensor, self_->tensor, A_->tensor, (upper) ? "U" : "L", (transpose) ? "T" : "N", (unitriangular) ? "U" : "N");
    bool maybe_scalar = self_->isScalar() && A_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(res1, res2);
}
std::tuple<Tensor &,Tensor &> CPUDoubleType::symeig_out(Tensor & res1, Tensor & res2, const Tensor & self, bool eigenvectors, bool upper) const {
    auto res1_ = checked_cast_tensor<CPUDoubleTensor>(res1.pImpl,"res1",0, false);
    auto res2_ = checked_cast_tensor<CPUDoubleTensor>(res2.pImpl,"res2",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_syev(res1_->tensor, res2_->tensor, self_->tensor, (eigenvectors) ? "V" : "N", (upper) ? "U" : "L");
    bool maybe_scalar = self_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(res1, res2);
}
std::tuple<Tensor,Tensor> CPUDoubleType::symeig(const Tensor & self, bool eigenvectors, bool upper) const {
    auto res1_ = new CPUDoubleTensor(context);
    auto res1 = Tensor(res1_, false);
    auto res2_ = new CPUDoubleTensor(context);
    auto res2 = Tensor(res2_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_syev(res1_->tensor, res2_->tensor, self_->tensor, (eigenvectors) ? "V" : "N", (upper) ? "U" : "L");
    bool maybe_scalar = self_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(res1, res2);
}
std::tuple<Tensor &,Tensor &> CPUDoubleType::eig_out(Tensor & res1, Tensor & res2, const Tensor & self, bool eigenvectors) const {
    auto res1_ = checked_cast_tensor<CPUDoubleTensor>(res1.pImpl,"res1",0, false);
    auto res2_ = checked_cast_tensor<CPUDoubleTensor>(res2.pImpl,"res2",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_geev(res1_->tensor, res2_->tensor, self_->tensor, (eigenvectors) ? "V" : "N");
    bool maybe_scalar = self_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(res1, res2);
}
std::tuple<Tensor,Tensor> CPUDoubleType::eig(const Tensor & self, bool eigenvectors) const {
    auto res1_ = new CPUDoubleTensor(context);
    auto res1 = Tensor(res1_, false);
    auto res2_ = new CPUDoubleTensor(context);
    auto res2 = Tensor(res2_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_geev(res1_->tensor, res2_->tensor, self_->tensor, (eigenvectors) ? "V" : "N");
    bool maybe_scalar = self_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(res1, res2);
}
std::tuple<Tensor &,Tensor &,Tensor &> CPUDoubleType::svd_out(Tensor & res1, Tensor & res2, Tensor & res3, const Tensor & self, bool some) const {
    auto res1_ = checked_cast_tensor<CPUDoubleTensor>(res1.pImpl,"res1",0, false);
    auto res2_ = checked_cast_tensor<CPUDoubleTensor>(res2.pImpl,"res2",0, false);
    auto res3_ = checked_cast_tensor<CPUDoubleTensor>(res3.pImpl,"res3",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_gesvd(res1_->tensor, res2_->tensor, res3_->tensor, self_->tensor, (some) ? "S" : "A");
    bool maybe_scalar = self_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    res3_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(res1, res2, res3);
}
std::tuple<Tensor,Tensor,Tensor> CPUDoubleType::svd(const Tensor & self, bool some) const {
    auto res1_ = new CPUDoubleTensor(context);
    auto res1 = Tensor(res1_, false);
    auto res2_ = new CPUDoubleTensor(context);
    auto res2 = Tensor(res2_, false);
    auto res3_ = new CPUDoubleTensor(context);
    auto res3 = Tensor(res3_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_gesvd(res1_->tensor, res2_->tensor, res3_->tensor, self_->tensor, (some) ? "S" : "A");
    bool maybe_scalar = self_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    res3_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(res1, res2, res3);
}
Tensor & CPUDoubleType::inverse_out(Tensor & output, const Tensor & self) const {
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_getri(output_->tensor, self_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUDoubleType::inverse(const Tensor & self) const {
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_getri(output_->tensor, self_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUDoubleType::potrf_out(Tensor & output, const Tensor & self, bool upper) const {
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_potrf(output_->tensor, self_->tensor, (upper) ? "U" : "L");
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUDoubleType::potrf(const Tensor & self, bool upper) const {
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_potrf(output_->tensor, self_->tensor, (upper) ? "U" : "L");
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUDoubleType::potrs_out(Tensor & result, const Tensor & self, const Tensor & input2, bool upper) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto input2_ = checked_cast_tensor<CPUDoubleTensor>(input2.pImpl,"input2",2, false);
    THDoubleTensor_potrs(result_->tensor, self_->tensor, input2_->tensor, (upper) ? "U" : "L");
    result_->maybeScalar(self_->isScalar() && input2_->isScalar());
    return result;
}
Tensor CPUDoubleType::potrs(const Tensor & self, const Tensor & input2, bool upper) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto input2_ = checked_cast_tensor<CPUDoubleTensor>(input2.pImpl,"input2",2, false);
    THDoubleTensor_potrs(result_->tensor, self_->tensor, input2_->tensor, (upper) ? "U" : "L");
    result_->maybeScalar(self_->isScalar() && input2_->isScalar());
    return result;
}
Tensor & CPUDoubleType::potri_out(Tensor & output, const Tensor & self, bool upper) const {
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_potri(output_->tensor, self_->tensor, (upper) ? "U" : "L");
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUDoubleType::potri(const Tensor & self, bool upper) const {
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_potri(output_->tensor, self_->tensor, (upper) ? "U" : "L");
    output_->maybeScalar(self_->isScalar());
    return output;
}
std::tuple<Tensor &,Tensor &> CPUDoubleType::pstrf_out(Tensor & res1, Tensor & res2, const Tensor & self, bool upper, Scalar tol) const {
    auto res1_ = checked_cast_tensor<CPUDoubleTensor>(res1.pImpl,"res1",0, false);
    auto res2_ = checked_cast_tensor<CPUIntTensor>(res2.pImpl,"res2",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto tol_ = tol.toDouble();
    THDoubleTensor_pstrf(res1_->tensor, res2_->tensor, self_->tensor, (upper) ? "U" : "L", tol_);
    res2 -= 1;  // LAPACK returns 1-indexed pivots
    return std::tuple<Tensor &, Tensor &>(res1, res2);
}
std::tuple<Tensor,Tensor> CPUDoubleType::pstrf(const Tensor & self, bool upper, Scalar tol) const {
    auto res1_ = new CPUDoubleTensor(context);
    auto res1 = Tensor(res1_, false);
    auto res2_ = new CPUIntTensor(context);
    auto res2 = Tensor(res2_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto tol_ = tol.toDouble();
    THDoubleTensor_pstrf(res1_->tensor, res2_->tensor, self_->tensor, (upper) ? "U" : "L", tol_);
    res2 -= 1;  // LAPACK returns 1-indexed pivots
    return std::tuple<Tensor, Tensor>(res1, res2);
}
std::tuple<Tensor &,Tensor &> CPUDoubleType::qr_out(Tensor & res1, Tensor & res2, const Tensor & self) const {
    auto res1_ = checked_cast_tensor<CPUDoubleTensor>(res1.pImpl,"res1",0, false);
    auto res2_ = checked_cast_tensor<CPUDoubleTensor>(res2.pImpl,"res2",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_qr(res1_->tensor, res2_->tensor, self_->tensor);
    bool maybe_scalar = self_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(res1, res2);
}
std::tuple<Tensor,Tensor> CPUDoubleType::qr(const Tensor & self) const {
    auto res1_ = new CPUDoubleTensor(context);
    auto res1 = Tensor(res1_, false);
    auto res2_ = new CPUDoubleTensor(context);
    auto res2 = Tensor(res2_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_qr(res1_->tensor, res2_->tensor, self_->tensor);
    bool maybe_scalar = self_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(res1, res2);
}
std::tuple<Tensor &,Tensor &> CPUDoubleType::geqrf_out(Tensor & res1, Tensor & res2, const Tensor & self) const {
    auto res1_ = checked_cast_tensor<CPUDoubleTensor>(res1.pImpl,"res1",0, false);
    auto res2_ = checked_cast_tensor<CPUDoubleTensor>(res2.pImpl,"res2",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_geqrf(res1_->tensor, res2_->tensor, self_->tensor);
    bool maybe_scalar = self_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(res1, res2);
}
std::tuple<Tensor,Tensor> CPUDoubleType::geqrf(const Tensor & self) const {
    auto res1_ = new CPUDoubleTensor(context);
    auto res1 = Tensor(res1_, false);
    auto res2_ = new CPUDoubleTensor(context);
    auto res2 = Tensor(res2_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    THDoubleTensor_geqrf(res1_->tensor, res2_->tensor, self_->tensor);
    bool maybe_scalar = self_->isScalar();
    res1_->maybeScalar(maybe_scalar);
    res2_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(res1, res2);
}
Tensor & CPUDoubleType::orgqr_out(Tensor & result, const Tensor & self, const Tensor & input2) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto input2_ = checked_cast_tensor<CPUDoubleTensor>(input2.pImpl,"input2",2, false);
    THDoubleTensor_orgqr(result_->tensor, self_->tensor, input2_->tensor);
    result_->maybeScalar(self_->isScalar() && input2_->isScalar());
    return result;
}
Tensor CPUDoubleType::orgqr(const Tensor & self, const Tensor & input2) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto input2_ = checked_cast_tensor<CPUDoubleTensor>(input2.pImpl,"input2",2, false);
    THDoubleTensor_orgqr(result_->tensor, self_->tensor, input2_->tensor);
    result_->maybeScalar(self_->isScalar() && input2_->isScalar());
    return result;
}
Tensor & CPUDoubleType::ormqr_out(Tensor & result, const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto input2_ = checked_cast_tensor<CPUDoubleTensor>(input2.pImpl,"input2",2, false);
    auto input3_ = checked_cast_tensor<CPUDoubleTensor>(input3.pImpl,"input3",3, false);
    THDoubleTensor_ormqr(result_->tensor, self_->tensor, input2_->tensor, input3_->tensor, (left) ? "L" : "R", (transpose) ? "T" : "N");
    result_->maybeScalar(self_->isScalar() && input2_->isScalar() && input3_->isScalar());
    return result;
}
Tensor CPUDoubleType::ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto input2_ = checked_cast_tensor<CPUDoubleTensor>(input2.pImpl,"input2",2, false);
    auto input3_ = checked_cast_tensor<CPUDoubleTensor>(input3.pImpl,"input3",3, false);
    THDoubleTensor_ormqr(result_->tensor, self_->tensor, input2_->tensor, input3_->tensor, (left) ? "L" : "R", (transpose) ? "T" : "N");
    result_->maybeScalar(self_->isScalar() && input2_->isScalar() && input3_->isScalar());
    return result;
}
std::tuple<Tensor &,Tensor &> CPUDoubleType::btrifact_out(Tensor & result, Tensor & pivots, const Tensor & self, bool pivot) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto pivots_ = checked_cast_tensor<CPUIntTensor>(pivots.pImpl,"pivots",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    THDoubleTensor_btrifact(result_->tensor, pivots_->tensor, NULL, pivot, self_->tensor);
    bool maybe_scalar = self_->isScalar();
    result_->maybeScalar(maybe_scalar);
    pivots_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(result, pivots);
}
std::tuple<Tensor,Tensor> CPUDoubleType::btrifact(const Tensor & self, bool pivot) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto pivots_ = new CPUIntTensor(context);
    auto pivots = Tensor(pivots_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    THDoubleTensor_btrifact(result_->tensor, pivots_->tensor, NULL, pivot, self_->tensor);
    bool maybe_scalar = self_->isScalar();
    result_->maybeScalar(maybe_scalar);
    pivots_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(result, pivots);
}
std::tuple<Tensor &,Tensor &,Tensor &> CPUDoubleType::btrifact_with_info_out(Tensor & result, Tensor & pivots, Tensor & info, const Tensor & self, bool pivot) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto pivots_ = checked_cast_tensor<CPUIntTensor>(pivots.pImpl,"pivots",0, false);
    auto info_ = checked_cast_tensor<CPUIntTensor>(info.pImpl,"info",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    THDoubleTensor_btrifact(result_->tensor, pivots_->tensor, info_->tensor, pivot, self_->tensor);
    bool maybe_scalar = self_->isScalar();
    result_->maybeScalar(maybe_scalar);
    pivots_->maybeScalar(maybe_scalar);
    info_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(result, pivots, info);
}
std::tuple<Tensor,Tensor,Tensor> CPUDoubleType::btrifact_with_info(const Tensor & self, bool pivot) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto pivots_ = new CPUIntTensor(context);
    auto pivots = Tensor(pivots_, false);
    auto info_ = new CPUIntTensor(context);
    auto info = Tensor(info_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    THDoubleTensor_btrifact(result_->tensor, pivots_->tensor, info_->tensor, pivot, self_->tensor);
    bool maybe_scalar = self_->isScalar();
    result_->maybeScalar(maybe_scalar);
    pivots_->maybeScalar(maybe_scalar);
    info_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(result, pivots, info);
}
Tensor & CPUDoubleType::btrisolve_out(Tensor & result, const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto LU_data_ = checked_cast_tensor<CPUDoubleTensor>(LU_data.pImpl,"LU_data",2, false);
    auto LU_pivots_ = checked_cast_tensor<CPUIntTensor>(LU_pivots.pImpl,"LU_pivots",3, false);
    THDoubleTensor_btrisolve(result_->tensor, self_->tensor, LU_data_->tensor, LU_pivots_->tensor);
    result_->maybeScalar(self_->isScalar() && LU_data_->isScalar() && LU_pivots_->isScalar());
    return result;
}
Tensor CPUDoubleType::btrisolve(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto LU_data_ = checked_cast_tensor<CPUDoubleTensor>(LU_data.pImpl,"LU_data",2, false);
    auto LU_pivots_ = checked_cast_tensor<CPUIntTensor>(LU_pivots.pImpl,"LU_pivots",3, false);
    THDoubleTensor_btrisolve(result_->tensor, self_->tensor, LU_data_->tensor, LU_pivots_->tensor);
    result_->maybeScalar(self_->isScalar() && LU_data_->isScalar() && LU_pivots_->isScalar());
    return result;
}
Tensor & CPUDoubleType::random_(Tensor & self, int64_t from, int64_t to, Generator * generator) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    THDoubleTensor_clampedRandom(self_->tensor, generator_->generator, from, to);
    return self;
}
Tensor & CPUDoubleType::random_(Tensor & self, int64_t to, Generator * generator) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    THDoubleTensor_cappedRandom(self_->tensor, generator_->generator, to);
    return self;
}
Tensor & CPUDoubleType::random_(Tensor & self, Generator * generator) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    THDoubleTensor_random(self_->tensor, generator_->generator);
    return self;
}
Tensor & CPUDoubleType::multinomial_out(Tensor & result, const Tensor & self, int64_t num_samples, bool replacement, Generator * generator) const {
    auto result_ = checked_cast_tensor<CPULongTensor>(result.pImpl,"result",0, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    THDoubleTensor_multinomial(result_->tensor, generator_->generator, self_->tensor, num_samples, replacement);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor CPUDoubleType::multinomial(const Tensor & self, int64_t num_samples, bool replacement, Generator * generator) const {
    auto result_ = new CPULongTensor(context);
    auto result = Tensor(result_, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    THDoubleTensor_multinomial(result_->tensor, generator_->generator, self_->tensor, num_samples, replacement);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::uniform_(Tensor & self, double from, double to, Generator * generator) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    THDoubleTensor_uniform(self_->tensor, generator_->generator, from, to);
    return self;
}
Tensor & CPUDoubleType::normal_out(Tensor & output, const Tensor & mean, double std, Generator * generator) const {
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",0, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    auto mean_ = checked_cast_tensor<CPUDoubleTensor>(mean.pImpl,"mean",2, false);
    THDoubleTensor_normal_means(output_->tensor, generator_->generator, mean_->tensor, std);
    output_->maybeScalar(mean_->isScalar());
    return output;
}
Tensor CPUDoubleType::normal(const Tensor & mean, double std, Generator * generator) const {
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    auto mean_ = checked_cast_tensor<CPUDoubleTensor>(mean.pImpl,"mean",2, false);
    THDoubleTensor_normal_means(output_->tensor, generator_->generator, mean_->tensor, std);
    output_->maybeScalar(mean_->isScalar());
    return output;
}
Tensor & CPUDoubleType::normal_out(Tensor & output, double mean, const Tensor & std, Generator * generator) const {
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",0, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    auto std_ = checked_cast_tensor<CPUDoubleTensor>(std.pImpl,"std",3, false);
    THDoubleTensor_normal_stddevs(output_->tensor, generator_->generator, mean, std_->tensor);
    output_->maybeScalar(std_->isScalar());
    return output;
}
Tensor CPUDoubleType::normal(double mean, const Tensor & std, Generator * generator) const {
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    auto std_ = checked_cast_tensor<CPUDoubleTensor>(std.pImpl,"std",3, false);
    THDoubleTensor_normal_stddevs(output_->tensor, generator_->generator, mean, std_->tensor);
    output_->maybeScalar(std_->isScalar());
    return output;
}
Tensor & CPUDoubleType::normal_out(Tensor & output, const Tensor & mean, const Tensor & std, Generator * generator) const {
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",0, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    auto mean_ = checked_cast_tensor<CPUDoubleTensor>(mean.pImpl,"mean",2, false);
    auto std_ = checked_cast_tensor<CPUDoubleTensor>(std.pImpl,"std",3, false);
    THDoubleTensor_normal_means_stddevs(output_->tensor, generator_->generator, mean_->tensor, std_->tensor);
    output_->maybeScalar(mean_->isScalar() && std_->isScalar());
    return output;
}
Tensor CPUDoubleType::normal(const Tensor & mean, const Tensor & std, Generator * generator) const {
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    auto mean_ = checked_cast_tensor<CPUDoubleTensor>(mean.pImpl,"mean",2, false);
    auto std_ = checked_cast_tensor<CPUDoubleTensor>(std.pImpl,"std",3, false);
    THDoubleTensor_normal_means_stddevs(output_->tensor, generator_->generator, mean_->tensor, std_->tensor);
    output_->maybeScalar(mean_->isScalar() && std_->isScalar());
    return output;
}
Tensor & CPUDoubleType::normal_(Tensor & self, double mean, double std, Generator * generator) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    THDoubleTensor_normal(self_->tensor, generator_->generator, mean, std);
    return self;
}
Tensor & CPUDoubleType::cauchy_(Tensor & self, double median, double sigma, Generator * generator) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    THDoubleTensor_cauchy(self_->tensor, generator_->generator, median, sigma);
    return self;
}
Tensor & CPUDoubleType::log_normal_(Tensor & self, double mean, double std, Generator * generator) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    THDoubleTensor_logNormal(self_->tensor, generator_->generator, mean, std);
    return self;
}
Tensor & CPUDoubleType::exponential_(Tensor & self, double lambd, Generator * generator) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    THDoubleTensor_exponential(self_->tensor, generator_->generator, lambd);
    return self;
}
Tensor & CPUDoubleType::geometric_(Tensor & self, double p, Generator * generator) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    THDoubleTensor_geometric(self_->tensor, generator_->generator, p);
    return self;
}
Tensor & CPUDoubleType::bernoulli_out(Tensor & output, const Tensor & self, Generator * generator) const {
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",0, false);
    output.resize_(self.sizes());
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    THDoubleTensor_bernoulli_Tensor(output_->tensor, generator_->generator, self_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUDoubleType::bernoulli(const Tensor & self, Generator * generator) const {
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    output.resize_(self.sizes());
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    THDoubleTensor_bernoulli_Tensor(output_->tensor, generator_->generator, self_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUDoubleType::_standard_gamma_out(Tensor & output, const Tensor & self, Generator * generator) const {
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",0, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    THDoubleTensor_standard_gamma(output_->tensor, generator_->generator, self_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUDoubleType::_standard_gamma(const Tensor & self, Generator * generator) const {
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    THDoubleTensor_standard_gamma(output_->tensor, generator_->generator, self_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUDoubleType::_dirichlet_grad_out(Tensor & output, const Tensor & x, const Tensor & alpha, const Tensor & total) const {
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",0, false);
    auto x_ = checked_cast_tensor<CPUDoubleTensor>(x.pImpl,"x",1, false);
    auto alpha_ = checked_cast_tensor<CPUDoubleTensor>(alpha.pImpl,"alpha",2, false);
    auto total_ = checked_cast_tensor<CPUDoubleTensor>(total.pImpl,"total",3, false);
    THDoubleTensor_dirichlet_grad(output_->tensor, x_->tensor, alpha_->tensor, total_->tensor);
    output_->maybeScalar(x_->isScalar() && alpha_->isScalar() && total_->isScalar());
    return output;
}
Tensor CPUDoubleType::_dirichlet_grad(const Tensor & x, const Tensor & alpha, const Tensor & total) const {
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    auto x_ = checked_cast_tensor<CPUDoubleTensor>(x.pImpl,"x",1, false);
    auto alpha_ = checked_cast_tensor<CPUDoubleTensor>(alpha.pImpl,"alpha",2, false);
    auto total_ = checked_cast_tensor<CPUDoubleTensor>(total.pImpl,"total",3, false);
    THDoubleTensor_dirichlet_grad(output_->tensor, x_->tensor, alpha_->tensor, total_->tensor);
    output_->maybeScalar(x_->isScalar() && alpha_->isScalar() && total_->isScalar());
    return output;
}
Tensor CPUDoubleType::tensor(Storage & storage, int64_t storageOffset, IntList size, IntList stride) const {
    auto storage_ = checked_cast_storage<CPUDoubleStorage>(&storage,"storage",1);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    return Tensor((new CPUDoubleTensor(context, THDoubleTensor_newWithStorage(storage_->storage, storageOffset, size_, stride_)))->maybeScalar(size.size() == 0),false);
}
Tensor CPUDoubleType::tensor(IntList size) const {
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new CPUDoubleTensor(context, THDoubleTensor_newWithSize(size_, NULL)))->maybeScalar(size.size() == 0),false);
}
Tensor CPUDoubleType::tensor(IntList size, IntList stride) const {
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    return Tensor((new CPUDoubleTensor(context, THDoubleTensor_newWithSize(size_, stride_)))->maybeScalar(size.size() == 0),false);
}
Tensor CPUDoubleType::tensor() const {
    return Tensor((new CPUDoubleTensor(context, THDoubleTensor_new())),false);
}
Tensor CPUDoubleType::alias(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPUDoubleTensor(context, THDoubleTensor_newWithTensor(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor & CPUDoubleType::as_strided_out(Tensor & result, const Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    auto result_ = checked_cast_tensor<CPUDoubleTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    if (storage_offset == -1) {
      storage_offset = self_->tensor->storageOffset;
    }
    THDoubleTensor_setStorage(result_->tensor, self_->tensor->storage, storage_offset, size_, stride_);
    result_->maybeScalar(size.size() == 0);
    return result;
}
Tensor CPUDoubleType::as_strided(const Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    auto result_ = new CPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    if (storage_offset == -1) {
      storage_offset = self_->tensor->storageOffset;
    }
    THDoubleTensor_setStorage(result_->tensor, self_->tensor->storage, storage_offset, size_, stride_);
    result_->maybeScalar(size.size() == 0);
    return result;
}
Tensor & CPUDoubleType::as_strided_(Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    if (storage_offset == -1) {
      storage_offset = self_->tensor->storageOffset;
    }
    THDoubleTensor_setStorage(self_->tensor, self_->tensor->storage, storage_offset, size_, stride_);
    self_->maybeScalar(size.size() == 0);
    return self;
}
Tensor & CPUDoubleType::_cat_out(Tensor & self, TensorList tensors, int64_t dim) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",0, false);
    auto tensors_ = tensor_list_checked_cast<CPUDoubleTensor, Tensor, THDoubleTensor>(tensors,"tensors",1);
    THDoubleTensor_catArray(self_->tensor, tensors_.data(), tensors_.size(), dim);
    return self;
}
Tensor CPUDoubleType::_cat(TensorList tensors, int64_t dim) const {
    auto self_ = new CPUDoubleTensor(context);
    auto self = Tensor(self_, false);
    auto tensors_ = tensor_list_checked_cast<CPUDoubleTensor, Tensor, THDoubleTensor>(tensors,"tensors",1);
    THDoubleTensor_catArray(self_->tensor, tensors_.data(), tensors_.size(), dim);
    return self;
}
Tensor CPUDoubleType::_sparse_mask(const Tensor & self, SparseTensor mask) const {
    auto result_ = new SparseCPUDoubleTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto mask_ = checked_cast_tensor<SparseCPUDoubleTensor>(mask.tref.pImpl,"mask",2,false);
    THDoubleTensor_sparseMask(result_->tensor, self_->tensor, mask_->tensor);
    result_->maybeScalar(self_->isScalar());
    return result;
}
Tensor & CPUDoubleType::binary_cross_entropy_forward_out(Tensor & output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CPUDoubleTensor>(target.pImpl,"target",2, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",3, true);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",5, false);
    THNN_DoubleBCECriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, weight_ ? weight_->tensor : NULL, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor CPUDoubleType::binary_cross_entropy_forward(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CPUDoubleTensor>(target.pImpl,"target",2, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",3, true);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_DoubleBCECriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, weight_ ? weight_->tensor : NULL, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor & CPUDoubleType::binary_cross_entropy_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CPUDoubleTensor>(target.pImpl,"target",3, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",4, true);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",6, false);
    THNN_DoubleBCECriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, weight_ ? weight_->tensor : NULL, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::binary_cross_entropy_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CPUDoubleTensor>(target.pImpl,"target",3, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",4, true);
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleBCECriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, weight_ ? weight_->tensor : NULL, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUDoubleType::kl_div_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CPUDoubleTensor>(target.pImpl,"target",2, false);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",4, false);
    THNN_DoubleDistKLDivCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor CPUDoubleType::kl_div_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CPUDoubleTensor>(target.pImpl,"target",2, false);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_DoubleDistKLDivCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor & CPUDoubleType::kl_div_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CPUDoubleTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",5, false);
    THNN_DoubleDistKLDivCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::kl_div_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CPUDoubleTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleDistKLDivCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUDoubleType::l1_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CPUDoubleTensor>(target.pImpl,"target",2, false);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",4, false);
    THNN_DoubleAbsCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor CPUDoubleType::l1_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CPUDoubleTensor>(target.pImpl,"target",2, false);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_DoubleAbsCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor & CPUDoubleType::l1_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CPUDoubleTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",5, false);
    THNN_DoubleAbsCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CPUDoubleTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleAbsCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUDoubleType::mse_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CPUDoubleTensor>(target.pImpl,"target",2, false);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",4, false);
    THNN_DoubleMSECriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor CPUDoubleType::mse_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CPUDoubleTensor>(target.pImpl,"target",2, false);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_DoubleMSECriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor & CPUDoubleType::mse_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CPUDoubleTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",5, false);
    THNN_DoubleMSECriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::mse_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CPUDoubleTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleMSECriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUDoubleType::multi_margin_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CPULongTensor>(target.pImpl,"target",2, false);
    auto p_ = p.toDouble();
    auto margin_ = margin.toDouble();
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",5, true);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",7, false);
    THNN_DoubleMultiMarginCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, p_, weight_ ? weight_->tensor : NULL, margin_, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor CPUDoubleType::multi_margin_loss_forward(const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CPULongTensor>(target.pImpl,"target",2, false);
    auto p_ = p.toDouble();
    auto margin_ = margin.toDouble();
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",5, true);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_DoubleMultiMarginCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, p_, weight_ ? weight_->tensor : NULL, margin_, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor & CPUDoubleType::multi_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CPULongTensor>(target.pImpl,"target",3, false);
    auto p_ = p.toDouble();
    auto margin_ = margin.toDouble();
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",6, true);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",8, false);
    THNN_DoubleMultiMarginCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, p_, weight_ ? weight_->tensor : NULL, margin_, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::multi_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CPULongTensor>(target.pImpl,"target",3, false);
    auto p_ = p.toDouble();
    auto margin_ = margin.toDouble();
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",6, true);
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleMultiMarginCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, p_, weight_ ? weight_->tensor : NULL, margin_, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &> CPUDoubleType::multilabel_margin_loss_forward_out(Tensor & output, Tensor & is_target, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CPULongTensor>(target.pImpl,"target",2, false);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",4, false);
    auto is_target_ = checked_cast_tensor<CPUDoubleTensor>(is_target.pImpl,"is_target",4, false);
    THNN_DoubleMultiLabelMarginCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, is_target_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    is_target_->maybeScalar(target_->isScalar());
    return std::tuple<Tensor &, Tensor &>(output, is_target);
}
std::tuple<Tensor,Tensor> CPUDoubleType::multilabel_margin_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CPULongTensor>(target.pImpl,"target",2, false);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    auto is_target_ = new CPUDoubleTensor(context);
    auto is_target = Tensor(is_target_, false);
    THNN_DoubleMultiLabelMarginCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, is_target_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    is_target_->maybeScalar(target_->isScalar());
    return std::tuple<Tensor, Tensor>(output, is_target);
}
Tensor & CPUDoubleType::multilabel_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce, const Tensor & is_target) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CPULongTensor>(target.pImpl,"target",3, false);
    auto is_target_ = checked_cast_tensor<CPUDoubleTensor>(is_target.pImpl,"is_target",6, false);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",6, false);
    THNN_DoubleMultiLabelMarginCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, is_target_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::multilabel_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce, const Tensor & is_target) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CPULongTensor>(target.pImpl,"target",3, false);
    auto is_target_ = checked_cast_tensor<CPUDoubleTensor>(is_target.pImpl,"is_target",6, false);
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleMultiLabelMarginCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, is_target_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &> CPUDoubleType::nll_loss_forward_out(Tensor & output, Tensor & total_weight, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CPULongTensor>(target.pImpl,"target",2, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",3, true);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",6, false);
    auto total_weight_ = checked_cast_tensor<CPUDoubleTensor>(total_weight.pImpl,"total_weight",6, false);
    THNN_DoubleClassNLLCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, weight_ ? weight_->tensor : NULL, total_weight_->tensor, ignore_index, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    total_weight_->maybeScalar(true);
    return std::tuple<Tensor &, Tensor &>(output, total_weight);
}
std::tuple<Tensor,Tensor> CPUDoubleType::nll_loss_forward(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CPULongTensor>(target.pImpl,"target",2, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",3, true);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    auto total_weight_ = new CPUDoubleTensor(context);
    auto total_weight = Tensor(total_weight_, false);
    THNN_DoubleClassNLLCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, weight_ ? weight_->tensor : NULL, total_weight_->tensor, ignore_index, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    total_weight_->maybeScalar(true);
    return std::tuple<Tensor, Tensor>(output, total_weight);
}
Tensor & CPUDoubleType::nll_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CPULongTensor>(target.pImpl,"target",3, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",4, true);
    auto total_weight_ = checked_cast_tensor<CPUDoubleTensor>(total_weight.pImpl,"total_weight",8, false);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",8, false);
    THNN_DoubleClassNLLCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, weight_ ? weight_->tensor : NULL, total_weight_->tensor, ignore_index, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::nll_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CPULongTensor>(target.pImpl,"target",3, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",4, true);
    auto total_weight_ = checked_cast_tensor<CPUDoubleTensor>(total_weight.pImpl,"total_weight",8, false);
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleClassNLLCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, weight_ ? weight_->tensor : NULL, total_weight_->tensor, ignore_index, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &> CPUDoubleType::nll_loss2d_forward_out(Tensor & output, Tensor & total_weight, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CPULongTensor>(target.pImpl,"target",2, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",3, true);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",6, false);
    auto total_weight_ = checked_cast_tensor<CPUDoubleTensor>(total_weight.pImpl,"total_weight",6, false);
    THNN_DoubleSpatialClassNLLCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, weight_ ? weight_->tensor : NULL, total_weight_->tensor, ignore_index, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    total_weight_->maybeScalar(true);
    return std::tuple<Tensor &, Tensor &>(output, total_weight);
}
std::tuple<Tensor,Tensor> CPUDoubleType::nll_loss2d_forward(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CPULongTensor>(target.pImpl,"target",2, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",3, true);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    auto total_weight_ = new CPUDoubleTensor(context);
    auto total_weight = Tensor(total_weight_, false);
    THNN_DoubleSpatialClassNLLCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, weight_ ? weight_->tensor : NULL, total_weight_->tensor, ignore_index, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    total_weight_->maybeScalar(true);
    return std::tuple<Tensor, Tensor>(output, total_weight);
}
Tensor & CPUDoubleType::nll_loss2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CPULongTensor>(target.pImpl,"target",3, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",4, true);
    auto total_weight_ = checked_cast_tensor<CPUDoubleTensor>(total_weight.pImpl,"total_weight",8, false);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",8, false);
    THNN_DoubleSpatialClassNLLCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, weight_ ? weight_->tensor : NULL, total_weight_->tensor, ignore_index, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::nll_loss2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CPULongTensor>(target.pImpl,"target",3, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",4, true);
    auto total_weight_ = checked_cast_tensor<CPUDoubleTensor>(total_weight.pImpl,"total_weight",8, false);
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleSpatialClassNLLCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, weight_ ? weight_->tensor : NULL, total_weight_->tensor, ignore_index, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUDoubleType::smooth_l1_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CPUDoubleTensor>(target.pImpl,"target",2, false);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",4, false);
    THNN_DoubleSmoothL1Criterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor CPUDoubleType::smooth_l1_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CPUDoubleTensor>(target.pImpl,"target",2, false);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_DoubleSmoothL1Criterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor & CPUDoubleType::smooth_l1_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CPUDoubleTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",5, false);
    THNN_DoubleSmoothL1Criterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::smooth_l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CPUDoubleTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleSmoothL1Criterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUDoubleType::soft_margin_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CPUDoubleTensor>(target.pImpl,"target",2, false);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",4, false);
    THNN_DoubleSoftMarginCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor CPUDoubleType::soft_margin_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto target_ = checked_cast_tensor<CPUDoubleTensor>(target.pImpl,"target",2, false);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_DoubleSoftMarginCriterion_updateOutput(context->thc_state, self_->tensor, target_->tensor, output_->tensor, size_average, reduce);
    output_->maybeScalar(reduce || self_->isScalar());
    return output;
}
Tensor & CPUDoubleType::soft_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CPUDoubleTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",5, false);
    THNN_DoubleSoftMarginCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::soft_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto target_ = checked_cast_tensor<CPUDoubleTensor>(target.pImpl,"target",3, false);
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleSoftMarginCriterion_updateGradInput(context->thc_state, self_->tensor, target_->tensor, grad_output_->tensor, grad_input_->tensor, size_average, reduce);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUDoubleType::elu_forward_out(Tensor & output, const Tensor & self, Scalar alpha, Scalar scale) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toDouble();
    auto scale_ = scale.toDouble();
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",3, false);
    THNN_DoubleELU_updateOutput(context->thc_state, self_->tensor, output_->tensor, alpha_, scale_, false);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUDoubleType::elu_forward(const Tensor & self, Scalar alpha, Scalar scale) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toDouble();
    auto scale_ = scale.toDouble();
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_DoubleELU_updateOutput(context->thc_state, self_->tensor, output_->tensor, alpha_, scale_, false);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUDoubleType::elu_backward_out(Tensor & grad_input, const Tensor & grad_output, Scalar alpha, Scalar scale, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto alpha_ = alpha.toDouble();
    auto scale_ = scale.toDouble();
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",4, false);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_DoubleELU_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, output_->tensor, alpha_, scale_);
    grad_input_->maybeScalar(output_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::elu_backward(const Tensor & grad_output, Scalar alpha, Scalar scale, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto alpha_ = alpha.toDouble();
    auto scale_ = scale.toDouble();
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",4, false);
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleELU_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, output_->tensor, alpha_, scale_);
    grad_input_->maybeScalar(output_->isScalar());
    return grad_input;
}
Tensor & CPUDoubleType::elu_forward_(Tensor & self, Scalar alpha, Scalar scale) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto alpha_ = alpha.toDouble();
    auto scale_ = scale.toDouble();
    THNN_DoubleELU_updateOutput(context->thc_state, self_->tensor, self_->tensor, alpha_, scale_, true);
    return self;
}
Tensor & CPUDoubleType::glu_forward_out(Tensor & output, const Tensor & self, int64_t dim) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",2, false);
    THNN_DoubleGatedLinear_updateOutput(context->thc_state, self_->tensor, output_->tensor, dim);
    output_->maybeScalar(false);
    return output;
}
Tensor CPUDoubleType::glu_forward(const Tensor & self, int64_t dim) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_DoubleGatedLinear_updateOutput(context->thc_state, self_->tensor, output_->tensor, dim);
    output_->maybeScalar(false);
    return output;
}
Tensor & CPUDoubleType::glu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    dim = maybe_wrap_dim(dim, self_);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_DoubleGatedLinear_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, dim);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::glu_backward(const Tensor & grad_output, const Tensor & self, int64_t dim) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    dim = maybe_wrap_dim(dim, self_);
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleGatedLinear_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, dim);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUDoubleType::hardshrink_forward_out(Tensor & output, const Tensor & self, Scalar lambd) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto lambd_ = lambd.toDouble();
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",2, false);
    THNN_DoubleHardShrink_updateOutput(context->thc_state, self_->tensor, output_->tensor, lambd_);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUDoubleType::hardshrink_forward(const Tensor & self, Scalar lambd) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto lambd_ = lambd.toDouble();
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_DoubleHardShrink_updateOutput(context->thc_state, self_->tensor, output_->tensor, lambd_);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUDoubleType::hardshrink_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar lambd) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto lambd_ = lambd.toDouble();
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_DoubleHardShrink_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, lambd_);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::hardshrink_backward(const Tensor & grad_output, const Tensor & self, Scalar lambd) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto lambd_ = lambd.toDouble();
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleHardShrink_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, lambd_);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUDoubleType::hardtanh_forward_out(Tensor & output, const Tensor & self, Scalar min_val, Scalar max_val) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto min_val_ = min_val.toDouble();
    auto max_val_ = max_val.toDouble();
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",3, false);
    THNN_DoubleHardTanh_updateOutput(context->thc_state, self_->tensor, output_->tensor, min_val_, max_val_, false);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUDoubleType::hardtanh_forward(const Tensor & self, Scalar min_val, Scalar max_val) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto min_val_ = min_val.toDouble();
    auto max_val_ = max_val.toDouble();
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_DoubleHardTanh_updateOutput(context->thc_state, self_->tensor, output_->tensor, min_val_, max_val_, false);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUDoubleType::hardtanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto min_val_ = min_val.toDouble();
    auto max_val_ = max_val.toDouble();
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_DoubleHardTanh_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, min_val_, max_val_, false);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::hardtanh_backward(const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto min_val_ = min_val.toDouble();
    auto max_val_ = max_val.toDouble();
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleHardTanh_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, min_val_, max_val_, false);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUDoubleType::hardtanh_forward_(Tensor & self, Scalar min_val, Scalar max_val) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto min_val_ = min_val.toDouble();
    auto max_val_ = max_val.toDouble();
    THNN_DoubleHardTanh_updateOutput(context->thc_state, self_->tensor, self_->tensor, min_val_, max_val_, true);
    return self;
}
Tensor & CPUDoubleType::leaky_relu_forward_out(Tensor & output, const Tensor & self, Scalar negative_slope) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto negative_slope_ = negative_slope.toDouble();
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",2, false);
    THNN_DoubleLeakyReLU_updateOutput(context->thc_state, self_->tensor, output_->tensor, negative_slope_, false);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUDoubleType::leaky_relu_forward(const Tensor & self, Scalar negative_slope) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto negative_slope_ = negative_slope.toDouble();
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_DoubleLeakyReLU_updateOutput(context->thc_state, self_->tensor, output_->tensor, negative_slope_, false);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUDoubleType::leaky_relu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar negative_slope) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto negative_slope_ = negative_slope.toDouble();
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_DoubleLeakyReLU_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, negative_slope_, false);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::leaky_relu_backward(const Tensor & grad_output, const Tensor & self, Scalar negative_slope) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto negative_slope_ = negative_slope.toDouble();
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleLeakyReLU_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, negative_slope_, false);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUDoubleType::leaky_relu_forward_(Tensor & self, Scalar negative_slope) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto negative_slope_ = negative_slope.toDouble();
    THNN_DoubleLeakyReLU_updateOutput(context->thc_state, self_->tensor, self_->tensor, negative_slope_, true);
    return self;
}
std::tuple<Tensor &,Tensor &> CPUDoubleType::log_sigmoid_forward_out(Tensor & output, Tensor & buffer, const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",1, false);
    auto buffer_ = checked_cast_tensor<CPUDoubleTensor>(buffer.pImpl,"buffer",1, false);
    THNN_DoubleLogSigmoid_updateOutput(context->thc_state, self_->tensor, output_->tensor, buffer_->tensor);
    output_->maybeScalar(self_->isScalar());
    buffer_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &>(output, buffer);
}
std::tuple<Tensor,Tensor> CPUDoubleType::log_sigmoid_forward(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    auto buffer_ = new CPUDoubleTensor(context);
    auto buffer = Tensor(buffer_, false);
    THNN_DoubleLogSigmoid_updateOutput(context->thc_state, self_->tensor, output_->tensor, buffer_->tensor);
    output_->maybeScalar(self_->isScalar());
    buffer_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor>(output, buffer);
}
Tensor & CPUDoubleType::log_sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & buffer) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto buffer_ = checked_cast_tensor<CPUDoubleTensor>(buffer.pImpl,"buffer",3, false);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_DoubleLogSigmoid_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, buffer_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::log_sigmoid_backward(const Tensor & grad_output, const Tensor & self, const Tensor & buffer) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto buffer_ = checked_cast_tensor<CPUDoubleTensor>(buffer.pImpl,"buffer",3, false);
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleLogSigmoid_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, buffer_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUDoubleType::log_softmax_forward_out(Tensor & output, const Tensor & self, int64_t dim) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",2, false);
    THNN_DoubleLogSoftMax_updateOutput(context->thc_state, self_->tensor, output_->tensor, dim);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUDoubleType::log_softmax_forward(const Tensor & self, int64_t dim) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_DoubleLogSoftMax_updateOutput(context->thc_state, self_->tensor, output_->tensor, dim);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUDoubleType::log_softmax_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",4, false);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_DoubleLogSoftMax_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, output_->tensor, dim);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::log_softmax_backward(const Tensor & grad_output, const Tensor & self, int64_t dim, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",4, false);
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleLogSoftMax_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, output_->tensor, dim);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUDoubleType::prelu_forward_out(Tensor & output, const Tensor & self, const Tensor & weight) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",2, false);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",2, false);
    THNN_DoublePReLU_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUDoubleType::prelu_forward(const Tensor & self, const Tensor & weight) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",2, false);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_DoublePReLU_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
std::tuple<Tensor &,Tensor &> CPUDoubleType::prelu_backward_out(Tensor & grad_input, Tensor & grad_weight, const Tensor & grad_output, const Tensor & self, const Tensor & weight) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",3, false);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",3, true);
    auto grad_weight_ = checked_cast_tensor<CPUDoubleTensor>(grad_weight.pImpl,"grad_weight",3, true);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    if (grad_input_) THNN_DoublePReLU_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor);
    if (grad_weight_) THNN_DoublePReLU_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &>(grad_input, grad_weight);
}
std::tuple<Tensor,Tensor> CPUDoubleType::prelu_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, std::array<bool,2> output_mask) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",3, false);
    auto grad_input_ = output_mask[0] ? new CPUDoubleTensor(context) : nullptr;
    auto grad_input = Tensor(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_input_, false);
    auto grad_weight_ = output_mask[1] ? new CPUDoubleTensor(context) : nullptr;
    auto grad_weight = Tensor(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_weight_, false);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    if (grad_input_) THNN_DoublePReLU_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor);
    if (grad_weight_) THNN_DoublePReLU_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor>(grad_input, grad_weight);
}
Tensor & CPUDoubleType::rrelu_with_noise_forward_out(Tensor & output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto noise_ = checked_cast_tensor<CPUDoubleTensor>(noise.pImpl,"noise",2, false);
    auto lower_ = lower.toDouble();
    auto upper_ = upper.toDouble();
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",6, false);
    THNN_DoubleRReLU_updateOutput(context->thc_state, self_->tensor, output_->tensor, noise_->tensor, lower_, upper_, training, false, generator_->generator);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUDoubleType::rrelu_with_noise_forward(const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto noise_ = checked_cast_tensor<CPUDoubleTensor>(noise.pImpl,"noise",2, false);
    auto lower_ = lower.toDouble();
    auto upper_ = upper.toDouble();
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_DoubleRReLU_updateOutput(context->thc_state, self_->tensor, output_->tensor, noise_->tensor, lower_, upper_, training, false, generator_->generator);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUDoubleType::rrelu_with_noise_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto noise_ = checked_cast_tensor<CPUDoubleTensor>(noise.pImpl,"noise",3, false);
    auto lower_ = lower.toDouble();
    auto upper_ = upper.toDouble();
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",6, false);
    THNN_DoubleRReLU_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, noise_->tensor, lower_, upper_, training, false);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::rrelu_with_noise_backward(const Tensor & grad_output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto noise_ = checked_cast_tensor<CPUDoubleTensor>(noise.pImpl,"noise",3, false);
    auto lower_ = lower.toDouble();
    auto upper_ = upper.toDouble();
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleRReLU_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, noise_->tensor, lower_, upper_, training, false);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUDoubleType::rrelu_with_noise_forward_(Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto noise_ = checked_cast_tensor<CPUDoubleTensor>(noise.pImpl,"noise",2, false);
    auto lower_ = lower.toDouble();
    auto upper_ = upper.toDouble();
    auto generator_ = check_generator<CPUGenerator>(generator, &context->defaultGenerator(backend()));
    THNN_DoubleRReLU_updateOutput(context->thc_state, self_->tensor, self_->tensor, noise_->tensor, lower_, upper_, training, true, generator_->generator);
    return self;
}
Tensor & CPUDoubleType::softmax_forward_out(Tensor & output, const Tensor & self, int64_t dim) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",2, false);
    THNN_DoubleSoftMax_updateOutput(context->thc_state, self_->tensor, output_->tensor, dim);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUDoubleType::softmax_forward(const Tensor & self, int64_t dim) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_DoubleSoftMax_updateOutput(context->thc_state, self_->tensor, output_->tensor, dim);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUDoubleType::softmax_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",4, false);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_DoubleSoftMax_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, output_->tensor, dim);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::softmax_backward(const Tensor & grad_output, const Tensor & self, int64_t dim, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    dim = maybe_wrap_dim(dim, self_);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",4, false);
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleSoftMax_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, output_->tensor, dim);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUDoubleType::softplus_forward_out(Tensor & output, const Tensor & self, Scalar beta, Scalar threshold) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toDouble();
    auto threshold_ = threshold.toDouble();
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",3, false);
    THNN_DoubleSoftPlus_updateOutput(context->thc_state, self_->tensor, output_->tensor, beta_, threshold_);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUDoubleType::softplus_forward(const Tensor & self, Scalar beta, Scalar threshold) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto beta_ = beta.toDouble();
    auto threshold_ = threshold.toDouble();
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_DoubleSoftPlus_updateOutput(context->thc_state, self_->tensor, output_->tensor, beta_, threshold_);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUDoubleType::softplus_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto beta_ = beta.toDouble();
    auto threshold_ = threshold.toDouble();
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",5, false);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",5, false);
    THNN_DoubleSoftPlus_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, output_->tensor, beta_, threshold_);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::softplus_backward(const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto beta_ = beta.toDouble();
    auto threshold_ = threshold.toDouble();
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",5, false);
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleSoftPlus_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, output_->tensor, beta_, threshold_);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUDoubleType::softshrink_forward_out(Tensor & output, const Tensor & self, Scalar lambd) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto lambd_ = lambd.toDouble();
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",2, false);
    THNN_DoubleSoftShrink_updateOutput(context->thc_state, self_->tensor, output_->tensor, lambd_);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUDoubleType::softshrink_forward(const Tensor & self, Scalar lambd) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto lambd_ = lambd.toDouble();
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_DoubleSoftShrink_updateOutput(context->thc_state, self_->tensor, output_->tensor, lambd_);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUDoubleType::softshrink_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar lambd) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto lambd_ = lambd.toDouble();
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_DoubleSoftShrink_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, lambd_);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::softshrink_backward(const Tensor & grad_output, const Tensor & self, Scalar lambd) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto lambd_ = lambd.toDouble();
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleSoftShrink_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, lambd_);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUDoubleType::threshold_forward_out(Tensor & output, const Tensor & self, Scalar threshold, Scalar value) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto threshold_ = threshold.toDouble();
    auto value_ = value.toDouble();
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",3, false);
    THNN_DoubleThreshold_updateOutput(context->thc_state, self_->tensor, output_->tensor, threshold_, value_, false);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUDoubleType::threshold_forward(const Tensor & self, Scalar threshold, Scalar value) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto threshold_ = threshold.toDouble();
    auto value_ = value.toDouble();
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_DoubleThreshold_updateOutput(context->thc_state, self_->tensor, output_->tensor, threshold_, value_, false);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUDoubleType::threshold_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar threshold, Scalar value) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto threshold_ = threshold.toDouble();
    auto value_ = value.toDouble();
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_DoubleThreshold_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, threshold_, value_, false);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::threshold_backward(const Tensor & grad_output, const Tensor & self, Scalar threshold, Scalar value) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto threshold_ = threshold.toDouble();
    auto value_ = value.toDouble();
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleThreshold_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, threshold_, value_, false);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUDoubleType::threshold_forward_(Tensor & self, Scalar threshold, Scalar value) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto threshold_ = threshold.toDouble();
    auto value_ = value.toDouble();
    THNN_DoubleThreshold_updateOutput(context->thc_state, self_->tensor, self_->tensor, threshold_, value_, true);
    return self;
}
Tensor & CPUDoubleType::adaptive_avg_pool2d_forward_out(Tensor & output, const Tensor & self, IntList output_size) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 2);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",2, false);
    THNN_DoubleSpatialAdaptiveAveragePooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[1], output_size_[0]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUDoubleType::adaptive_avg_pool2d_forward(const Tensor & self, IntList output_size) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 2);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_DoubleSpatialAdaptiveAveragePooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[1], output_size_[0]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUDoubleType::adaptive_avg_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",2, false);
    THNN_DoubleSpatialAdaptiveAveragePooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::adaptive_avg_pool2d_backward(const Tensor & grad_output, const Tensor & self) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleSpatialAdaptiveAveragePooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUDoubleType::adaptive_avg_pool3d_forward_out(Tensor & output, const Tensor & self, IntList output_size) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 2);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",2, false);
    THNN_DoubleVolumetricAdaptiveAveragePooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[0], output_size_[2], output_size_[1]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUDoubleType::adaptive_avg_pool3d_forward(const Tensor & self, IntList output_size) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 2);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_DoubleVolumetricAdaptiveAveragePooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[0], output_size_[2], output_size_[1]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUDoubleType::adaptive_avg_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",2, false);
    THNN_DoubleVolumetricAdaptiveAveragePooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::adaptive_avg_pool3d_backward(const Tensor & grad_output, const Tensor & self) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleVolumetricAdaptiveAveragePooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &> CPUDoubleType::adaptive_max_pool2d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList output_size) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 2);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",2, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",2, false);
    THNN_DoubleSpatialAdaptiveMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, output_size_[1], output_size_[0]);
    bool maybe_scalar = self_->isScalar();
    output_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(output, indices);
}
std::tuple<Tensor,Tensor> CPUDoubleType::adaptive_max_pool2d_forward(const Tensor & self, IntList output_size) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 2);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    auto indices_ = new CPULongTensor(context);
    auto indices = Tensor(indices_, false);
    THNN_DoubleSpatialAdaptiveMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, output_size_[1], output_size_[0]);
    bool maybe_scalar = self_->isScalar();
    output_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(output, indices);
}
Tensor & CPUDoubleType::adaptive_max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",3, false);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_DoubleSpatialAdaptiveMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::adaptive_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",3, false);
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleSpatialAdaptiveMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &> CPUDoubleType::adaptive_max_pool3d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList output_size) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 2);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",2, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",2, false);
    THNN_DoubleVolumetricAdaptiveMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, output_size_[0], output_size_[2], output_size_[1]);
    bool maybe_scalar = self_->isScalar();
    output_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(output, indices);
}
std::tuple<Tensor,Tensor> CPUDoubleType::adaptive_max_pool3d_forward(const Tensor & self, IntList output_size) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 2);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    auto indices_ = new CPULongTensor(context);
    auto indices = Tensor(indices_, false);
    THNN_DoubleVolumetricAdaptiveMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, output_size_[0], output_size_[2], output_size_[1]);
    bool maybe_scalar = self_->isScalar();
    output_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(output, indices);
}
Tensor & CPUDoubleType::adaptive_max_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",3, false);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_DoubleVolumetricAdaptiveMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::adaptive_max_pool3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",3, false);
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleVolumetricAdaptiveMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUDoubleType::avg_pool2d_forward_out(Tensor & output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 2);
    auto stride_ = check_intlist<2>(stride, "stride", 3, kernel_size);
    auto padding_ = check_intlist<2>(padding, "padding", 4);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",6, false);
    THNN_DoubleSpatialAveragePooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], ceil_mode, count_include_pad);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUDoubleType::avg_pool2d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 2);
    auto stride_ = check_intlist<2>(stride, "stride", 3, kernel_size);
    auto padding_ = check_intlist<2>(padding, "padding", 4);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_DoubleSpatialAveragePooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], ceil_mode, count_include_pad);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUDoubleType::avg_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto stride_ = check_intlist<2>(stride, "stride", 4, kernel_size);
    auto padding_ = check_intlist<2>(padding, "padding", 5);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",7, false);
    THNN_DoubleSpatialAveragePooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], ceil_mode, count_include_pad);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::avg_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto stride_ = check_intlist<2>(stride, "stride", 4, kernel_size);
    auto padding_ = check_intlist<2>(padding, "padding", 5);
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleSpatialAveragePooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], ceil_mode, count_include_pad);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUDoubleType::avg_pool3d_forward_out(Tensor & output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 2);
    auto stride_ = check_intlist<3>(stride, "stride", 3, kernel_size);
    auto padding_ = check_intlist<3>(padding, "padding", 4);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",6, false);
    THNN_DoubleVolumetricAveragePooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], ceil_mode, count_include_pad);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUDoubleType::avg_pool3d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 2);
    auto stride_ = check_intlist<3>(stride, "stride", 3, kernel_size);
    auto padding_ = check_intlist<3>(padding, "padding", 4);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_DoubleVolumetricAveragePooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], ceil_mode, count_include_pad);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUDoubleType::avg_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto stride_ = check_intlist<3>(stride, "stride", 4, kernel_size);
    auto padding_ = check_intlist<3>(padding, "padding", 5);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",7, false);
    THNN_DoubleVolumetricAveragePooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], ceil_mode, count_include_pad);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::avg_pool3d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto stride_ = check_intlist<3>(stride, "stride", 4, kernel_size);
    auto padding_ = check_intlist<3>(padding, "padding", 5);
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleVolumetricAveragePooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], ceil_mode, count_include_pad);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &> CPUDoubleType::fractional_max_pool2d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & random_samples) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 2);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 3);
    auto random_samples_ = checked_cast_tensor<CPUDoubleTensor>(random_samples.pImpl,"random_samples",4, false);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",4, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",4, false);
    THNN_DoubleSpatialFractionalMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[1], output_size_[0], kernel_size_[1], kernel_size_[0], indices_->tensor, random_samples_->tensor);
    output_->maybeScalar(false);
    return std::tuple<Tensor &, Tensor &>(output, indices);
}
std::tuple<Tensor,Tensor> CPUDoubleType::fractional_max_pool2d_forward(const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & random_samples) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 2);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 3);
    auto random_samples_ = checked_cast_tensor<CPUDoubleTensor>(random_samples.pImpl,"random_samples",4, false);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    auto indices_ = new CPULongTensor(context);
    auto indices = Tensor(indices_, false);
    THNN_DoubleSpatialFractionalMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[1], output_size_[0], kernel_size_[1], kernel_size_[0], indices_->tensor, random_samples_->tensor);
    output_->maybeScalar(false);
    return std::tuple<Tensor, Tensor>(output, indices);
}
Tensor & CPUDoubleType::fractional_max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 4);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",5, false);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",5, false);
    THNN_DoubleSpatialFractionalMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, output_size_[1], output_size_[0], kernel_size_[1], kernel_size_[0], indices_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::fractional_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 4);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",5, false);
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleSpatialFractionalMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, output_size_[1], output_size_[0], kernel_size_[1], kernel_size_[0], indices_->tensor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &> CPUDoubleType::max_pool2d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 2);
    auto stride_ = check_intlist<2>(stride, "stride", 3, kernel_size);
    auto padding_ = check_intlist<2>(padding, "padding", 4);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 5);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",6, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",6, false);
    THNN_DoubleSpatialDilatedMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], ceil_mode);
    bool maybe_scalar = self_->isScalar();
    output_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(output, indices);
}
std::tuple<Tensor,Tensor> CPUDoubleType::max_pool2d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 2);
    auto stride_ = check_intlist<2>(stride, "stride", 3, kernel_size);
    auto padding_ = check_intlist<2>(padding, "padding", 4);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 5);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    auto indices_ = new CPULongTensor(context);
    auto indices = Tensor(indices_, false);
    THNN_DoubleSpatialDilatedMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], ceil_mode);
    bool maybe_scalar = self_->isScalar();
    output_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(output, indices);
}
Tensor & CPUDoubleType::max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto stride_ = check_intlist<2>(stride, "stride", 4, kernel_size);
    auto padding_ = check_intlist<2>(padding, "padding", 5);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 6);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",8, false);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",8, false);
    THNN_DoubleSpatialDilatedMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], ceil_mode);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::max_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto stride_ = check_intlist<2>(stride, "stride", 4, kernel_size);
    auto padding_ = check_intlist<2>(padding, "padding", 5);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 6);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",8, false);
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleSpatialDilatedMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], ceil_mode);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &> CPUDoubleType::max_pool3d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 2);
    auto stride_ = check_intlist<3>(stride, "stride", 3, kernel_size);
    auto padding_ = check_intlist<3>(padding, "padding", 4);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 5);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",6, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",6, false);
    THNN_DoubleVolumetricDilatedMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], ceil_mode);
    bool maybe_scalar = self_->isScalar();
    output_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &>(output, indices);
}
std::tuple<Tensor,Tensor> CPUDoubleType::max_pool3d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 2);
    auto stride_ = check_intlist<3>(stride, "stride", 3, kernel_size);
    auto padding_ = check_intlist<3>(padding, "padding", 4);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 5);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    auto indices_ = new CPULongTensor(context);
    auto indices = Tensor(indices_, false);
    THNN_DoubleVolumetricDilatedMaxPooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], ceil_mode);
    bool maybe_scalar = self_->isScalar();
    output_->maybeScalar(maybe_scalar);
    indices_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor>(output, indices);
}
Tensor & CPUDoubleType::max_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto stride_ = check_intlist<3>(stride, "stride", 4, kernel_size);
    auto padding_ = check_intlist<3>(padding, "padding", 5);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 6);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",8, false);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",8, false);
    THNN_DoubleVolumetricDilatedMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], ceil_mode);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::max_pool3d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto stride_ = check_intlist<3>(stride, "stride", 4, kernel_size);
    auto padding_ = check_intlist<3>(padding, "padding", 5);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 6);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",8, false);
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleVolumetricDilatedMaxPooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], ceil_mode);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUDoubleType::max_unpool2d_forward_out(Tensor & output, const Tensor & self, const Tensor & indices, IntList output_size) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",2, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 3);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",3, false);
    THNN_DoubleSpatialMaxUnpooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, output_size_[1], output_size_[0]);
    output_->maybeScalar(self_->isScalar() && indices_->isScalar());
    return output;
}
Tensor CPUDoubleType::max_unpool2d_forward(const Tensor & self, const Tensor & indices, IntList output_size) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",2, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 3);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_DoubleSpatialMaxUnpooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, output_size_[1], output_size_[0]);
    output_->maybeScalar(self_->isScalar() && indices_->isScalar());
    return output;
}
Tensor & CPUDoubleType::max_unpool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",3, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 4);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_DoubleSpatialMaxUnpooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor, output_size_[1], output_size_[0]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::max_unpool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",3, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 4);
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleSpatialMaxUnpooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor, output_size_[1], output_size_[0]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUDoubleType::max_unpool3d_forward_out(Tensor & output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",2, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 3);
    auto stride_ = check_intlist<3>(stride, "stride", 4);
    auto padding_ = check_intlist<3>(padding, "padding", 5);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",5, false);
    THNN_DoubleVolumetricMaxUnpooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, output_size_[0], output_size_[2], output_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1]);
    output_->maybeScalar(self_->isScalar() && indices_->isScalar());
    return output;
}
Tensor CPUDoubleType::max_unpool3d_forward(const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",2, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 3);
    auto stride_ = check_intlist<3>(stride, "stride", 4);
    auto padding_ = check_intlist<3>(padding, "padding", 5);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_DoubleVolumetricMaxUnpooling_updateOutput(context->thc_state, self_->tensor, output_->tensor, indices_->tensor, output_size_[0], output_size_[2], output_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1]);
    output_->maybeScalar(self_->isScalar() && indices_->isScalar());
    return output;
}
Tensor & CPUDoubleType::max_unpool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",3, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 4);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",6, false);
    THNN_DoubleVolumetricMaxUnpooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor, output_size_[0], output_size_[2], output_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::max_unpool3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto indices_ = checked_cast_tensor<CPULongTensor>(indices.pImpl,"indices",3, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 4);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleVolumetricMaxUnpooling_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, indices_->tensor, output_size_[0], output_size_[2], output_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUDoubleType::reflection_pad1d_forward_out(Tensor & output, const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<2>(padding, "padding", 2);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",2, false);
    THNN_DoubleTemporalReflectionPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUDoubleType::reflection_pad1d_forward(const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<2>(padding, "padding", 2);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_DoubleTemporalReflectionPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUDoubleType::reflection_pad1d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<2>(padding, "padding", 3);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_DoubleTemporalReflectionPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::reflection_pad1d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<2>(padding, "padding", 3);
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleTemporalReflectionPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUDoubleType::reflection_pad2d_forward_out(Tensor & output, const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<4>(padding, "padding", 2);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",2, false);
    THNN_DoubleSpatialReflectionPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1], padding_[2], padding_[3]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUDoubleType::reflection_pad2d_forward(const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<4>(padding, "padding", 2);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_DoubleSpatialReflectionPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1], padding_[2], padding_[3]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUDoubleType::reflection_pad2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<4>(padding, "padding", 3);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_DoubleSpatialReflectionPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1], padding_[2], padding_[3]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::reflection_pad2d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<4>(padding, "padding", 3);
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleSpatialReflectionPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1], padding_[2], padding_[3]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUDoubleType::replication_pad1d_forward_out(Tensor & output, const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<2>(padding, "padding", 2);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",2, false);
    THNN_DoubleTemporalReplicationPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUDoubleType::replication_pad1d_forward(const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<2>(padding, "padding", 2);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_DoubleTemporalReplicationPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUDoubleType::replication_pad1d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<2>(padding, "padding", 3);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_DoubleTemporalReplicationPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::replication_pad1d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<2>(padding, "padding", 3);
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleTemporalReplicationPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUDoubleType::replication_pad2d_forward_out(Tensor & output, const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<4>(padding, "padding", 2);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",2, false);
    THNN_DoubleSpatialReplicationPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1], padding_[2], padding_[3]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUDoubleType::replication_pad2d_forward(const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<4>(padding, "padding", 2);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_DoubleSpatialReplicationPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1], padding_[2], padding_[3]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUDoubleType::replication_pad2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<4>(padding, "padding", 3);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_DoubleSpatialReplicationPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1], padding_[2], padding_[3]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::replication_pad2d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<4>(padding, "padding", 3);
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleSpatialReplicationPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1], padding_[2], padding_[3]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUDoubleType::replication_pad3d_forward_out(Tensor & output, const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<6>(padding, "padding", 2);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",2, false);
    THNN_DoubleVolumetricReplicationPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1], padding_[2], padding_[3], padding_[4], padding_[5]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUDoubleType::replication_pad3d_forward(const Tensor & self, IntList padding) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto padding_ = check_intlist<6>(padding, "padding", 2);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_DoubleVolumetricReplicationPadding_updateOutput(context->thc_state, self_->tensor, output_->tensor, padding_[0], padding_[1], padding_[2], padding_[3], padding_[4], padding_[5]);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUDoubleType::replication_pad3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<6>(padding, "padding", 3);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_DoubleVolumetricReplicationPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1], padding_[2], padding_[3], padding_[4], padding_[5]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::replication_pad3d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto padding_ = check_intlist<6>(padding, "padding", 3);
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleVolumetricReplicationPadding_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, padding_[0], padding_[1], padding_[2], padding_[3], padding_[4], padding_[5]);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUDoubleType::upsample_linear1d_forward_out(Tensor & output, const Tensor & self, IntList output_size, bool align_corners) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<1>(output_size, "output_size", 2);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",3, false);
    THNN_DoubleTemporalUpSamplingLinear_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[0], align_corners);
    return output;
}
Tensor CPUDoubleType::upsample_linear1d_forward(const Tensor & self, IntList output_size, bool align_corners) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<1>(output_size, "output_size", 2);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_DoubleTemporalUpSamplingLinear_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[0], align_corners);
    return output;
}
Tensor & CPUDoubleType::upsample_linear1d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_size_ = check_intlist<1>(output_size, "output_size", 2);
    auto input_size_ = check_intlist<3>(input_size, "input_size", 3);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_DoubleTemporalUpSamplingLinear_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, input_size_[0], input_size_[1], input_size_[2], output_size_[0], align_corners);
    grad_input_->maybeScalar(false);
    return grad_input;
}
Tensor CPUDoubleType::upsample_linear1d_backward(const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_size_ = check_intlist<1>(output_size, "output_size", 2);
    auto input_size_ = check_intlist<3>(input_size, "input_size", 3);
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleTemporalUpSamplingLinear_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, input_size_[0], input_size_[1], input_size_[2], output_size_[0], align_corners);
    grad_input_->maybeScalar(false);
    return grad_input;
}
Tensor & CPUDoubleType::upsample_bilinear2d_forward_out(Tensor & output, const Tensor & self, IntList output_size, bool align_corners) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 2);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",3, false);
    THNN_DoubleSpatialUpSamplingBilinear_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[0], output_size_[1], align_corners);
    return output;
}
Tensor CPUDoubleType::upsample_bilinear2d_forward(const Tensor & self, IntList output_size, bool align_corners) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 2);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_DoubleSpatialUpSamplingBilinear_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[0], output_size_[1], align_corners);
    return output;
}
Tensor & CPUDoubleType::upsample_bilinear2d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 2);
    auto input_size_ = check_intlist<4>(input_size, "input_size", 3);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_DoubleSpatialUpSamplingBilinear_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, input_size_[0], input_size_[1], input_size_[2], input_size_[3], output_size_[0], output_size_[1], align_corners);
    grad_input_->maybeScalar(false);
    return grad_input;
}
Tensor CPUDoubleType::upsample_bilinear2d_backward(const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_size_ = check_intlist<2>(output_size, "output_size", 2);
    auto input_size_ = check_intlist<4>(input_size, "input_size", 3);
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleSpatialUpSamplingBilinear_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, input_size_[0], input_size_[1], input_size_[2], input_size_[3], output_size_[0], output_size_[1], align_corners);
    grad_input_->maybeScalar(false);
    return grad_input;
}
Tensor & CPUDoubleType::upsample_trilinear3d_forward_out(Tensor & output, const Tensor & self, IntList output_size, bool align_corners) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 2);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",3, false);
    THNN_DoubleVolumetricUpSamplingTrilinear_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[0], output_size_[1], output_size_[2], align_corners);
    return output;
}
Tensor CPUDoubleType::upsample_trilinear3d_forward(const Tensor & self, IntList output_size, bool align_corners) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 2);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_DoubleVolumetricUpSamplingTrilinear_updateOutput(context->thc_state, self_->tensor, output_->tensor, output_size_[0], output_size_[1], output_size_[2], align_corners);
    return output;
}
Tensor & CPUDoubleType::upsample_trilinear3d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 2);
    auto input_size_ = check_intlist<5>(input_size, "input_size", 3);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",4, false);
    THNN_DoubleVolumetricUpSamplingTrilinear_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, input_size_[0], input_size_[1], input_size_[2], input_size_[3], input_size_[4], output_size_[0], output_size_[1], output_size_[2], align_corners);
    grad_input_->maybeScalar(false);
    return grad_input;
}
Tensor CPUDoubleType::upsample_trilinear3d_backward(const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_size_ = check_intlist<3>(output_size, "output_size", 2);
    auto input_size_ = check_intlist<5>(input_size, "input_size", 3);
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleVolumetricUpSamplingTrilinear_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, input_size_[0], input_size_[1], input_size_[2], input_size_[3], input_size_[4], output_size_[0], output_size_[1], output_size_[2], align_corners);
    grad_input_->maybeScalar(false);
    return grad_input;
}
Tensor & CPUDoubleType::upsample_nearest1d_forward_out(Tensor & output, const Tensor & self, int64_t scale_factor) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",2, false);
    THNN_DoubleTemporalUpSamplingNearest_updateOutput(context->thc_state, self_->tensor, output_->tensor, scale_factor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUDoubleType::upsample_nearest1d_forward(const Tensor & self, int64_t scale_factor) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_DoubleTemporalUpSamplingNearest_updateOutput(context->thc_state, self_->tensor, output_->tensor, scale_factor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUDoubleType::upsample_nearest1d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_DoubleTemporalUpSamplingNearest_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, scale_factor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::upsample_nearest1d_backward(const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleTemporalUpSamplingNearest_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, scale_factor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUDoubleType::upsample_nearest2d_forward_out(Tensor & output, const Tensor & self, int64_t scale_factor) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",2, false);
    THNN_DoubleSpatialUpSamplingNearest_updateOutput(context->thc_state, self_->tensor, output_->tensor, scale_factor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUDoubleType::upsample_nearest2d_forward(const Tensor & self, int64_t scale_factor) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_DoubleSpatialUpSamplingNearest_updateOutput(context->thc_state, self_->tensor, output_->tensor, scale_factor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUDoubleType::upsample_nearest2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_DoubleSpatialUpSamplingNearest_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, scale_factor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::upsample_nearest2d_backward(const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleSpatialUpSamplingNearest_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, scale_factor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUDoubleType::upsample_nearest3d_forward_out(Tensor & output, const Tensor & self, int64_t scale_factor) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",2, false);
    THNN_DoubleVolumetricUpSamplingNearest_updateOutput(context->thc_state, self_->tensor, output_->tensor, scale_factor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUDoubleType::upsample_nearest3d_forward(const Tensor & self, int64_t scale_factor) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_DoubleVolumetricUpSamplingNearest_updateOutput(context->thc_state, self_->tensor, output_->tensor, scale_factor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUDoubleType::upsample_nearest3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",3, false);
    THNN_DoubleVolumetricUpSamplingNearest_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, scale_factor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::upsample_nearest3d_backward(const Tensor & grad_output, const Tensor & self, int64_t scale_factor) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleVolumetricUpSamplingNearest_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_->tensor, scale_factor);
    grad_input_->maybeScalar(self_->isScalar());
    return grad_input;
}
Tensor & CPUDoubleType::_sigmoid_forward_out(Tensor & output, const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",1, false);
    THNN_DoubleSigmoid_updateOutput(context->thc_state, self_->tensor, output_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUDoubleType::_sigmoid_forward(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_DoubleSigmoid_updateOutput(context->thc_state, self_->tensor, output_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUDoubleType::_sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",2, false);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",2, false);
    THNN_DoubleSigmoid_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, output_->tensor);
    grad_input_->maybeScalar(output_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::_sigmoid_backward(const Tensor & grad_output, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",2, false);
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleSigmoid_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, output_->tensor);
    grad_input_->maybeScalar(output_->isScalar());
    return grad_input;
}
Tensor & CPUDoubleType::_tanh_forward_out(Tensor & output, const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",1, false);
    THNN_DoubleTanh_updateOutput(context->thc_state, self_->tensor, output_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor CPUDoubleType::_tanh_forward(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    THNN_DoubleTanh_updateOutput(context->thc_state, self_->tensor, output_->tensor);
    output_->maybeScalar(self_->isScalar());
    return output;
}
Tensor & CPUDoubleType::_tanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",2, false);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",2, false);
    THNN_DoubleTanh_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, output_->tensor);
    grad_input_->maybeScalar(output_->isScalar());
    return grad_input;
}
Tensor CPUDoubleType::_tanh_backward(const Tensor & grad_output, const Tensor & output) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",2, false);
    auto grad_input_ = new CPUDoubleTensor(context);
    auto grad_input = Tensor(grad_input_, false);
    THNN_DoubleTanh_updateGradInput(context->thc_state, grad_output_->tensor, grad_input_->tensor, output_->tensor);
    grad_input_->maybeScalar(output_->isScalar());
    return grad_input;
}
std::tuple<Tensor &,Tensor &,Tensor &> CPUDoubleType::thnn_batch_norm_forward_out(Tensor & output, Tensor & save_mean, Tensor & save_std, const Tensor & self, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",2, true);
    auto bias_ = checked_cast_tensor<CPUDoubleTensor>(bias.pImpl,"bias",3, true);
    auto running_mean_ = checked_cast_tensor<CPUDoubleTensor>(running_mean.pImpl,"running_mean",4, true);
    auto running_var_ = checked_cast_tensor<CPUDoubleTensor>(running_var.pImpl,"running_var",5, true);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",8, false);
    auto save_mean_ = checked_cast_tensor<CPUDoubleTensor>(save_mean.pImpl,"save_mean",8, false);
    auto save_std_ = checked_cast_tensor<CPUDoubleTensor>(save_std.pImpl,"save_std",8, false);
    THNN_DoubleBatchNormalization_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_ ? weight_->tensor : NULL, bias_ ? bias_->tensor : NULL, running_mean_ ? running_mean_->tensor : NULL, running_var_ ? running_var_->tensor : NULL, save_mean_->tensor, save_std_->tensor, training, momentum, eps);
    bool maybe_scalar = self_->isScalar() && (!weight_ || weight_->isScalar()) && (!bias_ || bias_->isScalar()) && (!running_mean_ || running_mean_->isScalar()) && (!running_var_ || running_var_->isScalar());
    output_->maybeScalar(maybe_scalar);
    save_mean_->maybeScalar(maybe_scalar);
    save_std_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(output, save_mean, save_std);
}
std::tuple<Tensor,Tensor,Tensor> CPUDoubleType::thnn_batch_norm_forward(const Tensor & self, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",2, true);
    auto bias_ = checked_cast_tensor<CPUDoubleTensor>(bias.pImpl,"bias",3, true);
    auto running_mean_ = checked_cast_tensor<CPUDoubleTensor>(running_mean.pImpl,"running_mean",4, true);
    auto running_var_ = checked_cast_tensor<CPUDoubleTensor>(running_var.pImpl,"running_var",5, true);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    auto save_mean_ = new CPUDoubleTensor(context);
    auto save_mean = Tensor(save_mean_, false);
    auto save_std_ = new CPUDoubleTensor(context);
    auto save_std = Tensor(save_std_, false);
    THNN_DoubleBatchNormalization_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_ ? weight_->tensor : NULL, bias_ ? bias_->tensor : NULL, running_mean_ ? running_mean_->tensor : NULL, running_var_ ? running_var_->tensor : NULL, save_mean_->tensor, save_std_->tensor, training, momentum, eps);
    bool maybe_scalar = self_->isScalar() && (!weight_ || weight_->isScalar()) && (!bias_ || bias_->isScalar()) && (!running_mean_ || running_mean_->isScalar()) && (!running_var_ || running_var_->isScalar());
    output_->maybeScalar(maybe_scalar);
    save_mean_->maybeScalar(maybe_scalar);
    save_std_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(output, save_mean, save_std);
}
std::tuple<Tensor &,Tensor &,Tensor &> CPUDoubleType::thnn_batch_norm_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, bool training, double eps, const Tensor & save_mean, const Tensor & save_std) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",3, true);
    auto running_mean_ = checked_cast_tensor<CPUDoubleTensor>(running_mean.pImpl,"running_mean",4, true);
    auto running_var_ = checked_cast_tensor<CPUDoubleTensor>(running_var.pImpl,"running_var",5, true);
    auto save_mean_ = checked_cast_tensor<CPUDoubleTensor>(save_mean.pImpl,"save_mean",8, true);
    auto save_std_ = checked_cast_tensor<CPUDoubleTensor>(save_std.pImpl,"save_std",9, true);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",9, true);
    auto grad_weight_ = checked_cast_tensor<CPUDoubleTensor>(grad_weight.pImpl,"grad_weight",9, true);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = checked_cast_tensor<CPUDoubleTensor>(grad_bias.pImpl,"grad_bias",9, true);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    THNN_DoubleBatchNormalization_backward(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, weight_ ? weight_->tensor : NULL, running_mean_ ? running_mean_->tensor : NULL, running_var_ ? running_var_->tensor : NULL, save_mean_ ? save_mean_->tensor : NULL, save_std_ ? save_std_->tensor : NULL, training, 1, eps);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &, Tensor &>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> CPUDoubleType::thnn_batch_norm_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, bool training, double eps, const Tensor & save_mean, const Tensor & save_std, std::array<bool,3> output_mask) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",3, true);
    auto running_mean_ = checked_cast_tensor<CPUDoubleTensor>(running_mean.pImpl,"running_mean",4, true);
    auto running_var_ = checked_cast_tensor<CPUDoubleTensor>(running_var.pImpl,"running_var",5, true);
    auto save_mean_ = checked_cast_tensor<CPUDoubleTensor>(save_mean.pImpl,"save_mean",8, true);
    auto save_std_ = checked_cast_tensor<CPUDoubleTensor>(save_std.pImpl,"save_std",9, true);
    auto grad_input_ = output_mask[0] ? new CPUDoubleTensor(context) : nullptr;
    auto grad_input = Tensor(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_input_, false);
    auto grad_weight_ = output_mask[1] ? new CPUDoubleTensor(context) : nullptr;
    auto grad_weight = Tensor(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_weight_, false);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = output_mask[2] ? new CPUDoubleTensor(context) : nullptr;
    auto grad_bias = Tensor(grad_bias_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_bias_, false);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    THNN_DoubleBatchNormalization_backward(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, weight_ ? weight_->tensor : NULL, running_mean_ ? running_mean_->tensor : NULL, running_var_ ? running_var_->tensor : NULL, save_mean_ ? save_mean_->tensor : NULL, save_std_ ? save_std_->tensor : NULL, training, 1, eps);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor &,Tensor &,Tensor &> CPUDoubleType::thnn_conv_transpose2d_forward_out(Tensor & output, Tensor & columns, Tensor & ones, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CPUDoubleTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto output_padding_ = check_intlist<2>(output_padding, "output_padding", 7);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 8);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",8, false);
    auto columns_ = checked_cast_tensor<CPUDoubleTensor>(columns.pImpl,"columns",8, false);
    auto ones_ = checked_cast_tensor<CPUDoubleTensor>(ones.pImpl,"ones",8, false);
    THNN_DoubleSpatialFullDilatedConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], output_padding_[1], output_padding_[0]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    columns_->maybeScalar(maybe_scalar);
    ones_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(output, columns, ones);
}
std::tuple<Tensor,Tensor,Tensor> CPUDoubleType::thnn_conv_transpose2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CPUDoubleTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto output_padding_ = check_intlist<2>(output_padding, "output_padding", 7);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 8);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    auto columns_ = new CPUDoubleTensor(context);
    auto columns = Tensor(columns_, false);
    auto ones_ = new CPUDoubleTensor(context);
    auto ones = Tensor(ones_, false);
    THNN_DoubleSpatialFullDilatedConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], output_padding_[1], output_padding_[0]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    columns_->maybeScalar(maybe_scalar);
    ones_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(output, columns, ones);
}
std::tuple<Tensor &,Tensor &,Tensor &> CPUDoubleType::thnn_conv_transpose2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & columns, const Tensor & ones) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto output_padding_ = check_intlist<2>(output_padding, "output_padding", 7);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 8);
    auto columns_ = checked_cast_tensor<CPUDoubleTensor>(columns.pImpl,"columns",9, false);
    auto ones_ = checked_cast_tensor<CPUDoubleTensor>(ones.pImpl,"ones",10, false);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",10, true);
    auto grad_weight_ = checked_cast_tensor<CPUDoubleTensor>(grad_weight.pImpl,"grad_weight",10, true);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = checked_cast_tensor<CPUDoubleTensor>(grad_bias.pImpl,"grad_bias",10, true);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(1) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_DoubleSpatialFullDilatedConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, columns_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], output_padding_[1], output_padding_[0]);
    if (grad_weight_ || grad_bias_) THNN_DoubleSpatialFullDilatedConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], output_padding_[1], output_padding_[0], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &, Tensor &>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> CPUDoubleType::thnn_conv_transpose2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto output_padding_ = check_intlist<2>(output_padding, "output_padding", 7);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 8);
    auto columns_ = checked_cast_tensor<CPUDoubleTensor>(columns.pImpl,"columns",9, false);
    auto ones_ = checked_cast_tensor<CPUDoubleTensor>(ones.pImpl,"ones",10, false);
    auto grad_input_ = output_mask[0] ? new CPUDoubleTensor(context) : nullptr;
    auto grad_input = Tensor(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_input_, false);
    auto grad_weight_ = output_mask[1] ? new CPUDoubleTensor(context) : nullptr;
    auto grad_weight = Tensor(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_weight_, false);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = output_mask[2] ? new CPUDoubleTensor(context) : nullptr;
    auto grad_bias = Tensor(grad_bias_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_bias_, false);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(1) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_DoubleSpatialFullDilatedConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, columns_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], output_padding_[1], output_padding_[0]);
    if (grad_weight_ || grad_bias_) THNN_DoubleSpatialFullDilatedConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], output_padding_[1], output_padding_[0], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor &,Tensor &,Tensor &> CPUDoubleType::thnn_conv_transpose3d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CPUDoubleTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto output_padding_ = check_intlist<3>(output_padding, "output_padding", 7);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 8);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",8, false);
    auto finput_ = checked_cast_tensor<CPUDoubleTensor>(finput.pImpl,"finput",8, false);
    auto fgrad_input_ = checked_cast_tensor<CPUDoubleTensor>(fgrad_input.pImpl,"fgrad_input",8, false);
    THNN_DoubleVolumetricFullDilatedConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], output_padding_[0], output_padding_[2], output_padding_[1]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    finput_->maybeScalar(maybe_scalar);
    fgrad_input_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(output, finput, fgrad_input);
}
std::tuple<Tensor,Tensor,Tensor> CPUDoubleType::thnn_conv_transpose3d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CPUDoubleTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto output_padding_ = check_intlist<3>(output_padding, "output_padding", 7);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 8);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    auto finput_ = new CPUDoubleTensor(context);
    auto finput = Tensor(finput_, false);
    auto fgrad_input_ = new CPUDoubleTensor(context);
    auto fgrad_input = Tensor(fgrad_input_, false);
    THNN_DoubleVolumetricFullDilatedConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], output_padding_[0], output_padding_[2], output_padding_[1]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    finput_->maybeScalar(maybe_scalar);
    fgrad_input_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(output, finput, fgrad_input);
}
std::tuple<Tensor &,Tensor &,Tensor &> CPUDoubleType::thnn_conv_transpose3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & finput, const Tensor & fgrad_input) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto output_padding_ = check_intlist<3>(output_padding, "output_padding", 7);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 8);
    auto finput_ = checked_cast_tensor<CPUDoubleTensor>(finput.pImpl,"finput",9, false);
    auto fgrad_input_ = checked_cast_tensor<CPUDoubleTensor>(fgrad_input.pImpl,"fgrad_input",10, false);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",10, true);
    auto grad_weight_ = checked_cast_tensor<CPUDoubleTensor>(grad_weight.pImpl,"grad_weight",10, true);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = checked_cast_tensor<CPUDoubleTensor>(grad_bias.pImpl,"grad_bias",10, true);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(1) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_DoubleVolumetricFullDilatedConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], output_padding_[0], output_padding_[2], output_padding_[1]);
    if (grad_weight_ || grad_bias_) THNN_DoubleVolumetricFullDilatedConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], output_padding_[0], output_padding_[2], output_padding_[1], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &, Tensor &>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> CPUDoubleType::thnn_conv_transpose3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto output_padding_ = check_intlist<3>(output_padding, "output_padding", 7);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 8);
    auto finput_ = checked_cast_tensor<CPUDoubleTensor>(finput.pImpl,"finput",9, false);
    auto fgrad_input_ = checked_cast_tensor<CPUDoubleTensor>(fgrad_input.pImpl,"fgrad_input",10, false);
    auto grad_input_ = output_mask[0] ? new CPUDoubleTensor(context) : nullptr;
    auto grad_input = Tensor(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_input_, false);
    auto grad_weight_ = output_mask[1] ? new CPUDoubleTensor(context) : nullptr;
    auto grad_weight = Tensor(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_weight_, false);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = output_mask[2] ? new CPUDoubleTensor(context) : nullptr;
    auto grad_bias = Tensor(grad_bias_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_bias_, false);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(1) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_DoubleVolumetricFullDilatedConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], output_padding_[0], output_padding_[2], output_padding_[1]);
    if (grad_weight_ || grad_bias_) THNN_DoubleVolumetricFullDilatedConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], output_padding_[0], output_padding_[2], output_padding_[1], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor &,Tensor &,Tensor &> CPUDoubleType::thnn_conv2d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CPUDoubleTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",6, false);
    auto finput_ = checked_cast_tensor<CPUDoubleTensor>(finput.pImpl,"finput",6, false);
    auto fgrad_input_ = checked_cast_tensor<CPUDoubleTensor>(fgrad_input.pImpl,"fgrad_input",6, false);
    THNN_DoubleSpatialConvolutionMM_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    finput_->maybeScalar(maybe_scalar);
    fgrad_input_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(output, finput, fgrad_input);
}
std::tuple<Tensor,Tensor,Tensor> CPUDoubleType::thnn_conv2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CPUDoubleTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    auto finput_ = new CPUDoubleTensor(context);
    auto finput = Tensor(finput_, false);
    auto fgrad_input_ = new CPUDoubleTensor(context);
    auto fgrad_input = Tensor(fgrad_input_, false);
    THNN_DoubleSpatialConvolutionMM_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    finput_->maybeScalar(maybe_scalar);
    fgrad_input_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(output, finput, fgrad_input);
}
std::tuple<Tensor &,Tensor &,Tensor &> CPUDoubleType::thnn_conv2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto finput_ = checked_cast_tensor<CPUDoubleTensor>(finput.pImpl,"finput",7, false);
    auto fgrad_input_ = checked_cast_tensor<CPUDoubleTensor>(fgrad_input.pImpl,"fgrad_input",8, false);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",8, true);
    auto grad_weight_ = checked_cast_tensor<CPUDoubleTensor>(grad_weight.pImpl,"grad_weight",8, true);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = checked_cast_tensor<CPUDoubleTensor>(grad_bias.pImpl,"grad_bias",8, true);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_DoubleSpatialConvolutionMM_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, finput_->tensor, fgrad_input_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
    if (grad_weight_ || grad_bias_) THNN_DoubleSpatialConvolutionMM_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &, Tensor &>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> CPUDoubleType::thnn_conv2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto finput_ = checked_cast_tensor<CPUDoubleTensor>(finput.pImpl,"finput",7, false);
    auto fgrad_input_ = checked_cast_tensor<CPUDoubleTensor>(fgrad_input.pImpl,"fgrad_input",8, false);
    auto grad_input_ = output_mask[0] ? new CPUDoubleTensor(context) : nullptr;
    auto grad_input = Tensor(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_input_, false);
    auto grad_weight_ = output_mask[1] ? new CPUDoubleTensor(context) : nullptr;
    auto grad_weight = Tensor(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_weight_, false);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = output_mask[2] ? new CPUDoubleTensor(context) : nullptr;
    auto grad_bias = Tensor(grad_bias_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_bias_, false);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_DoubleSpatialConvolutionMM_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, finput_->tensor, fgrad_input_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
    if (grad_weight_ || grad_bias_) THNN_DoubleSpatialConvolutionMM_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor &,Tensor &,Tensor &> CPUDoubleType::thnn_conv3d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CPUDoubleTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",6, false);
    auto finput_ = checked_cast_tensor<CPUDoubleTensor>(finput.pImpl,"finput",6, false);
    auto fgrad_input_ = checked_cast_tensor<CPUDoubleTensor>(fgrad_input.pImpl,"fgrad_input",6, false);
    THNN_DoubleVolumetricConvolutionMM_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    finput_->maybeScalar(maybe_scalar);
    fgrad_input_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(output, finput, fgrad_input);
}
std::tuple<Tensor,Tensor,Tensor> CPUDoubleType::thnn_conv3d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CPUDoubleTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    auto finput_ = new CPUDoubleTensor(context);
    auto finput = Tensor(finput_, false);
    auto fgrad_input_ = new CPUDoubleTensor(context);
    auto fgrad_input = Tensor(fgrad_input_, false);
    THNN_DoubleVolumetricConvolutionMM_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    finput_->maybeScalar(maybe_scalar);
    fgrad_input_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(output, finput, fgrad_input);
}
std::tuple<Tensor &,Tensor &,Tensor &> CPUDoubleType::thnn_conv3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto finput_ = checked_cast_tensor<CPUDoubleTensor>(finput.pImpl,"finput",7, false);
    auto fgrad_input_ = checked_cast_tensor<CPUDoubleTensor>(fgrad_input.pImpl,"fgrad_input",8, false);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",8, true);
    auto grad_weight_ = checked_cast_tensor<CPUDoubleTensor>(grad_weight.pImpl,"grad_weight",8, true);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = checked_cast_tensor<CPUDoubleTensor>(grad_bias.pImpl,"grad_bias",8, true);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_DoubleVolumetricConvolutionMM_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1]);
    if (grad_weight_ || grad_bias_) THNN_DoubleVolumetricConvolutionMM_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &, Tensor &>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> CPUDoubleType::thnn_conv3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto finput_ = checked_cast_tensor<CPUDoubleTensor>(finput.pImpl,"finput",7, false);
    auto fgrad_input_ = checked_cast_tensor<CPUDoubleTensor>(fgrad_input.pImpl,"fgrad_input",8, false);
    auto grad_input_ = output_mask[0] ? new CPUDoubleTensor(context) : nullptr;
    auto grad_input = Tensor(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_input_, false);
    auto grad_weight_ = output_mask[1] ? new CPUDoubleTensor(context) : nullptr;
    auto grad_weight = Tensor(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_weight_, false);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = output_mask[2] ? new CPUDoubleTensor(context) : nullptr;
    auto grad_bias = Tensor(grad_bias_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_bias_, false);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_DoubleVolumetricConvolutionMM_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1]);
    if (grad_weight_ || grad_bias_) THNN_DoubleVolumetricConvolutionMM_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, finput_->tensor, fgrad_input_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor &,Tensor &,Tensor &> CPUDoubleType::thnn_conv_dilated2d_forward_out(Tensor & output, Tensor & columns, Tensor & ones, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CPUDoubleTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",7, false);
    auto columns_ = checked_cast_tensor<CPUDoubleTensor>(columns.pImpl,"columns",7, false);
    auto ones_ = checked_cast_tensor<CPUDoubleTensor>(ones.pImpl,"ones",7, false);
    THNN_DoubleSpatialDilatedConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    columns_->maybeScalar(maybe_scalar);
    ones_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(output, columns, ones);
}
std::tuple<Tensor,Tensor,Tensor> CPUDoubleType::thnn_conv_dilated2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CPUDoubleTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    auto columns_ = new CPUDoubleTensor(context);
    auto columns = Tensor(columns_, false);
    auto ones_ = new CPUDoubleTensor(context);
    auto ones = Tensor(ones_, false);
    THNN_DoubleSpatialDilatedConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    columns_->maybeScalar(maybe_scalar);
    ones_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(output, columns, ones);
}
std::tuple<Tensor &,Tensor &,Tensor &> CPUDoubleType::thnn_conv_dilated2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
    auto columns_ = checked_cast_tensor<CPUDoubleTensor>(columns.pImpl,"columns",8, false);
    auto ones_ = checked_cast_tensor<CPUDoubleTensor>(ones.pImpl,"ones",9, false);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",9, true);
    auto grad_weight_ = checked_cast_tensor<CPUDoubleTensor>(grad_weight.pImpl,"grad_weight",9, true);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = checked_cast_tensor<CPUDoubleTensor>(grad_bias.pImpl,"grad_bias",9, true);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_DoubleSpatialDilatedConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, columns_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
    if (grad_weight_ || grad_bias_) THNN_DoubleSpatialDilatedConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &, Tensor &>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> CPUDoubleType::thnn_conv_dilated2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<2>(stride, "stride", 5);
    auto padding_ = check_intlist<2>(padding, "padding", 6);
    auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
    auto columns_ = checked_cast_tensor<CPUDoubleTensor>(columns.pImpl,"columns",8, false);
    auto ones_ = checked_cast_tensor<CPUDoubleTensor>(ones.pImpl,"ones",9, false);
    auto grad_input_ = output_mask[0] ? new CPUDoubleTensor(context) : nullptr;
    auto grad_input = Tensor(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_input_, false);
    auto grad_weight_ = output_mask[1] ? new CPUDoubleTensor(context) : nullptr;
    auto grad_weight = Tensor(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_weight_, false);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = output_mask[2] ? new CPUDoubleTensor(context) : nullptr;
    auto grad_bias = Tensor(grad_bias_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_bias_, false);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_DoubleSpatialDilatedConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, columns_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
    if (grad_weight_ || grad_bias_) THNN_DoubleSpatialDilatedConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor &,Tensor &,Tensor &> CPUDoubleType::thnn_conv_dilated3d_forward_out(Tensor & output, Tensor & columns, Tensor & ones, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CPUDoubleTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 7);
    auto output_ = checked_cast_tensor<CPUDoubleTensor>(output.pImpl,"output",7, false);
    auto columns_ = checked_cast_tensor<CPUDoubleTensor>(columns.pImpl,"columns",7, false);
    auto ones_ = checked_cast_tensor<CPUDoubleTensor>(ones.pImpl,"ones",7, false);
    THNN_DoubleVolumetricDilatedConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    columns_->maybeScalar(maybe_scalar);
    ones_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor &, Tensor &, Tensor &>(output, columns, ones);
}
std::tuple<Tensor,Tensor,Tensor> CPUDoubleType::thnn_conv_dilated3d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",1, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",2, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 3);
    auto bias_ = checked_cast_tensor<CPUDoubleTensor>(bias.pImpl,"bias",4, true);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 7);
    auto output_ = new CPUDoubleTensor(context);
    auto output = Tensor(output_, false);
    auto columns_ = new CPUDoubleTensor(context);
    auto columns = Tensor(columns_, false);
    auto ones_ = new CPUDoubleTensor(context);
    auto ones = Tensor(ones_, false);
    THNN_DoubleVolumetricDilatedConvolution_updateOutput(context->thc_state, self_->tensor, output_->tensor, weight_->tensor, bias_ ? bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1]);
    bool maybe_scalar = self_->isScalar() && weight_->isScalar() && (!bias_ || bias_->isScalar());
    output_->maybeScalar(maybe_scalar);
    columns_->maybeScalar(maybe_scalar);
    ones_->maybeScalar(maybe_scalar);
    return std::tuple<Tensor, Tensor, Tensor>(output, columns, ones);
}
std::tuple<Tensor &,Tensor &,Tensor &> CPUDoubleType::thnn_conv_dilated3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 7);
    auto columns_ = checked_cast_tensor<CPUDoubleTensor>(columns.pImpl,"columns",8, false);
    auto ones_ = checked_cast_tensor<CPUDoubleTensor>(ones.pImpl,"ones",9, false);
    auto grad_input_ = checked_cast_tensor<CPUDoubleTensor>(grad_input.pImpl,"grad_input",9, true);
    auto grad_weight_ = checked_cast_tensor<CPUDoubleTensor>(grad_weight.pImpl,"grad_weight",9, true);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = checked_cast_tensor<CPUDoubleTensor>(grad_bias.pImpl,"grad_bias",9, true);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_DoubleVolumetricDilatedConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, columns_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1]);
    if (grad_weight_ || grad_bias_) THNN_DoubleVolumetricDilatedConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor &, Tensor &, Tensor &>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> CPUDoubleType::thnn_conv_dilated3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) const {
    auto grad_output_ = checked_cast_tensor<CPUDoubleTensor>(grad_output.pImpl,"grad_output",1, false);
    auto self_ = checked_cast_tensor<CPUDoubleTensor>(self.pImpl,"self",2, false);
    auto weight_ = checked_cast_tensor<CPUDoubleTensor>(weight.pImpl,"weight",3, false);
    auto kernel_size_ = check_intlist<3>(kernel_size, "kernel_size", 4);
    auto stride_ = check_intlist<3>(stride, "stride", 5);
    auto padding_ = check_intlist<3>(padding, "padding", 6);
    auto dilation_ = check_intlist<3>(dilation, "dilation", 7);
    auto columns_ = checked_cast_tensor<CPUDoubleTensor>(columns.pImpl,"columns",8, false);
    auto ones_ = checked_cast_tensor<CPUDoubleTensor>(ones.pImpl,"ones",9, false);
    auto grad_input_ = output_mask[0] ? new CPUDoubleTensor(context) : nullptr;
    auto grad_input = Tensor(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_input_, false);
    auto grad_weight_ = output_mask[1] ? new CPUDoubleTensor(context) : nullptr;
    auto grad_weight = Tensor(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_weight_, false);
    if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
    }
    auto grad_bias_ = output_mask[2] ? new CPUDoubleTensor(context) : nullptr;
    auto grad_bias = Tensor(grad_bias_ == nullptr ? (TensorImpl*)UndefinedTensor::singleton() : (TensorImpl*)grad_bias_, false);
    if (grad_bias.defined()) {
        grad_bias.resize_({ weight.size(0) });
        grad_bias.zero_();
    }
    if (grad_input_) THNN_DoubleVolumetricDilatedConvolution_updateGradInput(context->thc_state, self_->tensor, grad_output_->tensor, grad_input_ ? grad_input_->tensor : NULL, weight_->tensor, columns_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1]);
    if (grad_weight_ || grad_bias_) THNN_DoubleVolumetricDilatedConvolution_accGradParameters(context->thc_state, self_->tensor, grad_output_->tensor, grad_weight_ ? grad_weight_->tensor : NULL, grad_bias_ ? grad_bias_->tensor : NULL, columns_->tensor, ones_->tensor, kernel_size_[0], kernel_size_[2], kernel_size_[1], stride_[0], stride_[2], stride_[1], padding_[0], padding_[2], padding_[1], dilation_[0], dilation_[2], dilation_[1], 1);
    if (grad_input_) grad_input_->maybeScalar(self_->isScalar());
    return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}
Tensor & CPUDoubleType::abs_out(Tensor & result, const Tensor & self) const {
    return  at::native::_abs_out_cpu(result, self);
}
Tensor & CPUDoubleType::ceil_out(Tensor & result, const Tensor & self) const {
    return  at::native::_ceil_out_cpu(result, self);
}
Tensor & CPUDoubleType::cos_out(Tensor & result, const Tensor & self) const {
    return  at::native::_cos_out_cpu(result, self);
}
Tensor CPUDoubleType::embedding_dense_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) const {
    return  at::native::embedding_backward_cpu(grad, indices, num_weights, padding_idx, scale_grad_by_freq);
}
Tensor & CPUDoubleType::embedding_renorm_(Tensor & self, const Tensor & indices, double max_norm, double norm_type) const {
    return  at::native::embedding_renorm_cpu_(self, indices, max_norm, norm_type);
}
std::tuple<Tensor,Tensor,Tensor> CPUDoubleType::embedding_bag(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse) const {
    return  at::native::embedding_bag_cpu(weight, indices, offsets, scale_grad_by_freq, mode, sparse);
}
Tensor CPUDoubleType::embedding_bag_dense_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode) const {
    return  at::native::embedding_bag_backward_cpu(grad, indices, offsets, offset2bag, bag_size, num_weights, scale_grad_by_freq, mode);
}
Tensor & CPUDoubleType::exp_out(Tensor & result, const Tensor & self) const {
    return  at::native::_exp_out_cpu(result, self);
}
Tensor & CPUDoubleType::eye_out(Tensor & result, int64_t n, int64_t m) const {
    return  at::native::eye_out_cpu(result, n, m);
}
Tensor & CPUDoubleType::floor_out(Tensor & result, const Tensor & self) const {
    return  at::native::_floor_out_cpu(result, self);
}
Tensor CPUDoubleType::_fft_with_size(const Tensor & self, int64_t signal_ndim, bool complex_input, bool complex_output, bool inverse, IntList checked_signal_sizes, bool normalized, bool onesided, IntList output_sizes) const {
    return  at::native::_fft_mkl(self, signal_ndim, complex_input, complex_output, inverse, checked_signal_sizes, normalized, onesided, output_sizes);
}
Tensor & CPUDoubleType::log_out(Tensor & result, const Tensor & self) const {
    return  at::native::_log_out_cpu(result, self);
}
std::tuple<Tensor,Tensor> CPUDoubleType::RoiPooling2d_forward(const Tensor & input, const Tensor & rois, int64_t pooledHeight, int64_t pooledWidth, double spatialScale) const {
    return  at::native::RoiPooling2d_forward_cpu(input, rois, pooledHeight, pooledWidth, spatialScale);
}
Tensor CPUDoubleType::RoiPooling2d_backward(const Tensor & input, const Tensor & rois, int64_t pooledHeight, int64_t pooledWidth, double spatialScale, const Tensor & gradOutput, const Tensor & argmaxes) const {
    return  at::native::RoiPooling2d_backward_cpu(input, rois, pooledHeight, pooledWidth, spatialScale, gradOutput, argmaxes);
}
Tensor & CPUDoubleType::round_out(Tensor & result, const Tensor & self) const {
    return  at::native::_round_out_cpu(result, self);
}
Tensor & CPUDoubleType::sin_out(Tensor & result, const Tensor & self) const {
    return  at::native::_sin_out_cpu(result, self);
}
Tensor & CPUDoubleType::sspaddmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return  at::native::_sspaddmm_out_only_sparse(result, self, mat1, mat2, beta, alpha);
}
Tensor CPUDoubleType::sum(const Tensor & self) const {
    return  at::native::_sum_cpu(self);
}
Tensor & CPUDoubleType::sum_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    return  at::native::_sum_out_cpu(result, self, dim, keepdim);
}
Tensor & CPUDoubleType::sqrt_out(Tensor & result, const Tensor & self) const {
    return  at::native::_sqrt_out_cpu(result, self);
}
Tensor CPUDoubleType::prod(const Tensor & self) const {
    return  at::native::_prod_cpu(self);
}
Tensor & CPUDoubleType::prod_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    return  at::native::_prod_out_cpu(result, self, dim, keepdim);
}
Tensor & CPUDoubleType::trunc_out(Tensor & result, const Tensor & self) const {
    return  at::native::_trunc_out_cpu(result, self);
}
std::tuple<Tensor,Tensor> CPUDoubleType::_unique(const Tensor & self, bool sorted, bool return_inverse) const {
    return  at::native::_unique_cpu(self, sorted, return_inverse);
}
Tensor CPUDoubleType::_s_where(const Tensor & condition, const Tensor & self, const Tensor & other) const {
    return  at::native::_s_where_cpu(condition, self, other);
}
Tensor CPUDoubleType::_standard_gamma_grad(const Tensor & self, const Tensor & output) const {
    return  at::native::_standard_gamma_grad_cpu(self, output);
}
Tensor CPUDoubleType::poisson(const Tensor & self, Generator * generator) const {
    return  at::native::_s_poisson_cpu(self, generator);
}

}
