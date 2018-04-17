// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/CPUHalfType.h"
#include "ATen/CPUHalfStorage.h"
#include "ATen/CPUHalfTensor.h"
#include "ATen/CPUGenerator.h"
#include "ATen/CPUByteTensor.h"
#include "ATen/CPUIntTensor.h"
#include "ATen/CPULongTensor.h"
#include "ATen/Tensor.h"
#include "ATen/CPUHalfTensor.h"
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

CPUHalfType::CPUHalfType(Context* context)
: Type(context, /*is_variable_or_undefined=*/false) {}
ScalarType CPUHalfType::scalarType() const {
  return ScalarType::Half;
}
Backend CPUHalfType::backend() const {
  return Backend::CPU;
}
bool CPUHalfType::is_cuda() const { return backend() == kCUDA || backend() == kSparseCUDA; }
bool CPUHalfType::is_sparse() const { return backend() == kSparseCPU || backend() == kSparseCUDA; }
bool CPUHalfType::is_distributed() const { return false; }

std::unique_ptr<Storage> CPUHalfType::storage() const {
  return std::unique_ptr<Storage>(new CPUHalfStorage(context));
}
std::unique_ptr<Storage> CPUHalfType::storage(size_t size) const {
  return std::unique_ptr<Storage>(new CPUHalfStorage(context,size));
}
std::unique_ptr<Storage> CPUHalfType::storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const {
    return std::unique_ptr<Storage>(
      new CPUHalfStorage(context,data,size,deleter));
}
std::unique_ptr<Storage> CPUHalfType::storageWithAllocator(int64_t size, std::unique_ptr<Allocator> allocator) const {
    return std::unique_ptr<Storage>(
        new CPUHalfStorage(context, size, std::move(allocator)));
}
Tensor CPUHalfType::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THHalfTensor_retain( (THHalfTensor*) th_pointer);
  return Tensor(new CPUHalfTensor(context,(THHalfTensor*)(th_pointer)), false);
}
std::unique_ptr<Storage> CPUHalfType::unsafeStorageFromTH(void * th_pointer, bool retain) const {
  if (retain)
    THHalfStorage_retain( (THHalfStorage*) th_pointer);
  return std::unique_ptr<Storage>(new CPUHalfStorage(context, (THHalfStorage*) th_pointer));
}
std::unique_ptr<Generator> CPUHalfType::generator() const {
  return std::unique_ptr<Generator>(new CPUGenerator(context));
}

const char * CPUHalfType::toString() const {
  return CPUHalfType::typeString();
}
TypeID CPUHalfType::ID() const {
  return TypeID::CPUHalf;
}

std::size_t CPUHalfType::elementSizeInBytes() const {
  return sizeof(Half);
}

const char * CPUHalfType::typeString() {
  return "CPUHalfType";
}

/* example
Tensor * CPUHalfType::add(Tensor & a, Tensor & b) {
  std::cout << "add CPUHalfTensor\n";
  return &a;
}
*/

int64_t CPUHalfType::storage_offset(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUHalfTensor>(self.pImpl,"self",1, false);
    return static_cast<int64_t>(THHalfTensor_storageOffset(self_->tensor));
}
Tensor & CPUHalfType::resize_(Tensor & self, IntList size) const {
    auto self_ = checked_cast_tensor<CPUHalfTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    THHalfTensor_resize(self_->tensor, size_, NULL);
    self_->maybeScalar(size.size() == 0);
    return self;
}
int64_t CPUHalfType::numel(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUHalfTensor>(self.pImpl,"self",1, false);
    return static_cast<int64_t>(THHalfTensor_nElement(self_->tensor));
}
Tensor & CPUHalfType::set_(Tensor & self, Storage & storage) const {
    auto self_ = checked_cast_tensor<CPUHalfTensor>(self.pImpl,"self",1, false);
    auto storage_ = checked_cast_storage<CPUHalfStorage>(&storage,"storage",2);
    THHalfTensor_setStorage(self_->tensor, storage_->storage, 0, THLongStorageView::makeFromLength(static_cast<int64_t>(storage.size())), NULL);
    self_->maybeScalar(false);
    return self;
}
Tensor & CPUHalfType::set_(Tensor & self, Storage & sourceStorage, int64_t storage_offset, IntList size, IntList stride) const {
    auto self_ = checked_cast_tensor<CPUHalfTensor>(self.pImpl,"self",1, false);
    auto sourceStorage_ = checked_cast_storage<CPUHalfStorage>(&sourceStorage,"sourceStorage",2);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    THHalfTensor_setStorage(self_->tensor, sourceStorage_->storage, storage_offset, size_, stride_);
    self_->maybeScalar(size.size() == 0);
    return self;
}
Tensor & CPUHalfType::set_(Tensor & self, const Tensor & source) const {
    auto self_ = checked_cast_tensor<CPUHalfTensor>(self.pImpl,"self",1, false);
    auto source_ = checked_cast_tensor<CPUHalfTensor>(source.pImpl,"source",2, false);
    THHalfTensor_set(self_->tensor, source_->tensor);
    self_->maybeScalar(source_->isScalar());
    return self;
}
Tensor & CPUHalfType::set_(Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUHalfTensor>(self.pImpl,"self",1, false);
    THHalfTensor_setStorage(self_->tensor, NULL, 0, NULL, NULL);
    self_->maybeScalar(false);
    return self;
}
bool CPUHalfType::is_contiguous(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUHalfTensor>(self.pImpl,"self",1, false);
    return THHalfTensor_isContiguous(self_->tensor);
}
bool CPUHalfType::is_set_to(const Tensor & self, const Tensor & tensor) const {
    auto self_ = checked_cast_tensor<CPUHalfTensor>(self.pImpl,"self",1, false);
    auto tensor_ = checked_cast_tensor<CPUHalfTensor>(tensor.pImpl,"tensor",2, false);
    return THHalfTensor_isSetTo(self_->tensor, tensor_->tensor);
}
Tensor CPUHalfType::transpose(const Tensor & self, int64_t dim0, int64_t dim1) const {
    auto self_ = checked_cast_tensor<CPUHalfTensor>(self.pImpl,"self",1, false);
    dim0 = maybe_wrap_dim(dim0, self_);
    dim1 = maybe_wrap_dim(dim1, self_);
    return Tensor((new CPUHalfTensor(context, THHalfTensor_newTranspose(self_->tensor, dim0, dim1)))->maybeScalar(self_->isScalar()),false);
}
Tensor CPUHalfType::t(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUHalfTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPUHalfTensor(context, THHalfTensor_newTranspose(self_->tensor, 0, 1)))->maybeScalar(self_->isScalar()),false);
}
Tensor CPUHalfType::clone(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUHalfTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPUHalfTensor(context, THHalfTensor_newClone(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor CPUHalfType::unfold(const Tensor & self, int64_t dimension, int64_t size, int64_t step) const {
    auto result_ = new CPUHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUHalfTensor>(self.pImpl,"self",1, false);
    dimension = maybe_wrap_dim(dimension, self_);
    THHalfTensor_unfold(result_->tensor, self_->tensor, dimension, size, step);
    result_->maybeScalar(self_->isScalar());
    return result;
}
void* CPUHalfType::data_ptr(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUHalfTensor>(self.pImpl,"self",1, false);
    return THHalfTensor_data(self_->tensor);
}
Tensor CPUHalfType::tensor(Storage & storage, int64_t storageOffset, IntList size, IntList stride) const {
    auto storage_ = checked_cast_storage<CPUHalfStorage>(&storage,"storage",1);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    return Tensor((new CPUHalfTensor(context, THHalfTensor_newWithStorage(storage_->storage, storageOffset, size_, stride_)))->maybeScalar(size.size() == 0),false);
}
Tensor CPUHalfType::tensor(IntList size) const {
    auto size_ = THLongStorageView::makeFromSize(size);
    return Tensor((new CPUHalfTensor(context, THHalfTensor_newWithSize(size_, NULL)))->maybeScalar(size.size() == 0),false);
}
Tensor CPUHalfType::tensor(IntList size, IntList stride) const {
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    return Tensor((new CPUHalfTensor(context, THHalfTensor_newWithSize(size_, stride_)))->maybeScalar(size.size() == 0),false);
}
Tensor CPUHalfType::tensor() const {
    return Tensor((new CPUHalfTensor(context, THHalfTensor_new())),false);
}
Tensor CPUHalfType::alias(const Tensor & self) const {
    auto self_ = checked_cast_tensor<CPUHalfTensor>(self.pImpl,"self",1, false);
    return Tensor((new CPUHalfTensor(context, THHalfTensor_newWithTensor(self_->tensor)))->maybeScalar(self_->isScalar()),false);
}
Tensor & CPUHalfType::as_strided_out(Tensor & result, const Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    auto result_ = checked_cast_tensor<CPUHalfTensor>(result.pImpl,"result",0, false);
    auto self_ = checked_cast_tensor<CPUHalfTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    if (storage_offset == -1) {
      storage_offset = self_->tensor->storageOffset;
    }
    THHalfTensor_setStorage(result_->tensor, self_->tensor->storage, storage_offset, size_, stride_);
    result_->maybeScalar(size.size() == 0);
    return result;
}
Tensor CPUHalfType::as_strided(const Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    auto result_ = new CPUHalfTensor(context);
    auto result = Tensor(result_, false);
    auto self_ = checked_cast_tensor<CPUHalfTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    if (storage_offset == -1) {
      storage_offset = self_->tensor->storageOffset;
    }
    THHalfTensor_setStorage(result_->tensor, self_->tensor->storage, storage_offset, size_, stride_);
    result_->maybeScalar(size.size() == 0);
    return result;
}
Tensor & CPUHalfType::as_strided_(Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    auto self_ = checked_cast_tensor<CPUHalfTensor>(self.pImpl,"self",1, false);
    auto size_ = THLongStorageView::makeFromSize(size);
    auto stride_ = THLongStorageView::makeFromStride(stride, is_noelem_tensor_size(size));
    if (storage_offset == -1) {
      storage_offset = self_->tensor->storageOffset;
    }
    THHalfTensor_setStorage(self_->tensor, self_->tensor->storage, storage_offset, size_, stride_);
    self_->maybeScalar(size.size() == 0);
    return self;
}

}
