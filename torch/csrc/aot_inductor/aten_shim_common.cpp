#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <torch/csrc/aot_inductor/c/aten_shim.h>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else

#include <ATen/ops/_addmm_activation.h>
#include <ATen/ops/addmm.h>
#include <ATen/ops/as_strided.h>
#include <ATen/ops/bmm.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/mm.h>

#endif

// This file will be compiled as a part of libtorch, and the AOTInductor
// compiled model binary will load these C functions from the deployment host.
// As explained in aten_shim.h, this helps to solve ABI compatibility issues.
//
// In addtion, there is another benefit of adding a shim layer: for AOTInductor
// generated fallback aten calls, we don't have to go through its regular heavy
// machinery to maintain aten tensors and call through dispatacher. For
// instance, we could implement a fast path to call device-specific mm library
// instead of calling at::mm as a fallback. TODO: this fuctionality is largely
// not done yet.

namespace {

// An internal struct to represent a tensor in AOTInductor
//
// This is NOT meant to be a general replacement of at::Tensor. The design
// choices made here are highly couple with AOTInductor's codegen, especially
// with respect to resource management.
// 1. Because AOTInductor codegen does static memory planning, we don't need to
//    do any reference counting on the storage pointer. We store data_ptr as a
//    raw pointer, with its corresponding deleter to free the device storage
//    when the codegen-ed aot_tensor_free is called. It gets a little trickier
//    as we cross the boundary between AOTInductorTensor and at::Tensor.
//    See the comments in aten_tensor_to_aot_tensor and
//    aot_tensor_to_aten_tensor for more details.
// 2. sizes and strides are not dynamically allocated and managed here. They are
//    either coming from input/output aten tensors, or are codegen-ed as
//    constant arrays by AOTInductor. In both cases, sizes and strides are
//    guaranteed to be live during the model's execution. In a few cases,
//    AOTInductor may need to codegen extra arguments to for an op's output
//    tensor's sizes and strides, e.g. at::convolution.
// 3. As for AOTInductorTensor objects, they are dynamically allocated and
//    managed by std::unique_ptr. They can also be statically planned and
//    allocated later.
struct AOTInductorTensor {
  void* data_ptr;
  void* deleter;
  int64_t ndim;
  const int64_t* sizes;
  const int64_t* strides;
  int64_t storage_offset;
  int8_t device_type;
  int8_t device_index;
  int8_t dtype;
  int padding : 24;

  AOTInductorTensor(
      void* data_ptr,
      void* deleter,
      int64_t ndim,
      const int64_t* sizes,
      const int64_t* strides,
      int64_t storage_offset,
      int8_t device_type,
      int8_t device_index,
      int8_t dtype)
      : data_ptr(data_ptr),
        deleter(deleter),
        ndim(ndim),
        sizes(sizes),
        strides(strides),
        storage_offset(storage_offset),
        device_type(device_type),
        device_index(device_index),
        dtype(dtype),
        padding(0) {}

#ifndef NDEBUG
  void print() {
    std::cout << "data_ptr=" << data_ptr << ", ";
    std::cout << "deleter=" << deleter << ", ";
    std::cout << "ndim=" << ndim << ", ";
    std::cout << "sizes=" << sizes << ", ";
    std::cout << "strides=" << strides << ", ";
    std::cout << "storage_offset=" << storage_offset << ", ";
    std::cout << "device_type=" << int32_t(device_type) << ", ";
    std::cout << "device_index=" << int32_t(device_index) << ", ";
    std::cout << "dtype=" << int32_t(dtype) << std::endl;
  }
#endif
};

size_t get_dtype_bytes(AOTInductorScalarType dtype) {
  switch (dtype) {
    case kAOTInductorBool:
    case kAOTInductorByte:
    case kAOTInductorChar:
    case kAOTInductorQInt8:
    case kAOTInductorQUInt8:
      return 1;
    case kAOTInductorShort:
    case kAOTInductorHalf:
    case kAOTInductorBFloat16:
      return 2;
    case kAOTInductorInt:
    case kAOTInductorFloat:
    case kAOTInductorComplexHalf:
    case kAOTInductorQInt32:
      return 4;
    case kAOTInductorLong:
    case kAOTInductorDouble:
    case kAOTInductorComplexFloat:
      return 8;
    case kAOTInductorComplexDouble:
      return 16;
    default:
      AOT_INDUCTOR_CHECK(false, "Unkown AOTInductorScalarType")
      return 0;
  }
}

// TODO: this probably can also be statically planned to avoid any dynamic
// allocation
thread_local std::vector<std::unique_ptr<AOTInductorTensor>> tensor_manager_;
#ifndef NDEBUG
thread_local std::unordered_set<void*>* live_buffer_;
#endif

inline AOTInductorTensorHandle register_tensor(
    void* data_ptr,
    void* deleter,
    int64_t ndim,
    const int64_t* sizes,
    const int64_t* strides,
    int64_t storage_offset,
    int8_t device_type,
    int8_t device_index,
    int8_t dtype) {
  tensor_manager_.emplace_back(std::make_unique<AOTInductorTensor>(
      data_ptr,
      deleter,
      ndim,
      sizes,
      strides,
      storage_offset,
      device_type,
      device_index,
      dtype));
  return tensor_manager_[tensor_manager_.size() - 1].get();
}

using DeleterFnPtr = void (*)(void*);

void deleteNothing(void*) {}

AOTInductorTensorHandle aten_tensor_to_aot_tensor(
    AtenTensorHandle aten_tensor,
    const int64_t* sizes,
    const int64_t* strides) {
  // There are two scenarios when we can convert at::Tensor to AtenTensor:
  // 1. at::Tensor is passed in as model inputs/outputs, which is handled
  // by convert_input_output_to_aot_tensor.
  // 2. at::Tensor is created by an aten fallback op,  e.g. tensor allocation
  // ops like at::empty_strided, or aten ops without an out-variant such as
  // at::convolution, which is handled by this function.
  //
  // Here we swap out at::Tensor's deleter with deleteNothing, which makes sure
  // RC on at::Tensor's storage_impl won't delete the underlying device storage.
  // Later the statically planned aot_inductor_free will call the original
  // deleter to free the device storage. We also keep track of those storages in
  // the debug mode to make sure there is no storage leak.
  at::Tensor* t = static_cast<at::Tensor*>(aten_tensor);
  void* data_ptr = t->data_ptr();
  c10::DataPtr& mutable_ptr =
      t->storage().unsafeGetStorageImpl()->mutable_data_ptr();
  DeleterFnPtr deleter = mutable_ptr.get_deleter();
  AOT_INDUCTOR_CHECK(
      mutable_ptr.get() == mutable_ptr.get_context() ||
          deleter == deleteNothing,
      "Expect data_ and context_ to be the same in a UniqueVoidPtr if the aten \
        tensor is created by aten; otherwise the aten tensor should be created \
        by aot_inductor and have a deleteNothing deleter")

  if (deleter != deleteNothing) {
    bool success = mutable_ptr.compare_exchange_deleter(deleter, deleteNothing);
    AOT_INDUCTOR_CHECK(
        success,
        "Unexpected deleter function on storage, could not swap function");
    (void)success; // get around the unused variable warning
  }

  AOTInductorTensorHandle aot_tensor = register_tensor(
      data_ptr,
      (void*)deleter,
      t->dim(),
      sizes, // sizes has to be passed in because t->sizes() can released early
      strides, //
      t->storage_offset(),
      static_cast<AOTInductorDeviceType>(t->device().type()),
      static_cast<AOTInductorDeviceIndex>(t->device().index()),
      static_cast<AOTInductorScalarType>(t->scalar_type()));

#ifndef NDEBUG
  live_buffer_->insert(static_cast<AOTInductorTensor*>(aot_tensor)->data_ptr);
#endif
  return aot_tensor;
}

void aot_tensor_to_aten_tensor(
    AOTInductorTensorHandle aot_tensor,
    AtenTensorHandle aten_tensor) {
  // Turning AOTInductorTensor into at::Tensor is needed when we want to call
  // into an aten op as fallback. As described in aten_tensor_to_aot_tensor,
  // the device storage is managed by AOTInductor's static memory planner.
  AOTInductorTensor* t = static_cast<AOTInductorTensor*>(aot_tensor);
  c10::Device device{
      static_cast<c10::DeviceType>(t->device_type),
      static_cast<c10::DeviceIndex>(t->device_index)};
  c10::TensorOptions options = c10::TensorOptions().device(device).dtype(
      static_cast<c10::ScalarType>(t->dtype));

  // set the deleter to deleteNothing to stop at::Tensor from freeing the
  // device storage
  (*static_cast<at::Tensor*>(aten_tensor)) =
      at::for_blob(t->data_ptr, c10::IntArrayRef(t->sizes, t->ndim))
          .strides(c10::IntArrayRef(t->strides, t->ndim))
          .storage_offset(t->storage_offset)
          .context(t->data_ptr, deleteNothing)
          .options(options)
          .make_tensor();
}

} // namespace

void aot_inductor_initialize() {
  // c10 types and enums should rarely change, but use this checking to
  // fail loudly in the worst case
  AOT_INDUCTOR_CHECK(
      sizeof(c10::DeviceType) == sizeof(AOTInductorDeviceType),
      "c10::DeviceType data type has changed")
  AOT_INDUCTOR_CHECK(
      sizeof(c10::DeviceIndex) == sizeof(AOTInductorDeviceIndex),
      "c10::DeviceIndex data type has changed")
  AOT_INDUCTOR_CHECK(
      sizeof(c10::ScalarType) == sizeof(AOTInductorScalarType),
      "c10::ScalarType data type has changed")

  AOT_INDUCTOR_CHECK(
      kAOTInductorCPU == int8_t(c10::DeviceType::CPU),
      "kAOTInductorCPU != c10::DeviceType::CPU")
  AOT_INDUCTOR_CHECK(
      kAOTInductorCUDA == int8_t(c10::DeviceType::CUDA),
      "kAOTInductorCUDA != c10::DeviceType::CUDA")

  AOT_INDUCTOR_CHECK(
      kAOTInductorByte == int8_t(c10::ScalarType::Byte),
      "kAOTInductorByte != c10::ScalarType::Byte")
  AOT_INDUCTOR_CHECK(
      kAOTInductorChar == int8_t(c10::ScalarType::Char),
      "kAOTInductorChar != c10::ScalarType::Char")
  AOT_INDUCTOR_CHECK(
      kAOTInductorShort == int8_t(c10::ScalarType::Short),
      "kAOTInductorShort != c10::ScalarType::Short")
  AOT_INDUCTOR_CHECK(
      kAOTInductorInt == int8_t(c10::ScalarType::Int),
      "kAOTInductorInt != c10::ScalarType::Int")
  AOT_INDUCTOR_CHECK(
      kAOTInductorLong == int8_t(c10::ScalarType::Long),
      "kAOTInductorLong != c10::ScalarType::Long")
  AOT_INDUCTOR_CHECK(
      kAOTInductorHalf == int8_t(c10::ScalarType::Half),
      "kAOTInductorHalf != c10::ScalarType::Half")
  AOT_INDUCTOR_CHECK(
      kAOTInductorFloat == int8_t(c10::ScalarType::Float),
      "kAOTInductorFloat != c10::ScalarType::Float")
  AOT_INDUCTOR_CHECK(
      kAOTInductorDouble == int8_t(c10::ScalarType::Double),
      "kAOTInductorDouble != c10::ScalarType::Double")
  AOT_INDUCTOR_CHECK(
      kAOTInductorComplexHalf == int8_t(c10::ScalarType::ComplexHalf),
      "kAOTInductorComplexHalf != c10::ScalarType::ComplexHalf")
  AOT_INDUCTOR_CHECK(
      kAOTInductorComplexFloat == int8_t(c10::ScalarType::ComplexFloat),
      "kAOTInductorComplexFloat != c10::ScalarType::ComplexFloat")
  AOT_INDUCTOR_CHECK(
      kAOTInductorComplexDouble == int8_t(c10::ScalarType::ComplexDouble),
      "kAOTInductorComplexDouble != c10::ScalarType::ComplexDouble")
  AOT_INDUCTOR_CHECK(
      kAOTInductorBool == int8_t(c10::ScalarType::Bool),
      "kAOTInductorBool != c10::ScalarType::Bool")
  AOT_INDUCTOR_CHECK(
      kAOTInductorQInt8 == int8_t(c10::ScalarType::QInt8),
      "kAOTInductorQInt8 != c10::ScalarType::QInt8")
  AOT_INDUCTOR_CHECK(
      kAOTInductorQUInt8 == int8_t(c10::ScalarType::QUInt8),
      "kAOTInductorQUInt8 != c10::ScalarType::QUInt8")
  AOT_INDUCTOR_CHECK(
      kAOTInductorQInt32 == int8_t(c10::ScalarType::QInt32),
      "kAOTInductorQInt32 != c10::ScalarType::QInt32")
  AOT_INDUCTOR_CHECK(
      kAOTInductorBFloat16 == int8_t(c10::ScalarType::BFloat16),
      "kAOTInductorBFloat16 != c10::ScalarType::BFloat16")

#ifndef NDEBUG
  live_buffer_ = new std::unordered_set<void*>;
#endif
}

void aot_inductor_destroy() {
#ifndef NDEBUG
  AOT_INDUCTOR_CHECK(
      live_buffer_->size() == 0, "Live intermediate buffer leaking");
  delete live_buffer_;
#endif
}

void aot_inductor_free(AOTInductorTensorHandle aot_tensor) {
  AOTInductorTensor* t = static_cast<AOTInductorTensor*>(aot_tensor);
  AOT_INDUCTOR_CHECK(t != nullptr, "Invalid AOTInductorTensorHandle")
  AOT_INDUCTOR_CHECK(
      t->deleter != nullptr, "Invalid AOTInductorTensorHandle deleter")

  if (t->deleter == deleteNothing)
    return;

  DeleterFnPtr deleter = DeleterFnPtr(t->deleter);
#ifndef NDEBUG
  live_buffer_->erase(t->data_ptr);
#endif
  deleter(t->data_ptr);
}

void* aot_inductor_data_ptr(AOTInductorTensorHandle aot_tensor) {
  // t->storage_offset is in the number of elements, need to turn it
  // into bytes
  AOTInductorTensor* t = static_cast<AOTInductorTensor*>(aot_tensor);
  char* base = static_cast<char*>(t->data_ptr);
  size_t offset_in_bytes = t->storage_offset * get_dtype_bytes(t->dtype);
  return (void*)(base + offset_in_bytes);
}

AOTInductorTensorHandle convert_input_output_to_aot_tensor(
    AtenTensorHandle aten_tensor) {
  // Input and output tensors are managed by the external runtime, so they are
  // kept live in the AOTInductor runtime, and thus we set their corresponding
  // deleter to deleteNothing.
  at::Tensor* t = static_cast<at::Tensor*>(aten_tensor);
  return register_tensor(
      t->data_ptr(),
      (void*)deleteNothing,
      t->dim(),
      t->sizes().data(), // sizes will stay live, so no need to register them
      t->strides()
          .data(), // strides will stay live, so no need to register them
      t->storage_offset(),
      static_cast<AOTInductorDeviceType>(t->device().type()),
      static_cast<AOTInductorDeviceIndex>(t->device().index()),
      static_cast<AOTInductorScalarType>(t->scalar_type()));
}

AOTInductorTensorHandle aot_inductor_empty_strided(
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    AOTInductorDeviceType device_type,
    AOTInductorDeviceIndex device_index,
    AOTInductorScalarType dtype) {
  c10::IntArrayRef sizes(sizes_ptr, ndim);
  c10::IntArrayRef strides(strides_ptr, ndim);
  c10::Device device{
      static_cast<c10::DeviceType>(device_type),
      static_cast<c10::DeviceIndex>(device_index)};
  c10::TensorOptions options = c10::TensorOptions().device(device).dtype(
      static_cast<c10::ScalarType>(dtype));

  // TODO: directly call the caching allocator
  at::Tensor aten_result = at::empty_strided(sizes, strides, options);
  return aten_tensor_to_aot_tensor(&aten_result, sizes_ptr, strides_ptr);
}

// See as_strided_tensorimpl
AOTInductorTensorHandle aot_inductor_as_strided(
    AOTInductorTensorHandle self,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset) {
  AOTInductorTensor* t = static_cast<AOTInductorTensor*>(self);
  return register_tensor(
      t->data_ptr,
      t->deleter,
      ndim,
      sizes_ptr,
      strides_ptr,
      storage_offset,
      t->device_type,
      t->device_index,
      t->dtype);
}

AOTInductorTensorHandle aot_inductor_addmm_out(
    AOTInductorTensorHandle out,
    AOTInductorTensorHandle self,
    AOTInductorTensorHandle mat1,
    AOTInductorTensorHandle mat2,
    float beta,
    float alpha) {
  at::Tensor out_tensor;
  at::Tensor self_tensor;
  at::Tensor mat1_tensor;
  at::Tensor mat2_tensor;
  aot_tensor_to_aten_tensor(out, &out_tensor);
  aot_tensor_to_aten_tensor(self, &self_tensor);
  aot_tensor_to_aten_tensor(mat1, &mat1_tensor);
  aot_tensor_to_aten_tensor(mat2, &mat2_tensor);
  // TODO: directly call cublas
  at::addmm_out(out_tensor, self_tensor, mat1_tensor, mat2_tensor, beta, alpha);
  return out;
}

AOTInductorTensorHandle aot_inductor__addmm_activation(
    AOTInductorTensorHandle self,
    AOTInductorTensorHandle mat1,
    AOTInductorTensorHandle mat2,
    float beta,
    float alpha,
    uint8_t use_gelu) {
  at::Tensor self_tensor;
  at::Tensor mat1_tensor;
  at::Tensor mat2_tensor;
  aot_tensor_to_aten_tensor(self, &self_tensor);
  aot_tensor_to_aten_tensor(mat1, &mat1_tensor);
  aot_tensor_to_aten_tensor(mat2, &mat2_tensor);
  // at::_addmm_activation is not an out-variant
  at::Tensor aten_result = at::_addmm_activation(
      self_tensor, mat1_tensor, mat2_tensor, beta, alpha, use_gelu != 0);
  AOTInductorTensor* t = static_cast<AOTInductorTensor*>(self);
  return aten_tensor_to_aot_tensor(&aten_result, t->sizes, t->strides);
}

AOTInductorTensorHandle aot_inductor_bmm_out(
    AOTInductorTensorHandle out,
    AOTInductorTensorHandle self,
    AOTInductorTensorHandle mat2) {
  at::Tensor out_tensor;
  at::Tensor self_tensor;
  at::Tensor mat2_tensor;
  aot_tensor_to_aten_tensor(out, &out_tensor);
  aot_tensor_to_aten_tensor(self, &self_tensor);
  aot_tensor_to_aten_tensor(mat2, &mat2_tensor);
  // TODO: directly call cublas
  at::bmm_out(out_tensor, self_tensor, mat2_tensor);
  return out;
}

AOTInductorTensorHandle aot_inductor_copy_(
    AOTInductorTensorHandle src,
    AOTInductorTensorHandle dst) {
  at::Tensor src_tensor;
  at::Tensor dst_tensor;
  aot_tensor_to_aten_tensor(src, &src_tensor);
  aot_tensor_to_aten_tensor(dst, &dst_tensor);
  dst_tensor.copy_(src_tensor);
  return dst;
}

AOTInductorTensorHandle aot_inductor_mm_out(
    AOTInductorTensorHandle out,
    AOTInductorTensorHandle self,
    AOTInductorTensorHandle mat2) {
  at::Tensor out_tensor;
  at::Tensor self_tensor;
  at::Tensor mat2_tensor;
  aot_tensor_to_aten_tensor(out, &out_tensor);
  aot_tensor_to_aten_tensor(self, &self_tensor);
  aot_tensor_to_aten_tensor(mat2, &mat2_tensor);
  // TODO: directly call cublas
  at::mm_out(out_tensor, self_tensor, mat2_tensor);
  return out;
}
