#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
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
#include <ATen/ops/convolution.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/mm.h>

#endif

// See the notes in shim.h for why we need a shim layer here.
//
// In addition, the shim layer has another benefit: for AOTInductor generated
// fallback aten calls, we don't have to go through its regular heavy machinery
// to maintain aten tensors and call through dispatcher. For instance, we could
// implement a fast path to call device-specific mm library instead of calling
// at::mm as a fallback.
// TODO: this fuctionality is largely not implmented.

#define CONVERT_EXCEPTION_TO_ERROR_CODE(...)                 \
  try {                                                      \
    __VA_ARGS__                                              \
  } catch (const std::exception& e) {                        \
    std::cerr << "Error: " << e.what() << std::endl;         \
    return AOTITorchError::Failure;                          \
  } catch (...) {                                            \
    std::cerr << "Unknown exception occurred." << std::endl; \
    return AOTITorchError::Failure;                          \
  }                                                          \
  return AOTITorchError::Success;

namespace {

// An internal struct to represent a tensor in AOTInductor
//
// This is NOT meant to be a general replacement of at::Tensor. The design
// choices made here are highly coupled with AOTInductor's codegen, especially
// with respect to resource management.
// 1. Because AOTInductor codegen does static memory planning, we don't need to
//    do any reference counting on the storage pointer. We store data_ptr as a
//    raw pointer, with its corresponding deleter to free the device storage
//    when the codegen-ed aoti_torch_free_tensor_storage is called. It gets a
//    little trickier as we cross the boundary between AOTITensor and
//    at::Tensor. See the comments in intermediate_aten_tensor_to_aoti_tensor
//    and aoti_tensor_to_aten_tensor for more details.
// 2. sizes and strides are not dynamically allocated or managed here. They
//    are either from input/output aten tensors, or are statically codegen-ed as
//    constant arrays. In both cases, sizes and strides are guaranteed to be
//    live during the model's execution.
struct AOTITensor {
  void* data_ptr;
  void* deleter;
  int64_t ndim;
  const int64_t* sizes;
  const int64_t* strides;
  int64_t storage_offset;
  int32_t device_type;
  int32_t device_index;
  int32_t dtype;
  int32_t padding;

  AOTITensor(
      void* data_ptr,
      void* deleter,
      int64_t ndim,
      const int64_t* sizes,
      const int64_t* strides,
      int64_t storage_offset,
      int32_t device_type,
      int32_t device_index,
      int32_t dtype)
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
};

// A placeholder for optional tensor
AOTITensor optional_tensor_(
    nullptr,
    nullptr,
    -1,
    nullptr,
    nullptr,
    -1,
    -1,
    -1,
    -1);

size_t get_dtype_bytes(int32_t dtype) {
  switch (static_cast<c10::ScalarType>(dtype)) {
    case c10::ScalarType::Bool:
    case c10::ScalarType::Byte:
    case c10::ScalarType::Char:
    case c10::ScalarType::QInt8:
    case c10::ScalarType::QUInt8:
      return 1;
    case c10::ScalarType::Short:
    case c10::ScalarType::Half:
    case c10::ScalarType::BFloat16:
      return 2;
    case c10::ScalarType::Int:
    case c10::ScalarType::Float:
    case c10::ScalarType::ComplexHalf:
    case c10::ScalarType::QInt32:
      return 4;
    case c10::ScalarType::Long:
    case c10::ScalarType::Double:
    case c10::ScalarType::ComplexFloat:
      return 8;
    case c10::ScalarType::ComplexDouble:
      return 16;
    default:
      TORCH_CHECK(false, "Unkown c10::ScalarType")
      return 0;
  }
}

// Convert AOTITensor into at::Tensor
//
// This is a private utility function, so ok to have at::Tensor in its signature
void aoti_tensor_to_aten_tensor(
    AOTITensorHandle aoti_tensor,
    at::Tensor& aten_tensor) {
  AOTITensor* t = reinterpret_cast<AOTITensor*>(aoti_tensor);
  c10::Device device{
      static_cast<c10::DeviceType>(t->device_type),
      static_cast<c10::DeviceIndex>(t->device_index)};
  c10::TensorOptions options = c10::TensorOptions().device(device).dtype(
      static_cast<c10::ScalarType>(t->dtype));

  // Set deleter to at::detail::noopDelete so that the storage won't be freed by
  // the newly allocated aten tensor
  aten_tensor = at::for_blob(t->data_ptr, c10::IntArrayRef(t->sizes, t->ndim))
                    .strides(c10::IntArrayRef(t->strides, t->ndim))
                    .storage_offset(t->storage_offset)
                    .context(t->data_ptr, at::detail::noopDelete)
                    .options(options)
                    .make_tensor();
}

using TensorManager = std::vector<std::unique_ptr<AOTITensor>>;

AOTITensorHandle create_managed_atoi_tensor(
    AOTITensorManagerHandle tensor_manager,
    void* data_ptr,
    void* deleter,
    int64_t ndim,
    const int64_t* sizes,
    const int64_t* strides,
    int64_t storage_offset,
    int32_t device_type,
    int32_t device_index,
    int32_t dtype) {
  TensorManager* manager = reinterpret_cast<TensorManager*>(tensor_manager);
  manager->emplace_back(std::make_unique<AOTITensor>(
      data_ptr,
      deleter,
      ndim,
      sizes,
      strides,
      storage_offset,
      device_type,
      device_index,
      dtype));
  return reinterpret_cast<AOTITensorHandle>(manager->back().get());
}

// There are two scenarios when we can convert at::Tensor to AtenTensor:
// 1. at::Tensor is passed in as model inputs/outputs, which is handled
// by convert_input_output_to_aoti_tensor.
// 2. at::Tensor is created by an aten fallback op,  e.g. tensor allocation
// ops like at::empty_strided, or aten ops without an out-variant such as
// at::convolution, which is handled by this function.
//
// This is a private utility function, so ok to have at::Tensor in its signature
AOTITensorHandle intermediate_aten_tensor_to_aoti_tensor(
    AOTITensorManagerHandle tensor_manager,
    at::Tensor& aten_tensor,
    const int64_t* sizes,
    const int64_t* strides) {
  c10::DataPtr& mutable_ptr =
      aten_tensor.storage().unsafeGetStorageImpl()->mutable_data_ptr();
  c10::DeleterFnPtr deleter = mutable_ptr.get_deleter();
  TORCH_CHECK(
      mutable_ptr.get() == mutable_ptr.get_context() ||
          deleter == at::detail::noopDelete,
      "Expect data_ and context_ to be the same in a UniqueVoidPtr if the aten \
        tensor is created by aten; otherwise the aten tensor should be created \
        by aot_inductor and have a at::detail::noopDelete deleter")

  // We swap out this intermediate aten tensor's deleter with noopDelete, which
  // makes sure RC on that tensor's storage_impl won't delete the underlying
  // storage, which effectively means all the intermediate tensors are managed
  // by AOTITensor and are statically planned to call deallocation.
  if (deleter != at::detail::noopDelete) {
    bool success =
        mutable_ptr.compare_exchange_deleter(deleter, at::detail::noopDelete);
    TORCH_CHECK(
        success,
        "Unexpected deleter function on storage, could not swap function");
    // (void)success; // get around the unused variable warning
  }

  return create_managed_atoi_tensor(
      tensor_manager,
      aten_tensor.data_ptr(),
      (void*)deleter,
      aten_tensor.dim(),
      sizes,
      strides,
      aten_tensor.storage_offset(),
      static_cast<int32_t>(aten_tensor.device().type()),
      static_cast<int32_t>(aten_tensor.device().index()),
      static_cast<int32_t>(aten_tensor.scalar_type()));
}

} // namespace

int32_t aoti_torch_device_type_cpu() {
  return (int32_t)c10::DeviceType::CPU;
}

int32_t aoti_torch_device_type_cuda() {
  return (int32_t)c10::DeviceType::CUDA;
}

int32_t aoti_torch_dtype_bfloat16() {
  return (int32_t)c10::ScalarType::BFloat16;
}

int32_t aoti_torch_dtype_float16() {
  return (int32_t)c10::ScalarType::Half;
}

int32_t aoti_torch_dtype_float32() {
  return (int32_t)c10::ScalarType::Float;
}

int32_t aoti_torch_dtype_float64() {
  return (int32_t)c10::ScalarType::Double;
}

int32_t aoti_torch_dtype_uint8() {
  return (int32_t)c10::ScalarType::Byte;
}

int32_t aoti_torch_dtype_int8() {
  return (int32_t)c10::ScalarType::Char;
}

int32_t aoti_torch_dtype_int16() {
  return (int32_t)c10::ScalarType::Short;
}

int32_t aoti_torch_dtype_int32() {
  return (int32_t)c10::ScalarType::Int;
}

int32_t aoti_torch_dtype_int64() {
  return (int32_t)c10::ScalarType::Long;
}

AOTITensorHandle aoti_torch_optional_tensor() {
  return reinterpret_cast<AOTITensorHandle>(&optional_tensor_);
}

AOTITorchError aoti_torch_create_tensor_manager(
    AOTITensorManagerHandle* tensor_manager,
    int64_t reserve_size) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    TensorManager* manager = new TensorManager();
    manager->reserve(reserve_size);
    *tensor_manager = reinterpret_cast<AOTITensorManagerHandle>(manager);
  });
}

AOTITorchError aoti_torch_destroy_tensor_manager(
    AOTITensorManagerHandle tensor_manager) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    TensorManager* manager = reinterpret_cast<TensorManager*>(tensor_manager);
    delete manager;
  });
}

AOTITorchError aoti_torch_extern_tensor_to_aoti_tensor(
    AOTITensorManagerHandle tensor_manager,
    AOTITensorHandle* aoti_tensor,
    AtenTensorHandle aten_tensor) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* atensor = reinterpret_cast<at::Tensor*>(aten_tensor);
    *aoti_tensor = create_managed_atoi_tensor(
        tensor_manager,
        atensor->data_ptr(),
        // Input and output tensors are managed by the external runtime, so we
        // set the deleter as at::detail::noopDelete to prevent the tensor
        // storage from being freed, and also the size and stride pointers are
        // guaranteed to be valid.
        (void*)at::detail::noopDelete,
        atensor->dim(),
        atensor->sizes().data(),
        atensor->strides().data(),
        0, // atensor->data_ptr() has added storage_offset, so set to 0 here
        static_cast<int32_t>(atensor->device().type()),
        static_cast<int32_t>(atensor->device().index()),
        static_cast<int32_t>(atensor->scalar_type()));
  });
}

AOTITorchError aoti_torch_free_tensor_storage(AOTITensorHandle aoti_tensor) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTITensor* t = reinterpret_cast<AOTITensor*>(aoti_tensor);
    TORCH_CHECK(
        t != nullptr && t->data_ptr && t->deleter != nullptr,
        "aoti_torch_free_tensor_storage: Invalid AOTITensorHandle")
    c10::DeleterFnPtr deleter = reinterpret_cast<c10::DeleterFnPtr>(t->deleter);
    deleter(t->data_ptr);
    t->data_ptr = nullptr; // avoid double free
  });
}

AOTITorchError aoti_torch_get_data_ptr(
    AOTITensorHandle aoti_tensor,
    void** data_ptr) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTITensor* t = reinterpret_cast<AOTITensor*>(aoti_tensor);
    char* base = static_cast<char*>(t->data_ptr);
    // t->storage_offset is in the number of elements, so turn it into bytes
    size_t offset_in_bytes = t->storage_offset * get_dtype_bytes(t->dtype);
    *data_ptr = (void*)(base + offset_in_bytes);
  });
}

AOTITensorHandle aoti_torch__reinterpret_tensor(
    AOTITensorManagerHandle tensor_manager,
    AOTITensorHandle self,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t offset_increment) {
  AOTITensor* t = reinterpret_cast<AOTITensor*>(self);
  return create_managed_atoi_tensor(
      tensor_manager,
      t->data_ptr,
      t->deleter,
      ndim,
      sizes_ptr,
      strides_ptr,
      t->storage_offset + offset_increment,
      t->device_type,
      t->device_index,
      t->dtype);
}

// TODO: implement a more efficient version instead of calling into aten
AOTITorchError aoti_torch_empty_strided(
    AOTITensorManagerHandle tensor_manager,
    AOTITensorHandle* out,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    c10::IntArrayRef sizes(sizes_ptr, ndim);
    c10::IntArrayRef strides(strides_ptr, ndim);
    c10::Device device{
        static_cast<c10::DeviceType>(device_type),
        static_cast<c10::DeviceIndex>(device_index)};
    c10::TensorOptions options = c10::TensorOptions().device(device).dtype(
        static_cast<c10::ScalarType>(dtype));
    at::Tensor aten_result = at::empty_strided(sizes, strides, options);
    *out = intermediate_aten_tensor_to_aoti_tensor(
        tensor_manager, aten_result, sizes_ptr, strides_ptr);
  });
}

// TODO: implement a more efficient version instead of calling into aten
AOTITorchError aoti_torch_tensor_copy_(
    AOTITensorHandle src,
    AOTITensorHandle dst) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor src_tensor;
    at::Tensor dst_tensor;

    aoti_tensor_to_aten_tensor(src, src_tensor);
    aoti_tensor_to_aten_tensor(dst, dst_tensor);
    dst_tensor.copy_(src_tensor);
  });
}

// TODO: implement a more efficient version instead of calling into aten
AOTITorchError aoti_torch_addmm_out(
    AOTITensorHandle out,
    AOTITensorHandle self,
    AOTITensorHandle mat1,
    AOTITensorHandle mat2,
    float beta,
    float alpha) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor out_tensor;
    at::Tensor self_tensor;
    at::Tensor mat1_tensor;
    at::Tensor mat2_tensor;
    aoti_tensor_to_aten_tensor(out, out_tensor);
    aoti_tensor_to_aten_tensor(self, self_tensor);
    aoti_tensor_to_aten_tensor(mat1, mat1_tensor);
    aoti_tensor_to_aten_tensor(mat2, mat2_tensor);
    at::addmm_out(
        out_tensor, self_tensor, mat1_tensor, mat2_tensor, beta, alpha);
  });
}

// TODO: implement a more efficient version instead of calling into aten
AOTITorchError aoti_torch_bmm_out(
    AOTITensorHandle out,
    AOTITensorHandle self,
    AOTITensorHandle mat2) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor out_tensor;
    at::Tensor self_tensor;
    at::Tensor mat2_tensor;
    aoti_tensor_to_aten_tensor(out, out_tensor);
    aoti_tensor_to_aten_tensor(self, self_tensor);
    aoti_tensor_to_aten_tensor(mat2, mat2_tensor);
    at::bmm_out(out_tensor, self_tensor, mat2_tensor);
  });
}

// TODO: implement a more efficient version instead of calling into aten
AOTITorchError aoti_torch_convolution(
    AOTITensorManagerHandle tensor_manager,
    AOTITensorHandle* out,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    AOTITensorHandle input,
    AOTITensorHandle weight,
    AOTITensorHandle bias,
    const int64_t* stride,
    size_t len_stride,
    const int64_t* padding,
    size_t len_padding,
    const int64_t* dilation,
    size_t len_dilation,
    int32_t transposed,
    const int64_t* output_padding,
    size_t len_output_padding,
    int32_t groups) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor input_tensor;
    at::Tensor weight_tensor;
    at::Tensor output_tensor;
    aoti_tensor_to_aten_tensor(input, input_tensor);
    aoti_tensor_to_aten_tensor(weight, weight_tensor);
    ndim;
    c10::IntArrayRef stride_ref(stride, len_stride);
    c10::IntArrayRef padding_ref(padding, len_padding);
    c10::IntArrayRef dilation_ref(dilation, len_dilation);
    c10::IntArrayRef output_padding_ref(output_padding, len_output_padding);

    if (bias == aoti_torch_optional_tensor()) {
      c10::optional<at::Tensor> optional_bias;
      output_tensor = at::convolution(
          input_tensor,
          weight_tensor,
          optional_bias,
          stride_ref,
          padding_ref,
          dilation_ref,
          transposed,
          output_padding_ref,
          groups);
    } else {
      at::Tensor bias_tensor;
      aoti_tensor_to_aten_tensor(bias, bias_tensor);
      output_tensor = at::convolution(
          input_tensor,
          weight_tensor,
          bias_tensor,
          stride_ref,
          padding_ref,
          dilation_ref,
          transposed,
          output_padding_ref,
          groups);
    }
    *out = intermediate_aten_tensor_to_aoti_tensor(
        tensor_manager, output_tensor, sizes_ptr, strides_ptr);
  });
}

// TODO: implement a more efficient version instead of calling into aten
AOTITorchError aoti_torch_mm_out(
    AOTITensorHandle out,
    AOTITensorHandle self,
    AOTITensorHandle mat2) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor out_tensor;
    at::Tensor self_tensor;
    at::Tensor mat2_tensor;
    aoti_tensor_to_aten_tensor(out, out_tensor);
    aoti_tensor_to_aten_tensor(self, self_tensor);
    aoti_tensor_to_aten_tensor(mat2, mat2_tensor);
    at::mm_out(out_tensor, self_tensor, mat2_tensor);
  });
}
