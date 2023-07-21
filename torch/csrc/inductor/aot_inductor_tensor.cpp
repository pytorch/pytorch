#include <c10/core/Allocator.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/csrc/inductor/aot_inductor_tensor.h>
#include <torch/csrc/inductor/inductor_ops.h>
#include <memory>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else

#include <ATen/ops/addmm.h>
#include <ATen/ops/as_strided.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/mm.h>
#endif

namespace {
AotInductorDevice convert_to_aot_inductor_device(c10::Device device) {
  AotInductorDevice result{
      AotInductorDeviceType(device.type()), device.index()};
  return result;
}

c10::Device convert_to_c10_device(AotInductorDevice device) {
  c10::Device result{c10::DeviceType(device.device_type), device.device_id};
  return result;
}

std::vector<std::vector<int64_t>>* sizes_manager;
std::vector<std::vector<int64_t>>* strides_manager;

const int64_t* register_sizes(c10::IntArrayRef sizes) {
  sizes_manager->emplace_back(sizes.vec());
  return sizes_manager->at(sizes_manager->size() - 1).data();
}

const int64_t* register_strides(c10::IntArrayRef strides) {
  strides_manager->emplace_back(strides.vec());
  return strides_manager->at(strides_manager->size() - 1).data();
}

using DeleterFnPtr = void (*)(void*);

void deleteNothing(void*) {}

} // namespace

void aot_inductor_initialize() {
  // c10 enums should rarely change, but use this checking to fail loudly in the
  // worst case
  AOT_INDUCTOR_CHECK(
      kAotInductorCPU == int(c10::kCPU), "kAotInductorCPU != c10::kCPU");
  AOT_INDUCTOR_CHECK(
      kAotInductorCUDA == int(c10::kCUDA), "kAotInductorCUDA != c10::kCUDA");

  AOT_INDUCTOR_CHECK(
      kAotInductorByte == int(c10::kByte), "kAotInductorByte != c10::kByte");
  AOT_INDUCTOR_CHECK(
      kAotInductorChar == int(c10::kChar), "kAotInductorChar != c10::kChar");
  AOT_INDUCTOR_CHECK(
      kAotInductorShort == int(c10::kShort),
      "kAotInductorShort != c10::kShort");
  AOT_INDUCTOR_CHECK(
      kAotInductorInt == int(c10::kInt), "kAotInductorInt != c10::kInt");
  AOT_INDUCTOR_CHECK(
      kAotInductorLong == int(c10::kLong), "kAotInductorLong != c10::kLong");
  AOT_INDUCTOR_CHECK(
      kAotInductorHalf == int(c10::kHalf), "kAotInductorHalf != c10::kHalf");
  AOT_INDUCTOR_CHECK(
      kAotInductorFloat == int(c10::kFloat),
      "kAotInductorFloat != c10::kFloat");
  AOT_INDUCTOR_CHECK(
      kAotInductorDouble == int(c10::kDouble),
      "kAotInductorDouble != c10::kDouble");
  AOT_INDUCTOR_CHECK(
      kAotInductorComplexHalf == int(c10::kComplexHalf),
      "kAotInductorComplexHalf != c10::kComplexHalf");
  AOT_INDUCTOR_CHECK(
      kAotInductorComplexFloat == int(c10::kComplexFloat),
      "kAotInductorComplexFloat != c10::kComplexFloat");
  AOT_INDUCTOR_CHECK(
      kAotInductorComplexDouble == int(c10::kComplexDouble),
      "kAotInductorComplexDouble != c10::kComplexDouble");
  AOT_INDUCTOR_CHECK(
      kAotInductorBool == int(c10::kBool), "kAotInductorBool != c10::kBool");
  AOT_INDUCTOR_CHECK(
      kAotInductorQInt8 == int(c10::kQInt8),
      "kAotInductorQInt8 != c10::kQInt8");
  AOT_INDUCTOR_CHECK(
      kAotInductorQUInt8 == int(c10::kQUInt8),
      "kAotInductorQUInt8 != c10::kQUInt8");
  AOT_INDUCTOR_CHECK(
      kAotInductorQInt32 == int(c10::kQInt32),
      "kAotInductorQInt32 != c10::kQInt32");
  AOT_INDUCTOR_CHECK(
      kAotInductorBFloat16 == int(c10::kBFloat16),
      "kAotInductorBFloat16 != c10::kBFloat16");

  // Should we simply codegen all sizes and strides, and get rid of sizes and
  // strides from AotInductorTensor?
  sizes_manager = new std::vector<std::vector<int64_t>>;
  strides_manager = new std::vector<std::vector<int64_t>>;
}

void aot_inductor_destroy() {
  delete sizes_manager;
  delete strides_manager;
}

void aot_inductor_tensor_free(AotInductorTensor aot_tensor) {
  DeleterFnPtr deleter = DeleterFnPtr(aot_tensor.deleter);
  if (deleter) {
    deleter(aot_tensor.data_ptr);
  }
}

AotInductorTensor aten_tensor_to_aot_tensor(void* aten_tensor, char keep_live) {
  at::Tensor* t = static_cast<at::Tensor*>(aten_tensor);
  void* data_ptr = t->data_ptr();
  DeleterFnPtr deleter = nullptr;

  // Input and output tensors are managed by external runtime, so no RC for them
  if (!keep_live) {
    // The storage is now owned by AotInductorTensor and will not be deleted by
    // at::Tensor. The codegen-ed aot_inductor_tensor_free will free the device
    // storage.
    c10::DataPtr& mutable_ptr =
        t->storage().unsafeGetStorageImpl()->mutable_data_ptr();
    AOT_INDUCTOR_CHECK(
        mutable_ptr.get() == mutable_ptr.get_context() ||
            !mutable_ptr.get_context(),
        "Expect data_ and context_ to be the same in a UniqueVoidPtr")
    deleter = mutable_ptr.get_deleter();
    bool succeeded =
        mutable_ptr.compare_exchange_deleter(deleter, &deleteNothing);
    AOT_INDUCTOR_CHECK(
        succeeded,
        "Unexpected deleter function on storage, could not swap function");
  }

  AotInductorTensor result = {
      data_ptr,
      (void*)deleter,
      convert_to_aot_inductor_device(t->device()),
      AotInductorScalarType(t->scalar_type()),
      t->dim(),
      keep_live ? t->sizes().data() : register_sizes(t->sizes()),
      keep_live ? t->strides().data() : register_strides(t->strides()),
      t->storage_offset(),
  };
  return result;
}

// aten_tensor is allocated by the caller and passed in as a pointer
void aot_tensor_to_aten_tensor(
    AotInductorTensor aot_tensor,
    void* aten_tensor) {
  c10::TensorOptions options =
      c10::TensorOptions()
          .device(convert_to_c10_device(aot_tensor.device))
          .dtype(c10::ScalarType(aot_tensor.dtype));

  // aot_tensor.data_ptr is managed by the runtime here, so create an aten
  // tensor without deleter here
  (*static_cast<at::Tensor*>(aten_tensor)) = at::from_blob(
      aot_tensor.data_ptr,
      c10::IntArrayRef(aot_tensor.sizes, aot_tensor.ndim),
      c10::IntArrayRef(aot_tensor.strides, aot_tensor.ndim),
      options);
}

AotInductorTensor aot_inductor_empty_strided(
    int64_t ndim,
    int64_t* sizes_ptr,
    int64_t* strides_ptr,
    AotInductorDevice device,
    AotInductorScalarType type) {
  c10::IntArrayRef sizes(sizes_ptr, ndim);
  c10::IntArrayRef strides(strides_ptr, ndim);
  // TODO: directly call the caching allocator
  at::Tensor result = at::empty_strided(
      sizes,
      strides,
      c10::TensorOptions(convert_to_c10_device(device))
          .dtype(c10::ScalarType(type)));
  return aten_tensor_to_aot_tensor(&result, false);
}

AotInductorTensor aot_inductor_as_strided(
    AotInductorTensor self,
    int64_t ndim,
    int64_t* sizes_ptr,
    int64_t* strides_ptr,
    int64_t offset) {
  // See as_strided_tensorimpl
  c10::IntArrayRef new_sizes(sizes_ptr, ndim);
  c10::IntArrayRef new_strides(strides_ptr, ndim);
  c10::IntArrayRef old_sizes(self.sizes, self.ndim);
  c10::IntArrayRef old_strides(self.strides, self.ndim);
  AotInductorTensor result = {
      self.data_ptr,
      self.deleter,
      self.device,
      self.dtype,
      ndim,
      new_sizes == old_sizes ? self.sizes : register_sizes(new_sizes),
      new_strides == old_strides ? self.strides : register_strides(new_strides),
      offset,
  };
  return result;
}

AotInductorTensor aot_inductor_addmm_out(
    AotInductorTensor out,
    AotInductorTensor self,
    AotInductorTensor mat1,
    AotInductorTensor mat2,
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
  return aten_tensor_to_aot_tensor(&out_tensor, false);
}
