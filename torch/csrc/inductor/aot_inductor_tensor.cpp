#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <torch/csrc/inductor/aot_inductor_tensor.h>
#include <torch/csrc/inductor/inductor_ops.h>
#include <cstdint>
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

size_t get_dtype_bytes(AotInductorScalarType dtype) {
  switch (dtype) {
    case kAotInductorBool:
    case kAotInductorByte:
    case kAotInductorChar:
    case kAotInductorQInt8:
    case kAotInductorQUInt8:
      return 1;
    case kAotInductorShort:
    case kAotInductorHalf:
    case kAotInductorBFloat16:
      return 2;
    case kAotInductorInt:
    case kAotInductorFloat:
    case kAotInductorComplexHalf:
    case kAotInductorQInt32:
      return 4;
    case kAotInductorLong:
    case kAotInductorDouble:
    case kAotInductorComplexFloat:
      return 8;
    case kAotInductorComplexDouble:
      return 16;
    default:
      AOT_INDUCTOR_CHECK(false, "Unkown AotInductorScalarType")
      return 0;
  }
}

thread_local std::vector<std::vector<int64_t>>* sizes_manager_;
thread_local std::vector<std::vector<int64_t>>* strides_manager_;
#ifndef NDEBUG
thread_local std::unordered_set<void*>* live_buffer_;
#endif

const int64_t* register_sizes(c10::IntArrayRef sizes) {
  sizes_manager_->emplace_back(sizes.vec());
  return sizes_manager_->at(sizes_manager_->size() - 1).data();
}

const int64_t* register_strides(c10::IntArrayRef strides) {
  strides_manager_->emplace_back(strides.vec());
  return strides_manager_->at(strides_manager_->size() - 1).data();
}

using DeleterFnPtr = void (*)(void*);

void deleteNothing(void*) {}

AotInductorTensor aten_tensor_to_aot_tensor(void* aten_tensor) {
  // Here, aten_tensor was created as an internel buffer. The AOTInductor
  // runtime should manage the lifetime of that buffer. Because AOTInductor
  // codegen knows the lifetime of all intermediate buffers, we don't need to do
  // RC on any AotInductorTensor. Instead, we just need to make sure we codegen
  // aot_inductor_free call for every allocated intermediate buffer after
  // its life ends.
  //
  // One complication here is we still need to call into aten ops by passing in
  // aten tensors, and aten tensor's memory management may accidentally free up
  // the underlying storage if we don't get RC right. To simplify the problem,
  // we just set the deleter of those cross-border aten tensors to
  // deleteNothing. Then aten ops can keep doing its RC but the underlying
  // storage won't be freed until aot_inductor_free is called. See the
  // implementation of aot_tensor_to_aten_tensor.
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
  }

  AotInductorTensor result = {
      data_ptr,
      (void*)deleter,
      convert_to_aot_inductor_device(t->device()),
      AotInductorScalarType(t->scalar_type()),
      t->dim(),
      register_sizes(t->sizes()),
      register_strides(t->strides()),
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
  // tensor with an empty deleter
  (*static_cast<at::Tensor*>(aten_tensor)) =
      at::for_blob(
          aot_tensor.data_ptr,
          c10::IntArrayRef(aot_tensor.sizes, aot_tensor.ndim))
          .strides(c10::IntArrayRef(aot_tensor.strides, aot_tensor.ndim))
          .storage_offset(aot_tensor.storage_offset)
          .context(aot_tensor.data_ptr, deleteNothing)
          .options(options)
          .make_tensor();
}

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
  sizes_manager_ = new std::vector<std::vector<int64_t>>;
  strides_manager_ = new std::vector<std::vector<int64_t>>;
#ifndef NDEBUG
  live_buffer_ = new std::unordered_set<void*>;
#endif
}

void aot_inductor_destroy() {
  delete sizes_manager_;
  delete strides_manager_;
#ifndef NDEBUG
  AOT_INDUCTOR_CHECK(
      live_buffer_->size() == 0, "Live intermediate buffer leaking");
  delete live_buffer_;
#endif
}

void aot_inductor_free(AotInductorTensor aot_tensor) {
  if (aot_tensor.deleter == nullptr || aot_tensor.deleter == deleteNothing)
    return;

  DeleterFnPtr deleter = DeleterFnPtr(aot_tensor.deleter);
#ifndef NDEBUG
  live_buffer_->erase(aot_tensor.data_ptr);
#endif
  deleter(aot_tensor.data_ptr);
}

void* aot_inductor_data_ptr(AotInductorTensor aot_tensor) {
  // aot_tensor.storage_offset is in the number of elements, need to turn it
  // into bytes
  char* base = static_cast<char*>(aot_tensor.data_ptr);
  size_t offset_in_bytes =
      aot_tensor.storage_offset * get_dtype_bytes(aot_tensor.dtype);
  return (void*)(base + offset_in_bytes);
}

AotInductorTensor convert_input_output_to_aot_tensor(void* aten_tensor) {
  // Input and output tensors are managed by the external runtime, so they are
  // kept live in the AOTInductor runtime, and thus their corresponding
  // AotInductorTensor will have their deleter as deleteNothing.
  at::Tensor* t = static_cast<at::Tensor*>(aten_tensor);
  AotInductorTensor result = {
      t->data_ptr(),
      (void*)deleteNothing,
      convert_to_aot_inductor_device(t->device()),
      AotInductorScalarType(t->scalar_type()),
      t->dim(),
      t->sizes().data(),
      t->strides().data(),
      t->storage_offset(),
  };
  return result;
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
  at::Tensor aten_result = at::empty_strided(
      sizes,
      strides,
      c10::TensorOptions(convert_to_c10_device(device))
          .dtype(c10::ScalarType(type)));
  AotInductorTensor result = aten_tensor_to_aot_tensor(&aten_result);
#ifndef NDEBUG
  live_buffer_->insert(result.data_ptr);
#endif
  return result;
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
  return aten_tensor_to_aot_tensor(&out_tensor);
}

AotInductorTensor aot_inductor__addmm_activation(
    AotInductorTensor self,
    AotInductorTensor mat1,
    AotInductorTensor mat2,
    float beta,
    float alpha,
    bool use_gelu) {
  at::Tensor self_tensor;
  at::Tensor mat1_tensor;
  at::Tensor mat2_tensor;
  aot_tensor_to_aten_tensor(self, &self_tensor);
  aot_tensor_to_aten_tensor(mat1, &mat1_tensor);
  aot_tensor_to_aten_tensor(mat2, &mat2_tensor);
  // When there is
  at::Tensor aten_result = at::_addmm_activation(
      self_tensor, mat1_tensor, mat2_tensor, beta, alpha, use_gelu);
  AotInductorTensor result = aten_tensor_to_aot_tensor(&aten_result);
#ifndef NDEBUG
  live_buffer_->insert(result.data_ptr);
#endif
  return result;
}

AotInductorTensor aot_inductor_bmm_out(
    AotInductorTensor out,
    AotInductorTensor self,
    AotInductorTensor mat2) {
  at::Tensor out_tensor;
  at::Tensor self_tensor;
  at::Tensor mat2_tensor;
  aot_tensor_to_aten_tensor(out, &out_tensor);
  aot_tensor_to_aten_tensor(self, &self_tensor);
  aot_tensor_to_aten_tensor(mat2, &mat2_tensor);
  // TODO: directly call cublas
  at::bmm_out(out_tensor, self_tensor, mat2_tensor);
  return aten_tensor_to_aot_tensor(&out_tensor);
}

AotInductorTensor aot_inductor_copy_(
    AotInductorTensor src,
    AotInductorTensor dst) {
  at::Tensor src_tensor;
  at::Tensor dst_tensor;
  aot_tensor_to_aten_tensor(src, &src_tensor);
  aot_tensor_to_aten_tensor(dst, &dst_tensor);
  dst_tensor.copy_(src_tensor);
  return aten_tensor_to_aot_tensor(&dst_tensor);
}

AotInductorTensor aot_inductor_mm_out(
    AotInductorTensor out,
    AotInductorTensor self,
    AotInductorTensor mat2) {
  at::Tensor out_tensor;
  at::Tensor self_tensor;
  at::Tensor mat2_tensor;
  aot_tensor_to_aten_tensor(out, &out_tensor);
  aot_tensor_to_aten_tensor(self, &self_tensor);
  aot_tensor_to_aten_tensor(mat2, &mat2_tensor);
  // TODO: directly call cublas
  at::mm_out(out_tensor, self_tensor, mat2_tensor);
  return aten_tensor_to_aot_tensor(&out_tensor);
}
