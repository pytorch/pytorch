#include <torch/csrc/inductor/aot_inductor_tensor.h>

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
std::unordered_map<AotInductorScalarType, c10::ScalarType>
    AotInductorScalarType_to_C10ScalarType = {
        {kAotInductorByte, c10::kByte},
        {kAotInductorChar, c10::kChar},
        {kAotInductorShort, c10::kShort},
        {kAotInductorInt, c10::kInt},
        {kAotInductorLong, c10::kLong},
        {kAotInductorHalf, c10::kHalf},
        {kAotInductorFloat, c10::kFloat},
        {kAotInductorDouble, c10::kDouble},
        {kAotInductorComplexHalf, c10::kComplexHalf},
        {kAotInductorComplexFloat, c10::kComplexFloat},
        {kAotInductorComplexDouble, c10::kComplexDouble},
        {kAotInductorBool, c10::kBool},
        {kAotInductorBFloat16, c10::kBFloat16},
};

std::unordered_map<c10::ScalarType, AotInductorScalarType>
    C10ScalarType_to_AotInductorScalarType = {
        {c10::kByte, kAotInductorByte},
        {c10::kChar, kAotInductorChar},
        {c10::kShort, kAotInductorShort},
        {c10::kInt, kAotInductorInt},
        {c10::kLong, kAotInductorLong},
        {c10::kHalf, kAotInductorHalf},
        {c10::kFloat, kAotInductorFloat},
        {c10::kDouble, kAotInductorDouble},
        {c10::kComplexHalf, kAotInductorComplexHalf},
        {c10::kComplexFloat, kAotInductorComplexFloat},
        {c10::kComplexDouble, kAotInductorComplexDouble},
        {c10::kBool, kAotInductorBool},
        {c10::kBFloat16, kAotInductorBFloat16},
};

std::unordered_map<AotInductorDeviceType, c10::DeviceType>
    AotInductorDeviceType_to_C10DeviceType = {
        {kAotInductorCPU, c10::kCPU},
        {kAotInductorCUDA, c10::kCUDA},
};

std::unordered_map<c10::DeviceType, AotInductorDeviceType>
    C10DeviceType_to_AotInductorDeviceType = {
        {c10::kCPU, kAotInductorCPU},
        {c10::kCUDA, kAotInductorCUDA},
};

AotInductorScalarType convert_to_aot_inductor_scalar_type(
    c10::ScalarType type) {
  TORCH_CHECK(
      C10ScalarType_to_AotInductorScalarType.count(type) != 0,
      "ScalarType not supported by AotInductor");
  return C10ScalarType_to_AotInductorScalarType.at(type);
}

c10::ScalarType convert_to_c10_scalar_type(AotInductorScalarType type) {
  TORCH_CHECK(
      AotInductorScalarType_to_C10ScalarType.count(type) != 0,
      "ScalarType not supported by AotInductor");
  return AotInductorScalarType_to_C10ScalarType.at(type);
}

AotInductorDevice convert_to_aot_inductor_device(c10::Device device) {
  TORCH_CHECK(
      C10DeviceType_to_AotInductorDeviceType.count(device.type()) != 0,
      "DeviceType not supported by AotInductor");
  AotInductorDevice result{
      C10DeviceType_to_AotInductorDeviceType.at(device.type()), device.index()};
  return result;
}

c10::Device convert_to_c10_device(AotInductorDevice device) {
  TORCH_CHECK(
      AotInductorDeviceType_to_C10DeviceType.count(device.device_type) != 0,
      "DeviceType not supported by AotInductor");
  c10::Device result{
      AotInductorDeviceType_to_C10DeviceType.at(device.device_type),
      device.device_id};
  return result;
}

std::vector<std::vector<int64_t>> sizes_cache;
std::vector<std::vector<int64_t>> strides_cache;

const int64_t* register_sizes(c10::IntArrayRef sizes) {
  sizes_cache.emplace_back(sizes.vec());
  return sizes_cache[sizes_cache.size() - 1].data();
}

const int64_t* register_strides(c10::IntArrayRef strides) {
  strides_cache.emplace_back(strides.vec());
  return strides_cache[strides_cache.size() - 1].data();
}

} // namespace

AotInductorTensor convert_to_aot_inductor_tensor(void* aten_tensor) {
  at::Tensor* t = static_cast<at::Tensor*>(aten_tensor);

  AotInductorTensor result = {
      t->data_ptr(),
      convert_to_aot_inductor_device(t->device()),
      convert_to_aot_inductor_scalar_type(t->scalar_type()),
      t->dim(),
      register_sizes(t->sizes()),
      register_strides(t->strides())};
  return result;
}

void convert_to_aten_tensor(
    AotInductorTensor inductor_tensor,
    void* aten_tensor) {
  c10::TensorOptions options =
      c10::TensorOptions()
          .device(convert_to_c10_device(inductor_tensor.device))
          .dtype(convert_to_c10_scalar_type(inductor_tensor.type));

  at::Tensor* t = static_cast<at::Tensor*>(aten_tensor);
  (*t) = at::from_blob(
      inductor_tensor.data_ptr,
      c10::IntArrayRef(inductor_tensor.sizes, inductor_tensor.ndim),
      c10::IntArrayRef(inductor_tensor.strides, inductor_tensor.ndim),
      options);
}

AotInductorTensor aot_inductor_empty_strided(
    int64_t dim,
    int64_t* sizes_ptr,
    int64_t* strides_ptr,
    AotInductorDevice device,
    AotInductorScalarType type) {
  c10::ArrayRef sizes(sizes_ptr, dim);
  c10::ArrayRef strides(strides_ptr, dim);
  at::Tensor result = at::empty_strided(
      sizes,
      strides,
      c10::TensorOptions(convert_to_c10_device(device))
          .dtype(convert_to_c10_scalar_type(type)));
  return convert_to_aot_inductor_tensor(&result);
}

AotInductorTensor aot_inductor_as_strided(
    AotInductorTensor self,
    int64_t* sizes_ptr,
    int64_t* strides_ptr,
    int64_t offset) {
  c10::ArrayRef sizes(sizes_ptr, self.ndim);
  c10::ArrayRef strides(strides_ptr, self.ndim);
  at::Tensor self_tensor;
  convert_to_aten_tensor(self, &self_tensor);
  at::Tensor result = at::as_strided(self_tensor, sizes, strides, offset);
  return convert_to_aot_inductor_tensor(&result);
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
  convert_to_aten_tensor(out, &out_tensor);
  convert_to_aten_tensor(self, &self_tensor);
  convert_to_aten_tensor(mat1, &mat1_tensor);
  convert_to_aten_tensor(mat2, &mat2_tensor);
  at::addmm_out(out_tensor, self_tensor, mat1_tensor, mat2_tensor, beta, alpha);
  return convert_to_aot_inductor_tensor(&out_tensor);
}
