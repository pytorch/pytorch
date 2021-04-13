#include <ATen/InferSize.h>
#include <ATen/Utils.h>
#include <c10/util/accumulate.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

#include <ATen/native/vulkan/Vulkan.h>
#include <ATen/native/vulkan/VulkanCommon.h>
#include <ATen/native/vulkan/VulkanConvolution.h>
#include <ATen/native/vulkan/VulkanOps.h>

#include <iostream>
#include <limits>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace detail {

void upsample_nearest2d(
    VulkanTensor& output,
    const VulkanTensor& input,
    int64_t IH,
    int64_t IW,
    int64_t OH,
    int64_t OW,
    int64_t IN,
    int64_t IC,
    float scaleH,
    float scaleW) {
  auto device = context().device();
  int64_t C = IN * IC;
  struct ConstBlock {
    float scaleX;
    float scaleY;
  };
  ConstBlock cb{scaleW,
                scaleH};
  VBuffer constBuffer = makeUniformConstBuffer((void*)&cb, sizeof(cb));

  VkDescriptorSetLayout descriptorSetLayout{};
  VkDescriptorPool descriptorPool{};
  VkDescriptorSet descriptorSet{};
  std::vector<VkDescriptorType> descriptorTypes{
      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
  createDescriptorSetLayoutSinglePool(
      device,
      descriptorTypes,
      &descriptorSetLayout,
      &descriptorPool,
      &descriptorSet);

  output.image()->bindStorageImage(descriptorSet, 0);
  input.image()->bindShaderRead(descriptorSet, 1);
  constBuffer.bind(descriptorSet, 2);

  WorkGroupSize workGroupSize{8, 8, 1};
  auto& computeUnit = context().computeUnitFactory().get(
      GLSL_SPV(upsample_nearest2d), descriptorSetLayout, workGroupSize);
  computeUnit.createCommandBuffer(descriptorSet);
  input.image()->addImageMemoryBarrierToShaderRead(computeUnit.commandBuffer());
  computeUnit.dispatchCommandBuffer(OW, OH, C, workGroupSize);
  computeUnit.endCommandBuffer();
  computeUnit.submitAndWaitCommandBuffer();
  vkDestroyDescriptorPool(device, descriptorPool, nullptr);
  vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
}

VulkanTensor reshape_copy(
    const VulkanTensor& input,
    std::vector<int64_t> shape) {
  input.sync_image_to_buffer();
  VulkanTensor output{infer_size(shape, input.numel())};
  copy_buffer_to_buffer(
      *(input.buffer()), *(output.buffer()), input.buffer()->sizeBytes());
  return output;
}

VulkanTensor cat(
    VulkanTensor& output,
    ArrayRef<VulkanTensor> inputs,
    int64_t dim) {
  VkDeviceSize outputOffset = 0;
  for (const auto& input : inputs) {
    input.sync_image_to_buffer();
    const auto sizeBytes = sizeof(float) * input.numel();
    copy_buffer_to_buffer(
        *(input.buffer()), *(output.buffer()), sizeBytes, 0, outputOffset);
    outputOffset += sizeBytes;
  }
  return output;
}

void adaptive_avg_pool2d(
    VulkanTensor& output,
    const VulkanTensor& input,
    const int64_t IH,
    const int64_t IW,
    const int64_t OH,
    const int64_t OW,
    const int64_t IN,
    const int64_t IC) {
  auto device = context().device();
  int64_t C = IN * IC;

  VkDescriptorSetLayout descriptorSetLayout{};
  VkDescriptorPool descriptorPool{};
  VkDescriptorSet descriptorSet{};
  std::vector<VkDescriptorType> descriptorTypes{
      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
  createDescriptorSetLayoutSinglePool(
      device,
      descriptorTypes,
      &descriptorSetLayout,
      &descriptorPool,
      &descriptorSet);

  output.image()->bindStorageImage(descriptorSet, 0);
  input.image()->bindShaderRead(descriptorSet, 1);

  WorkGroupSize workGroupSize{8, 8, 1};
  auto& computeUnit = context().computeUnitFactory().get(
      GLSL_SPV(adaptive_avg_pool2d), descriptorSetLayout, workGroupSize);
  computeUnit.createCommandBuffer(descriptorSet);
  input.image()->addImageMemoryBarrierToShaderRead(computeUnit.commandBuffer());
  computeUnit.dispatchCommandBuffer(OW, OH, C, workGroupSize);
  computeUnit.endCommandBuffer();
  computeUnit.submitAndWaitCommandBuffer();
  vkDestroyDescriptorPool(device, descriptorPool, nullptr);
  vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
}

void max_pool2d(
    VulkanTensor& output,
    const VulkanTensor& input,
    const int iH,
    const int iW,
    const int oH,
    const int oW,
    const int _n,
    const int _c,
    const int kH,
    const int kW,
    const int dH,
    const int dW,
    const int padH,
    const int padW,
    const int dilationH,
    const int dilationW) {
  auto device = context().device();
  const auto c = _n * _c;
  struct ConstBlock {
    int32_t inputSize[4];
    int32_t outputSize[4];
    int32_t kernelSize[2];
    int32_t stride[2];
    int32_t padding[2];
    int32_t dilate[2];
  };
  ConstBlock cb{
      {iW, iH, c, 0},
      {oW, oH, c, 0},
      {kW, kH},
      {dW, dH},
      {padW, padH},
      {dilationW, dilationH},
  };
  VBuffer constBuffer = makeUniformConstBuffer((void*)&cb, sizeof(cb));

  VkDescriptorSetLayout descriptorSetLayout{};
  VkDescriptorPool descriptorPool{};
  VkDescriptorSet descriptorSet{};
  std::vector<VkDescriptorType> descriptorTypes{
      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
  createDescriptorSetLayoutSinglePool(
      device,
      descriptorTypes,
      &descriptorSetLayout,
      &descriptorPool,
      &descriptorSet);

  output.image()->bindStorageImage(descriptorSet, 0);
  input.image()->bindShaderRead(descriptorSet, 1);
  constBuffer.bind(descriptorSet, 2);

  WorkGroupSize workGroupSize{8, 8, 1};
  auto& computeUnit = context().computeUnitFactory().get(
      GLSL_SPV(max_pool2d), descriptorSetLayout, workGroupSize);
  computeUnit.createCommandBuffer(descriptorSet);
  input.image()->addImageMemoryBarrierToShaderRead(computeUnit.commandBuffer());
  computeUnit.dispatchCommandBuffer(oW, oH, c, workGroupSize);
  computeUnit.endCommandBuffer();
  computeUnit.submitAndWaitCommandBuffer();
  vkDestroyDescriptorPool(device, descriptorPool, nullptr);
  vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
}

void avg_pool2d(
    VulkanTensor& output,
    const VulkanTensor& input,
    const int iH,
    const int iW,
    const int oH,
    const int oW,
    const int _n,
    const int _c,
    const int kH,
    const int kW,
    const int dH,
    const int dW,
    const int padH,
    const int padW) {
  auto device = context().device();
  const auto c = _n * _c;
  struct ConstBlock {
    int32_t kernelSize[2];
    int32_t stride[2];
    int32_t padding[2];
  };
  ConstBlock cb{
      {kW, kH},
      {dW, dH},
      {padW, padH},
  };
  VBuffer constBuffer = makeUniformConstBuffer((void*)&cb, sizeof(cb));

  VkDescriptorSetLayout descriptorSetLayout{};
  VkDescriptorPool descriptorPool{};
  VkDescriptorSet descriptorSet{};
  std::vector<VkDescriptorType> descriptorTypes{
      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
  createDescriptorSetLayoutSinglePool(
      device,
      descriptorTypes,
      &descriptorSetLayout,
      &descriptorPool,
      &descriptorSet);

  output.image()->bindStorageImage(descriptorSet, 0);
  input.image()->bindShaderRead(descriptorSet, 1);
  constBuffer.bind(descriptorSet, 2);

  WorkGroupSize workGroupSize{8, 8, 1};
  auto& computeUnit = context().computeUnitFactory().get(
      GLSL_SPV(avg_pool2d), descriptorSetLayout, workGroupSize);
  computeUnit.createCommandBuffer(descriptorSet);
  input.image()->addImageMemoryBarrierToShaderRead(computeUnit.commandBuffer());
  computeUnit.dispatchCommandBuffer(oW, oH, c, workGroupSize);
  computeUnit.endCommandBuffer();
  computeUnit.submitAndWaitCommandBuffer();
  vkDestroyDescriptorPool(device, descriptorPool, nullptr);
  vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
}

VulkanTensor transpose(
    const VulkanTensor& input,
    const int64_t dim0,
    const int64_t dim1) {
  const auto idim = input.dim();
  TORCH_INTERNAL_ASSERT(
      idim <= 6, "Vulkan transpose is implemented only for dim <= 6");
  auto device = context().device();
  struct ConstBlock {
    int32_t istrides[8];
    int32_t ostrides[8];
    int32_t odims[8];
    int32_t storageOffset;
  };

  auto isizes = input.sizes();
  auto osizes = isizes;
  std::swap(osizes[dim0], osizes[dim1]);
  VulkanTensor output{osizes};
  output.allocate_storage();

  std::array<int32_t, 8> idims8;
  idims8.fill(1);
  std::array<int32_t, 8> odims8;
  odims8.fill(1);
  std::copy(isizes.cbegin(), isizes.cend(), idims8.end() - idim);
  std::copy(osizes.cbegin(), osizes.cend(), odims8.end() - idim);
  std::array<int32_t, 8> istrides8;
  istrides8.fill(1);
  std::array<int32_t, 8> ostrides8;
  ostrides8.fill(1);
  for (int i = 6; i >= 0; --i) {
    istrides8[i] = idims8[i + 1] * istrides8[i + 1];
    ostrides8[i] = odims8[i + 1] * ostrides8[i + 1];
  }
  std::swap(istrides8[8 - idim + dim0], istrides8[8 - idim + dim1]);

  ConstBlock cb{};
  std::copy(istrides8.cbegin(), istrides8.cend(), std::begin(cb.istrides));
  std::copy(ostrides8.cbegin(), ostrides8.cend(), std::begin(cb.ostrides));
  std::copy(odims8.cbegin(), odims8.cend(), std::begin(cb.odims));
  cb.storageOffset = 0;

  VBuffer constBuffer = makeUniformConstBuffer((void*)&cb, sizeof(cb));

  VkDescriptorSetLayout descriptorSetLayout{};
  VkDescriptorPool descriptorPool{};
  VkDescriptorSet descriptorSet{};
  std::vector<VkDescriptorType> descriptorTypes{
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
  createDescriptorSetLayoutSinglePool(
      device,
      descriptorTypes,
      &descriptorSetLayout,
      &descriptorPool,
      &descriptorSet);

  output.buffer()->bind(descriptorSet, 0);
  input.buffer()->bind(descriptorSet, 1);
  constBuffer.bind(descriptorSet, 2);

  WorkGroupSize workGroupSize{8, 8, 1};
  auto& computeUnit = context().computeUnitFactory().get(
      GLSL_SPV(permute), descriptorSetLayout, workGroupSize);
  computeUnit.createCommandBuffer(descriptorSet);
  input.buffer()->addBufferMemoryBarrier(
      computeUnit.commandBuffer(), 0, input.buffer()->sizeBytes());
  computeUnit.dispatchCommandBuffer(
      odims8[6] * odims8[7],
      odims8[4] * odims8[5],
      odims8[2] * odims8[3],
      workGroupSize);
  computeUnit.endCommandBuffer();
  computeUnit.submitAndWaitCommandBuffer();
  vkDestroyDescriptorPool(device, descriptorPool, nullptr);
  vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
  return output;
}

VulkanTensor slice(
    const VulkanTensor& input,
    const int64_t dim,
    const int64_t _start,
    const int64_t _end,
    const int64_t step) {
  const auto isizes = input.sizes();
  auto osizes = isizes;
  auto start = _start;
  auto end = _end;
  if (start < 0) {
    start += isizes[dim];
  }
  if (end < 0) {
    end += isizes[dim];
  }
  if (start < 0) {
    start = 0;
  } else if (start >= isizes[dim]) {
    start = isizes[dim];
  }
  if (end < start) {
    end = start;
  } else if (end >= isizes[dim]) {
    end = isizes[dim];
  }
  const auto len = end - start;
  osizes[dim] = (len + step - 1) / step;

  VulkanTensor output{osizes};
  output.allocate_storage();

  auto idim = input.dim();
  std::array<int32_t, 8> idims8;
  idims8.fill(1);
  std::copy(isizes.cbegin(), isizes.cend(), idims8.end() - idim);
  std::array<int32_t, 8> istrides8;
  istrides8.fill(1);
  for (int i = 6; i >= 0; --i) {
    istrides8[i] = idims8[i + 1] * istrides8[i + 1];
  }

  std::array<int32_t, 8> odims8 = idims8;
  std::array<int32_t, 8> ostrides8 = istrides8;

  ostrides8[8 - idim + dim] *= step;
  auto storage_offset = start * istrides8[8 - idim + dim];

  auto device = context().device();
  struct ConstBlock {
    int32_t istrides[8];
    int32_t ostrides[8];
    int32_t odims[8];
    int32_t storageOffset;
  };

  ConstBlock cb{};
  std::copy(istrides8.cbegin(), istrides8.cend(), std::begin(cb.istrides));
  std::copy(ostrides8.cbegin(), ostrides8.cend(), std::begin(cb.ostrides));
  std::copy(odims8.cbegin(), odims8.cend(), std::begin(cb.odims));
  cb.storageOffset = storage_offset;

  VBuffer constBuffer = makeUniformConstBuffer((void*)&cb, sizeof(cb));

  VkDescriptorSetLayout descriptorSetLayout{};
  VkDescriptorPool descriptorPool{};
  VkDescriptorSet descriptorSet{};
  std::vector<VkDescriptorType> descriptorTypes{
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
  createDescriptorSetLayoutSinglePool(
      device,
      descriptorTypes,
      &descriptorSetLayout,
      &descriptorPool,
      &descriptorSet);

  output.buffer()->bind(descriptorSet, 0);
  input.buffer()->bind(descriptorSet, 1);
  constBuffer.bind(descriptorSet, 2);

  WorkGroupSize workGroupSize{8, 8, 1};
  auto& computeUnit = context().computeUnitFactory().get(
      GLSL_SPV(permute), descriptorSetLayout, workGroupSize);
  computeUnit.createCommandBuffer(descriptorSet);
  input.buffer()->addBufferMemoryBarrier(
      computeUnit.commandBuffer(), 0, input.buffer()->sizeBytes());
  computeUnit.dispatchCommandBuffer(
      odims8[6] * odims8[7],
      odims8[4] * odims8[5],
      odims8[2] * odims8[3],
      workGroupSize);
  computeUnit.endCommandBuffer();
  computeUnit.submitAndWaitCommandBuffer();
  vkDestroyDescriptorPool(device, descriptorPool, nullptr);
  vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
  return output;
}

void add(
    VulkanTensor& output,
    const VulkanTensor& input0,
    const VulkanTensor& input1,
    float alpha) {
  auto odim = output.dim();
  TORCH_INTERNAL_ASSERT(
      odim <= 4, "Vulkan add is implemented for dim <= 4, output dim > 4");
  auto i0dim = input0.dim();
  TORCH_INTERNAL_ASSERT(
      i0dim <= 4, "Vulkan add is implemented for dim <= 4, input0 dim > 4");
  auto i1dim = input1.dim();
  TORCH_INTERNAL_ASSERT(
      i1dim <= 4, "Vulkan add is implemented for dim <= 4, input1 dim > 4");

  auto os = output.sizes();
  auto i0s = input0.sizes();
  auto i1s = input1.sizes();

  std::array<int64_t, 4> os4 = {1, 1, 1, 1};
  std::copy(os.begin(), os.end(), os4.end() - odim);
  std::array<int64_t, 4> i0s4 = {1, 1, 1, 1};
  std::copy(i0s.cbegin(), i0s.cend(), i0s4.end() - i0dim);
  std::array<int64_t, 4> i1s4 = {1, 1, 1, 1};
  std::copy(i1s.cbegin(), i1s.cend(), i1s4.end() - i1dim);

  TORCH_INTERNAL_ASSERT(
      (os4 == i0s4) && (i0s4 == i1s4),
      "Vulkan add expects the same dimensions for all operands");

  auto C = os4[0] * os4[1];
  auto H = os4[2];
  auto W = os4[3];

  auto device = context().device();
  struct ConstBlock {
    float alpha;
  };
  ConstBlock cb{alpha};
  VBuffer constBuffer = makeUniformConstBuffer((void*)&cb, sizeof(cb));

  VkDescriptorSetLayout descriptorSetLayout{};
  VkDescriptorPool descriptorPool{};
  VkDescriptorSet descriptorSet{};
  std::vector<VkDescriptorType> descriptorTypes{
      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
  createDescriptorSetLayoutSinglePool(
      device,
      descriptorTypes,
      &descriptorSetLayout,
      &descriptorPool,
      &descriptorSet);

  output.image()->bindStorageImage(descriptorSet, 0);
  input0.image()->bindShaderRead(descriptorSet, 1);
  input1.image()->bindShaderRead(descriptorSet, 2);
  constBuffer.bind(descriptorSet, 3);

  WorkGroupSize workGroupSize{8, 8, 1};
  auto& computeUnit = context().computeUnitFactory().get(
      GLSL_SPV(add), descriptorSetLayout, workGroupSize);
  computeUnit.createCommandBuffer(descriptorSet);
  auto commandBuffer = computeUnit.commandBuffer();
  output.image()->addImageMemoryBarrierToGeneral(commandBuffer);
  input0.image()->addImageMemoryBarrierToShaderRead(commandBuffer);
  input1.image()->addImageMemoryBarrierToShaderRead(commandBuffer);
  computeUnit.dispatchCommandBuffer(W, H, C, workGroupSize);
  computeUnit.endCommandBuffer();
  computeUnit.submitAndWaitCommandBuffer();
  vkDestroyDescriptorPool(device, descriptorPool, nullptr);
  vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
}

void add(VulkanTensor& output, const VulkanTensor& input, const float s) {
  const auto sizes = input.sizes();

  const auto C = c10::multiply_integers(sizes.cbegin(), sizes.cend() - 2);
  const auto C_4 = UP_DIV(C, 4);
  const auto H = sizes[2];
  const auto W = sizes[3];

  auto device = context().device();
  struct ConstBlock {
    float s;
  };
  ConstBlock cb{s};
  VBuffer constBuffer = makeUniformConstBuffer((void*)&cb, sizeof(cb));

  VkDescriptorSetLayout descriptorSetLayout{};
  VkDescriptorPool descriptorPool{};
  VkDescriptorSet descriptorSet{};
  std::vector<VkDescriptorType> descriptorTypes{
      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
  createDescriptorSetLayoutSinglePool(
      device,
      descriptorTypes,
      &descriptorSetLayout,
      &descriptorPool,
      &descriptorSet);

  output.image()->bindStorageImage(descriptorSet, 0);
  input.image()->bindShaderRead(descriptorSet, 1);
  constBuffer.bind(descriptorSet, 2);

  WorkGroupSize workGroupSize{8, 8, 1};
  auto& computeUnit = context().computeUnitFactory().get(
      GLSL_SPV(add_scalar), descriptorSetLayout, workGroupSize);
  computeUnit.createCommandBuffer(descriptorSet);
  auto commandBuffer = computeUnit.commandBuffer();
  output.image()->addImageMemoryBarrierToGeneral(commandBuffer);
  input.image()->addImageMemoryBarrierToShaderRead(commandBuffer);
  computeUnit.dispatchCommandBuffer(W, H, C_4, workGroupSize);
  computeUnit.endCommandBuffer();
  computeUnit.submitAndWaitCommandBuffer();
  vkDestroyDescriptorPool(device, descriptorPool, nullptr);
  vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
}

void mul(VulkanTensor& output, const VulkanTensor& input, const float s) {
  const auto sizes = input.sizes();

  const auto C = c10::multiply_integers(sizes.cbegin(), sizes.cend() - 2);
  const auto C_4 = UP_DIV(C, 4);
  const auto H = sizes[2];
  const auto W = sizes[3];

  auto device = context().device();
  struct ConstBlock {
    float s;
  };
  ConstBlock cb{s};
  VBuffer constBuffer = makeUniformConstBuffer((void*)&cb, sizeof(cb));

  VkDescriptorSetLayout descriptorSetLayout{};
  VkDescriptorPool descriptorPool{};
  VkDescriptorSet descriptorSet{};
  std::vector<VkDescriptorType> descriptorTypes{
      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
  createDescriptorSetLayoutSinglePool(
      device,
      descriptorTypes,
      &descriptorSetLayout,
      &descriptorPool,
      &descriptorSet);

  output.image()->bindStorageImage(descriptorSet, 0);
  input.image()->bindShaderRead(descriptorSet, 1);
  constBuffer.bind(descriptorSet, 2);

  WorkGroupSize workGroupSize{8, 8, 1};
  auto& computeUnit = context().computeUnitFactory().get(
      GLSL_SPV(mul_scalar), descriptorSetLayout, workGroupSize);
  computeUnit.createCommandBuffer(descriptorSet);
  auto commandBuffer = computeUnit.commandBuffer();
  output.image()->addImageMemoryBarrierToGeneral(commandBuffer);
  input.image()->addImageMemoryBarrierToShaderRead(commandBuffer);
  computeUnit.dispatchCommandBuffer(W, H, C_4, workGroupSize);
  computeUnit.endCommandBuffer();
  computeUnit.submitAndWaitCommandBuffer();
  vkDestroyDescriptorPool(device, descriptorPool, nullptr);
  vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
}

VBuffer kernelNCHW_OCHW_repack_O4C4HWi4o4(
    const float* weights,
    const int OC,
    const int C,
    const int KH,
    const int KW) {
  const auto C_4 = UP_DIV(C, 4);
  const auto kBufSizeNumel = ALIGN_UP4(OC) * ALIGN_UP4(C) * KH * KW;
  auto size = sizeof(float) * kBufSizeNumel;
  VBuffer kernelBuffer{size};
  const int oc_4SizeNumel = KW * KH * C_4 * 16;
  auto mappedMemory = kernelBuffer.map();
  if (mappedMemory.ptr()) {
    float* basePtr = (float*)mappedMemory.ptr();
    memset(basePtr, 0, size);
    const float* src = weights;
    int ridx = 0;
    for (int oc = 0; oc < OC; ++oc) {
      int oc_4 = oc / 4;
      int oc_4_i = oc % 4;
      float* dst_oc = basePtr + oc_4 * oc_4SizeNumel;
      for (int ic = 0; ic < C; ++ic) {
        int ic_4 = ic / 4;
        int ic_4_i = ic % 4;
        float* dst_ic = dst_oc + ic_4 * KW * KH * 16;
        for (int ky = 0; ky < KH; ++ky) {
          float* dst_ky = dst_ic + ky * KW * 16;
          for (int kx = 0; kx < KW; ++kx) {
            float* dst_kx = dst_ky + kx * 16;
            dst_kx[4 * ic_4_i + oc_4_i] = src[ridx++];
          }
        }
      }
    }
  }
  mappedMemory.flushWriteToDevice();
  return kernelBuffer;
}

VBuffer bufferFromOptionalHostData(
    c10::optional<const float*> data,
    const uint32_t dataSize,
    const uint32_t bufferSize) {
  TORCH_INTERNAL_ASSERT(
      dataSize <= bufferSize,
      "buffer size(",
      bufferSize,
      ") is not enough for data(",
      dataSize,
      ")");
  const auto sizeAligned =
      ROUND_UP(bufferSize, context().limits().minStorageBufferOffsetAlignment);
  VBuffer buffer{sizeAligned};
  if (data.has_value()) {
    buffer.copy_from_host_to_device(*data, dataSize);
  } else {
    buffer.set_zeros();
  }
  return buffer;
}

VBuffer bufferZeros(const uint32_t size) {
  VBuffer buffer{size};
  buffer.set_zeros();
  return buffer;
}

void conv2d_depthwise(
    VulkanTensor& output,
    const VulkanTensor& input,
    const VulkanTensor& weight,
    const VBuffer& biasBuffer,
    const Conv2DParams& params,
    c10::optional<float> output_min,
    c10::optional<float> output_max) {
  TORCH_INTERNAL_ASSERT(params.G == params.C);
  auto osizes = output.sizes();
  TORCH_INTERNAL_ASSERT(osizes[2] == params.OH);
  TORCH_INTERNAL_ASSERT(osizes[3] == params.OW);
  struct ConstBlock {
    int32_t padding[2];
    int32_t kernelSize[2];
    int32_t stride[2];
    int32_t dilate[2];
    int32_t inputSize[4];
    int32_t outputSize[4];
    float outputMin;
    float outputMax;
  };
  ConstBlock cb{
      {safe_downcast<int32_t>(params.PX), safe_downcast<int32_t>(params.PY)},
      {safe_downcast<int32_t>(params.KW), safe_downcast<int32_t>(params.KH)},
      {safe_downcast<int32_t>(params.SX), safe_downcast<int32_t>(params.SY)},
      {safe_downcast<int32_t>(params.DX), safe_downcast<int32_t>(params.DY)},
      {safe_downcast<int32_t>(params.OW),
       safe_downcast<int32_t>(params.OH),
       safe_downcast<int32_t>(params.OC_4),
       0},
      {safe_downcast<int32_t>(params.W),
       safe_downcast<int32_t>(params.H),
       safe_downcast<int32_t>(params.C_4),
       0},
      output_min ? *output_min : -std::numeric_limits<float>::infinity(),
      output_max ? *output_max : std::numeric_limits<float>::infinity()};
  VBuffer constBuffer = makeUniformConstBuffer((void*)&cb, sizeof(cb));

  VkDescriptorSetLayout descriptorSetLayout{};
  VkDescriptorPool descriptorPool{};
  VkDescriptorSet descriptorSet{};
  std::vector<VkDescriptorType> descriptorTypes{
      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
  auto device = context().device();
  createDescriptorSetLayoutSinglePool(
      device,
      descriptorTypes,
      &descriptorSetLayout,
      &descriptorPool,
      &descriptorSet);

  output.image()->bindStorageImage(descriptorSet, 0);
  input.image()->bindShaderRead(descriptorSet, 1);
  weight.image()->bindShaderRead(descriptorSet, 2);
  biasBuffer.bind(descriptorSet, 3);
  constBuffer.bind(descriptorSet, 4);

  WorkGroupSize workGroupSize{8, 8, 1};
  auto& computeUnit = context().computeUnitFactory().get(
      GLSL_SPV(conv2d_dw_clamp), descriptorSetLayout, workGroupSize);
  computeUnit.createCommandBuffer(descriptorSet);
  auto commandBuffer = computeUnit.commandBuffer();
  output.image()->addImageMemoryBarrierToGeneral(commandBuffer);
  input.image()->addImageMemoryBarrierToShaderRead(commandBuffer);
  weight.image()->addImageMemoryBarrierToShaderRead(commandBuffer);
  computeUnit.dispatchCommandBuffer(
      params.OW, params.OH, params.OC_4, workGroupSize);
  computeUnit.endCommandBuffer();
  computeUnit.submitAndWaitCommandBuffer();

  vkDestroyDescriptorPool(device, descriptorPool, nullptr);
  vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
}

void conv2d_depthwise(
    VulkanTensor& output,
    const VulkanTensor& input,
    const VulkanTensor& weight,
    const c10::optional<const float*> bias,
    const Conv2DParams params,
    c10::optional<float> output_min,
    c10::optional<float> output_max) {
  conv2d_depthwise(
      output,
      input,
      weight,
      bufferFromOptionalHostData(
          bias,
          sizeof(float) * params.OC,
          sizeof(float) * ALIGN_UP4(params.OC)),
      params,
      output_min,
      output_max);
}

void conv2d_depthwise(
    VulkanTensor& output,
    const VulkanTensor& input,
    const float* weight,
    const c10::optional<const float*> bias,
    const Conv2DParams params,
    c10::optional<float> output_min,
    c10::optional<float> output_max) {
  VulkanTensor weightTensor{{params.OC, params.KH, params.KW}};
  weightTensor.set_data_from_host(weight);
  conv2d_depthwise(
      output,
      input,
      weightTensor,
      bufferFromOptionalHostData(
          bias,
          sizeof(float) * params.OC,
          sizeof(float) * ALIGN_UP4(params.OC)),
      params,
      output_min,
      output_max);
}

ImageSizes conv2d_prepack_weights_image_sizes(
    int64_t argOC,
    int64_t argC,
    int64_t KH,
    int64_t KW) {
  const int32_t C = safe_downcast<int32_t>(argC);
  const int32_t OC = safe_downcast<int32_t>(argOC);
  const int32_t Cup4 = ALIGN_UP4(C);
  const int32_t OC_4 = UP_DIV(OC, 4);
  const int32_t Z = safe_downcast<int32_t>(KH) * safe_downcast<int32_t>(KW);
  return {{Cup4, OC_4, Z}, {Cup4, OC_4, Z}};
}

void conv2d_prepack_weights_to_image(
    VImage& image,
    const float* weight,
    int64_t OC,
    int64_t C,
    int64_t KH,
    int64_t KW) {
  auto kernelBuffer = kernelNCHW_OCHW_repack_O4C4HWi4o4(weight, OC, C, KH, KW);
  auto OC_4 = UP_DIV(OC, 4);
  auto C_4 = UP_DIV(C, 4);

  auto expectedSizes = conv2d_prepack_weights_image_sizes(OC, C, KH, KW);
  TORCH_INTERNAL_ASSERT(
      image.sizes() == expectedSizes.imageSize,
      "Out VImage sizes do not match expected");

  struct ConstBlock {
    int32_t KWxKH;
    int32_t C_4;
  };
  ConstBlock cb{safe_downcast<int32_t>(KW * KH), safe_downcast<int32_t>(C_4)};
  VBuffer constBuffer = makeUniformConstBuffer((void*)&cb, sizeof(cb));

  VkDescriptorSetLayout descriptorSetLayout{};
  VkDescriptorPool descriptorPool{};
  VkDescriptorSet descriptorSet{};
  std::vector<VkDescriptorType> descriptorTypes{
      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
  createDescriptorSetLayoutSinglePool(
      context().device(),
      descriptorTypes,
      &descriptorSetLayout,
      &descriptorPool,
      &descriptorSet);

  image.bindStorageImage(descriptorSet, 0);
  kernelBuffer.bind(descriptorSet, 1);
  constBuffer.bind(descriptorSet, 2);

  WorkGroupSize workGroupSize{1, 1, 1};
  auto& computeUnit = context().computeUnitFactory().get(
      GLSL_SPV(KO4C4HW_to_image), descriptorSetLayout, workGroupSize);
  computeUnit.createCommandBuffer(descriptorSet);
  auto commandBuffer = computeUnit.commandBuffer();
  image.addImageMemoryBarrierToGeneral(commandBuffer);
  kernelBuffer.addBufferMemoryBarrier(
      commandBuffer, 0, kernelBuffer.sizeBytes());
  computeUnit.addMemoryBarrier(
      VK_PIPELINE_STAGE_HOST_BIT,
      VK_ACCESS_HOST_WRITE_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_READ_BIT);
  computeUnit.dispatchCommandBuffer(C_4, OC_4, KH * KW, workGroupSize);
  computeUnit.endCommandBuffer();
  computeUnit.submitAndWaitCommandBuffer();
  vkDestroyDescriptorPool(context().device(), descriptorPool, nullptr);
  vkDestroyDescriptorSetLayout(
      context().device(), descriptorSetLayout, nullptr);
}

VImage conv2d_prepack_weights_image(
    const float* weight,
    int64_t OC,
    int64_t C,
    int64_t KH,
    int64_t KW) {
  VImage image{conv2d_prepack_weights_image_sizes(OC, C, KH, KW)};
  conv2d_prepack_weights_to_image(image, weight, OC, C, KH, KW);
  return image;
}

void conv2d_prepack_weights(
    VulkanTensor& output,
    const float* weight,
    int64_t OC,
    int64_t C,
    int64_t KH,
    int64_t KW) {
  auto imageSizes = conv2d_prepack_weights_image_sizes(OC, C, KH, KW);
  conv2d_prepack_weights_to_image(
      *(output.image(imageSizes)), weight, OC, C, KH, KW);
}

void conv2d(
    VulkanTensor& output,
    const VulkanTensor& input,
    const VImage& kernelImage,
    const VBuffer& biasBuffer,
    const Conv2DParams& params,
    c10::optional<float> output_min,
    c10::optional<float> output_max) {
  TORCH_INTERNAL_ASSERT(
      params.G == 1, "Prepacked kernel VImage for non-group conv2d only");
  auto osizes = output.sizes();
  TORCH_INTERNAL_ASSERT(
      osizes[2] == params.OH,
      "Output tensor dims do not match specified conv2d params");
  TORCH_INTERNAL_ASSERT(
      osizes[3] == params.OW,
      "Output tensor dims do not match specified conv2d params");

  struct ConstBlock {
    int32_t padding[2];
    int32_t kernelSize[2];
    int32_t stride[2];
    int32_t dilate[2];
    int32_t inputSize[4];
    int32_t outputSize[4];
    float outputMin;
    float outputMax;
  };
  float outputMin =
      output_min ? *output_min : -std::numeric_limits<float>::infinity();
  float outputMax =
      output_max ? *output_max : std::numeric_limits<float>::infinity();
  ConstBlock cb{
      {safe_downcast<int32_t>(params.PX), safe_downcast<int32_t>(params.PY)},
      {safe_downcast<int32_t>(params.KW), safe_downcast<int32_t>(params.KH)},
      {safe_downcast<int32_t>(params.SX), safe_downcast<int32_t>(params.SY)},
      {safe_downcast<int32_t>(params.DX), safe_downcast<int32_t>(params.DY)},
      {safe_downcast<int32_t>(params.OW),
       safe_downcast<int32_t>(params.OH),
       safe_downcast<int32_t>(params.OC_4),
       safe_downcast<int32_t>(params.OC)},
      {safe_downcast<int32_t>(params.W),
       safe_downcast<int32_t>(params.H),
       safe_downcast<int32_t>(params.C_4),
       safe_downcast<int32_t>(params.C)},
      outputMin,
      outputMax};
  VBuffer constBuffer = makeUniformConstBuffer((void*)&cb, sizeof(cb));

  auto device = context().device();
  VkDescriptorSetLayout descriptorSetLayout{};
  VkDescriptorPool descriptorPool{};
  VkDescriptorSet descriptorSet{};
  std::vector<VkDescriptorType> descriptorTypes{
      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
  createDescriptorSetLayoutSinglePool(
      device,
      descriptorTypes,
      &descriptorSetLayout,
      &descriptorPool,
      &descriptorSet);

  output.image()->bindStorageImage(descriptorSet, 0);
  input.image()->bindShaderRead(descriptorSet, 1);
  kernelImage.bindShaderRead(descriptorSet, 2);
  biasBuffer.bind(descriptorSet, 3);
  constBuffer.bind(descriptorSet, 4);

  WorkGroupSize workGroupSize{1, 1, safe_downcast<uint32_t>(params.OC_4)};
  auto& computeUnit = context().computeUnitFactory().get(
      GLSL_SPV(conv2d_nogroup_clamp), descriptorSetLayout, workGroupSize);
  computeUnit.createCommandBuffer(descriptorSet);
  auto commandBuffer = computeUnit.commandBuffer();
  output.image()->addImageMemoryBarrierToGeneral(commandBuffer);
  input.image()->addImageMemoryBarrierToShaderRead(commandBuffer);
  kernelImage.addImageMemoryBarrierToShaderRead(commandBuffer);
  computeUnit.dispatchCommandBuffer(
      UP_DIV(params.OW, 4 * workGroupSize.x),
      UP_DIV(params.OH, workGroupSize.y),
      UP_DIV(params.OC_4, workGroupSize.z));
  computeUnit.endCommandBuffer();
  computeUnit.submitAndWaitCommandBuffer();

  vkDestroyDescriptorPool(device, descriptorPool, nullptr);
  vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
}

void conv2d(
    VulkanTensor& output,
    const VulkanTensor& input,
    const VImage& kernelImage,
    const c10::optional<const float*> bias,
    const Conv2DParams& params,
    c10::optional<float> output_min,
    c10::optional<float> output_max) {
  TORCH_INTERNAL_ASSERT(
      params.G == 1, "Prepacked kernel VImage for non-group conv2d only");
  conv2d(
      output,
      input,
      kernelImage,
      bufferFromOptionalHostData(
          bias,
          sizeof(float) * params.OC,
          sizeof(float) * ALIGN_UP4(params.OC)),
      params,
      output_min,
      output_max);
}

void conv2d(
    VulkanTensor& output,
    const VulkanTensor& input,
    const VulkanTensor& weight_prepacked,
    c10::optional<const float*> bias,
    const Conv2DParams params,
    c10::optional<float> output_min,
    c10::optional<float> output_max) {
  if (params.G > 1) {
    conv2d_depthwise(
        output,
        input,
        weight_prepacked,
        bufferFromOptionalHostData(
            bias,
            sizeof(float) * params.OC,
            sizeof(float) * ALIGN_UP4(params.OC)),
        params,
        output_min,
        output_max);
    return;
  }

  conv2d(
      output,
      input,
      *(weight_prepacked.image()),
      bias,
      params,
      output_min,
      output_max);
}

void conv2d(
    VulkanTensor& output,
    const VulkanTensor& input,
    const VulkanTensor& weight_prepacked,
    const VulkanTensor& bias,
    const Conv2DParams params,
    c10::optional<float> output_min,
    c10::optional<float> output_max) {
  if (params.G > 1) {
    conv2d_depthwise(
        output,
        input,
        weight_prepacked,
        *(bias.buffer()),
        params,
        output_min,
        output_max);
    return;
  }

  conv2d(
      output,
      input,
      *(weight_prepacked.image()),
      *(bias.buffer()),
      params,
      output_min,
      output_max);
}

void conv2d(
    VulkanTensor& output,
    const VulkanTensor& input,
    const float* weight,
    const c10::optional<const float*> bias,
    const Conv2DParams params,
    c10::optional<float> output_min,
    c10::optional<float> output_max) {
  if (params.G > 1) {
    TORCH_INTERNAL_ASSERT(
        params.G == params.C,
        "Vulkan conv2d supports only no-group and depthwise");
    conv2d_depthwise(
        output, input, weight, bias, params, output_min, output_max);
    return;
  }

  conv2d(
      output,
      input,
      conv2d_prepack_weights_image(
          weight, params.OC, params.C, params.KH, params.KW),
      bias,
      params,
      output_min,
      output_max);
}

void clamp(
    VulkanTensor& output,
    const VulkanTensor& input,
    float min,
    float max) {
  auto sizes = output.sizes();
  auto C = sizes[0] * sizes[1];
  auto H = sizes[2];
  auto W = sizes[3];

  auto device = context().device();
  struct ConstBlock {
    float min;
    float max;
  };
  ConstBlock cb{min, max};
  VBuffer constBuffer = makeUniformConstBuffer((void*)&cb, sizeof(cb));

  VkDescriptorSetLayout descriptorSetLayout{};
  VkDescriptorPool descriptorPool{};
  VkDescriptorSet descriptorSet{};
  std::vector<VkDescriptorType> descriptorTypes{
      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
  createDescriptorSetLayoutSinglePool(
      device,
      descriptorTypes,
      &descriptorSetLayout,
      &descriptorPool,
      &descriptorSet);

  output.image()->bindStorageImage(descriptorSet, 0);
  input.image()->bindShaderRead(descriptorSet, 1);
  constBuffer.bind(descriptorSet, 2);

  WorkGroupSize workGroupSize{8, 8, 1};
  auto& computeUnit = context().computeUnitFactory().get(
      GLSL_SPV(clamp), descriptorSetLayout, workGroupSize);
  computeUnit.createCommandBuffer(descriptorSet);
  auto commandBuffer = computeUnit.commandBuffer();
  output.image()->addImageMemoryBarrierToGeneral(commandBuffer);
  input.image()->addImageMemoryBarrierToShaderRead(commandBuffer);
  computeUnit.dispatchCommandBuffer(W, H, C, workGroupSize);
  computeUnit.endCommandBuffer();
  computeUnit.submitAndWaitCommandBuffer();
  vkDestroyDescriptorPool(device, descriptorPool, nullptr);
  vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
}

void addmm(
    VulkanTensor& output,
    c10::optional<const VulkanTensor> t,
    const VulkanTensor& m1,
    const VulkanTensor& m2,
    float beta,
    float alpha) {
  bool hasT = t.has_value();
  const auto m1Sizes = m1.sizes();
  const auto m2Sizes = m2.sizes();
  TORCH_INTERNAL_ASSERT(m1Sizes.size() == 2);
  TORCH_INTERNAL_ASSERT(m2Sizes.size() == 2);
  const auto m1W = m1Sizes[1];
  const auto m1C = 1;
  const auto m2H = m2Sizes[0];
  const auto m2C = 1;
  const auto OH = m1Sizes[0];
  const auto OW = m2Sizes[1];

  TORCH_INTERNAL_ASSERT(m1W == m2H);
  TORCH_INTERNAL_ASSERT(m1C == m2C);

  const auto C = m1C;
  const auto C_4 = UP_DIV(C, 4);

  auto device = context().device();

  struct ConstBlock {
    float alpha;
    float beta;
  };
  ConstBlock cb{alpha, beta};
  VBuffer constBuffer = makeUniformConstBuffer((void*)&cb, sizeof(cb));

  VkDescriptorSetLayout descriptorSetLayout{};
  VkDescriptorPool descriptorPool{};
  VkDescriptorSet descriptorSet{};
  std::vector<VkDescriptorType> descriptorTypes{};
  if (hasT) {
    descriptorTypes = {
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    };
  } else {
    descriptorTypes = {
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
    };
  }

  createDescriptorSetLayoutSinglePool(
      device,
      descriptorTypes,
      &descriptorSetLayout,
      &descriptorPool,
      &descriptorSet);

  output.image()->bindStorageImage(descriptorSet, 0);
  m1.image()->bindShaderRead(descriptorSet, 1);
  m2.image()->bindShaderRead(descriptorSet, 2);
  if (hasT) {
    (*t).image()->bindShaderRead(descriptorSet, 3);
    constBuffer.bind(descriptorSet, 4);
  }

  WorkGroupSize workGroupSize{8, 8, 1};
  if (hasT) {
    auto& computeUnit = context().computeUnitFactory().get(
        GLSL_SPV(addmm), descriptorSetLayout, workGroupSize);
    computeUnit.createCommandBuffer(descriptorSet);
    auto commandBuffer = computeUnit.commandBuffer();
    output.image()->addImageMemoryBarrierToGeneral(commandBuffer);
    m1.image()->addImageMemoryBarrierToShaderRead(commandBuffer);
    m2.image()->addImageMemoryBarrierToShaderRead(commandBuffer);
    (*t).image()->addImageMemoryBarrierToShaderRead(commandBuffer);
    computeUnit.dispatchCommandBuffer(OW, OH, C_4, workGroupSize);
    computeUnit.endCommandBuffer();
    computeUnit.submitAndWaitCommandBuffer();
  } else {
    auto& computeUnit = context().computeUnitFactory().get(
        GLSL_SPV(mm), descriptorSetLayout, workGroupSize);
    computeUnit.createCommandBuffer(descriptorSet);
    auto commandBuffer = computeUnit.commandBuffer();
    output.image()->addImageMemoryBarrierToGeneral(commandBuffer);
    m1.image()->addImageMemoryBarrierToShaderRead(commandBuffer);
    m2.image()->addImageMemoryBarrierToShaderRead(commandBuffer);
    computeUnit.dispatchCommandBuffer(OW, OH, C_4, workGroupSize);
    computeUnit.endCommandBuffer();
    computeUnit.submitAndWaitCommandBuffer();
  }
  vkDestroyDescriptorPool(device, descriptorPool, nullptr);
  vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
}

void mean(VulkanTensor& output, const VulkanTensor& input) {
  auto isizes = input.sizes();
  int32_t N = safe_downcast<int32_t>(isizes[0]);
  int32_t C = safe_downcast<int32_t>(isizes[1]);
  int32_t H = safe_downcast<int32_t>(isizes[2]);
  int32_t W = safe_downcast<int32_t>(isizes[3]);

  auto device = context().device();
  struct ConstBlock {
    int32_t W;
    int32_t H;
  };
  ConstBlock cb{W, H};
  VBuffer constBuffer = makeUniformConstBuffer((void*)&cb, sizeof(cb));

  VkDescriptorSetLayout descriptorSetLayout{};
  VkDescriptorPool descriptorPool{};
  VkDescriptorSet descriptorSet{};
  std::vector<VkDescriptorType> descriptorTypes{
      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
  createDescriptorSetLayoutSinglePool(
      device,
      descriptorTypes,
      &descriptorSetLayout,
      &descriptorPool,
      &descriptorSet);

  output.image()->bindStorageImage(descriptorSet, 0);
  input.image()->bindShaderRead(descriptorSet, 1);
  constBuffer.bind(descriptorSet, 2);

  WorkGroupSize workGroupSize{1, 1, 1};
  auto& computeUnit = context().computeUnitFactory().get(
      GLSL_SPV(mean2d), descriptorSetLayout, workGroupSize);
  computeUnit.createCommandBuffer(descriptorSet);
  auto commandBuffer = computeUnit.commandBuffer();
  output.image()->addImageMemoryBarrierToGeneral(commandBuffer);
  input.image()->addImageMemoryBarrierToShaderRead(commandBuffer);
  computeUnit.dispatchCommandBuffer(C, N, 1, workGroupSize);
  computeUnit.endCommandBuffer();
  computeUnit.submitAndWaitCommandBuffer();
  vkDestroyDescriptorPool(device, descriptorPool, nullptr);
  vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
}

} // namespace detail
} // namespace vulkan
} // namespace native
} // namespace at
