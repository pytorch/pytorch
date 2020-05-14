#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <iostream>
#include <limits>
#include <vector>

#include <ATen/native/vulkan/Vulkan.h>
#include <ATen/native/vulkan/VulkanCommon.h>
#include <ATen/native/vulkan/VulkanOps.h>

namespace at {
namespace native {
namespace vulkan {
namespace details {
namespace vulkan {

void upsample_nearest2d(
    VulkanTensor& output,
    const VulkanTensor& input,
    int64_t IH,
    int64_t IW,
    int64_t OH,
    int64_t OW,
    int64_t _N,
    int64_t _C,
    float scaleH,
    float scaleW) {
  auto device = context().device();
  auto physicalDevice = context().physicalDevice();
  int64_t C = _N * _C;
  struct ConstBlock {
    int32_t IW;
    int32_t IH;
    int32_t OW;
    int32_t OH;
    float scaleX;
    float scaleY;
  };
  ConstBlock cb{IW, IH, OW, OH, scaleW, scaleH};
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
  ComputeUnit computeUnit{
      at::native::vulkan::GLSL_SPV(vulkan_upsampleNearest2d),
      descriptorSetLayout,
      workGroupSize};
  computeUnit.createCommandBuffer(descriptorSet);
  input.image()->addImageMemoryBarrierGeneralToShaderRead(
      computeUnit.commandBuffer());
  computeUnit.dispatchCommandBuffer(OW, OH, C, workGroupSize);
  computeUnit.runCommandBuffer();
  vkDestroyDescriptorPool(device, descriptorPool, nullptr);
  vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
}

void add(
    VulkanTensor& output,
    const VulkanTensor& input0,
    const VulkanTensor& input1,
    float alpha) {
  TORCH_INTERNAL_ASSERT(
      output.dim() == 4,
      "Vulkan add is implemented for 4-dim tensors, output is not 4-dim");
  TORCH_INTERNAL_ASSERT(
      input0.dim() == 4,
      "Vulkan add is implemented for 4-dim tensors, input0 is not 4-dim");
  TORCH_INTERNAL_ASSERT(
      input1.dim() == 4,
      "Vulkan add is implemented for 4-dim tensors, input1 is not 4-dim");
  auto sizes = output.sizes();
  auto C = sizes[0] * sizes[1];
  auto H = sizes[2];
  auto W = sizes[3];

  auto device = context().device();
  auto physicalDevice = context().physicalDevice();
  struct ConstBlock {
    int32_t W;
    int32_t H;
    int32_t C;
    float alpha;
  };
  ConstBlock cb{W, H, C, alpha};
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
  ComputeUnit computeUnit{at::native::vulkan::GLSL_SPV(vulkan_add),
                          descriptorSetLayout,
                          workGroupSize};
  computeUnit.createCommandBuffer(descriptorSet);
  auto commandBuffer = computeUnit.commandBuffer();
  output.image()->addImageMemoryBarrierUndefinedToGeneral(commandBuffer);
  input0.image()->addImageMemoryBarrierGeneralToShaderRead(commandBuffer);
  input1.image()->addImageMemoryBarrierGeneralToShaderRead(commandBuffer);
  computeUnit.dispatchCommandBuffer(W, H, C, workGroupSize);
  computeUnit.runCommandBuffer();
  vkDestroyDescriptorPool(device, descriptorPool, nullptr);
  vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
}

VBuffer kernelNCHW_OCHW_repack_O4C4HWi4o4(
    const float* weights,
    const int OC,
    const int C,
    const int KH,
    const int KW) {
  const auto Cau4 = ALIGN_UP4(C);
  const auto C_4 = UP_DIV(C, 4);
  const auto kBufSizeNumel = ALIGN_UP4(OC) * Cau4 * KH * KW;
  auto size = sizeof(float) * kBufSizeNumel;
  auto sizeAligned =
      ROUND_UP(size, context().limits().minStorageBufferOffsetAlignment);
  VBuffer kernelBuffer{sizeAligned};
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
  return kernelBuffer;
}

VBuffer bufferFromOptionalHostData(
    c10::optional<float*> data,
    const uint32_t size) {
  const auto sizeAligned =
      ROUND_UP(size, context().limits().minStorageBufferOffsetAlignment);
  VBuffer buffer{sizeAligned};
  if (data.has_value()) {
    buffer.copy_from_host_to_device((void*)*data, size);
  } else {
    buffer.set_zeros();
  }
  return buffer;
}

VBuffer bufferZeros(const uint32_t size) {
  const auto sizeAligned =
      ROUND_UP(size, context().limits().minStorageBufferOffsetAlignment);
  VBuffer buffer{sizeAligned};
  buffer.set_zeros();
  return buffer;
}

uint32_t conv2d_biasBufferSize(uint32_t oc) {
  return sizeof(float) * ALIGN_UP4(oc);
}

void conv2d_depthwise(
    VulkanTensor& output,
    const VulkanTensor& input,
    const float* weight,
    int64_t KH,
    int64_t KW,
    const c10::optional<float*> bias,
    int64_t SY,
    int64_t SX,
    int64_t PY,
    int64_t PX,
    int64_t DY,
    int64_t DX,
    int64_t G) {
  auto osizes = output.sizes();
  Conv2DParams c2ds{
      input.sizes(), osizes[1], KH, KW, SY, SX, PY, PX, DY, DX, G};
  TORCH_INTERNAL_ASSERT(osizes[2] == c2ds.OH);
  TORCH_INTERNAL_ASSERT(osizes[3] == c2ds.OW);
  auto biasBuffer =
      bufferFromOptionalHostData(bias, conv2d_biasBufferSize(c2ds.OC));
  struct ConstBlock {
    int32_t padding[2];
    int32_t kernelSize[2];
    int32_t stride[2];
    int32_t dilate[2];
    int32_t inputSize[4];
    int32_t outputSize[4];
  };
  ConstBlock cb{{c2ds.PX, c2ds.PY},
                {c2ds.KW, c2ds.KH},
                {c2ds.SX, c2ds.SY},
                {c2ds.DX, c2ds.DY},
                {c2ds.OW, c2ds.OH, c2ds.OC_4, 0},
                {c2ds.W, c2ds.H, c2ds.C_4, 0}};
  VBuffer constBuffer = makeUniformConstBuffer((void*)&cb, sizeof(cb));

  VulkanTensor kernel{std::vector<int64_t>{c2ds.OC, c2ds.KH, c2ds.KW}};
  kernel.set_data_from_host(weight);

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
  kernel.image()->bindShaderRead(descriptorSet, 2);
  biasBuffer.bind(descriptorSet, 3);
  constBuffer.bind(descriptorSet, 4);

  WorkGroupSize workGroupSize{8, 8, 1};
  ComputeUnit computeUnit =
      ComputeUnit{at::native::vulkan::GLSL_SPV(vulkan_convDW_tex),
                  descriptorSetLayout,
                  workGroupSize};
  computeUnit.createCommandBuffer(descriptorSet);
  auto commandBuffer = computeUnit.commandBuffer();
  output.image()->addImageMemoryBarrierUndefinedToGeneral(commandBuffer);
  input.image()->addImageMemoryBarrierGeneralToShaderRead(commandBuffer);
  kernel.image()->addImageMemoryBarrierGeneralToShaderRead(commandBuffer);
  computeUnit.dispatchCommandBuffer(c2ds.OW, c2ds.OH, c2ds.OC_4, workGroupSize);
  computeUnit.runCommandBuffer();

  vkDestroyDescriptorPool(device, descriptorPool, nullptr);
  vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
}

ImageSizes conv2d_prepack_weights_image_sizes(
    int64_t OC,
    int64_t C,
    int64_t KH,
    int64_t KW) {
  return {{ALIGN_UP4(C), UP_DIV(OC, 4), KH * KW}, {C, OC, KH * KW}};
}

void conv2d_prepack_weights_to_image(
    VImage& image,
    const float* weight,
    int64_t OC,
    int64_t C,
    int64_t KH,
    int64_t KW) {
  auto device = context().device();
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
  ConstBlock cb{KW * KH, C_4};
  VBuffer constBuffer = makeUniformConstBuffer((void*)&cb, sizeof(cb));

  VkDescriptorSetLayout descriptorSetLayout{};
  VkDescriptorPool descriptorPool{};
  VkDescriptorSet descriptorSet{};
  std::vector<VkDescriptorType> descriptorTypes{
      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
  createDescriptorSetLayoutSinglePool(
      device,
      descriptorTypes,
      &descriptorSetLayout,
      &descriptorPool,
      &descriptorSet);

  image.bindStorageImage(descriptorSet, 0);
  kernelBuffer.bind(descriptorSet, 1);
  constBuffer.bind(descriptorSet, 2);

  WorkGroupSize workGroupSize{1, 1, 1};
  ComputeUnit computeUnit{at::native::vulkan::GLSL_SPV(vulkan_KO4C4HW_to_image),
                          descriptorSetLayout,
                          workGroupSize};
  computeUnit.createCommandBuffer(descriptorSet);
  auto commandBuffer = computeUnit.commandBuffer();
  image.addImageMemoryBarrierUndefinedToGeneral(commandBuffer);
  kernelBuffer.addBufferMemoryBarrier(
      commandBuffer, 0, kernelBuffer.sizeBytes());
  computeUnit.dispatchCommandBuffer(C_4, OC_4, KH * KW, workGroupSize);
  computeUnit.runCommandBuffer();
  vkDestroyDescriptorPool(device, descriptorPool, nullptr);
  vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
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
    const Conv2DParams& c2ds) {
  TORCH_INTERNAL_ASSERT(
      c2ds.G == 1, "Prepacked kernel VImage for non-group conv2d only");
  auto osizes = output.sizes();
  TORCH_INTERNAL_ASSERT(
      osizes[2] == c2ds.OH,
      "Output tensor dims do not match specified conv2d params");
  TORCH_INTERNAL_ASSERT(
      osizes[3] == c2ds.OW,
      "Output tensor dims do not match specified conv2d params");

  struct ConstBlock {
    int32_t padding[2];
    int32_t kernelSize[2];
    int32_t stride[2];
    int32_t dilate[2];
    int32_t inputSize[4];
    int32_t outputSize[4];
  };
  ConstBlock cb{{c2ds.PX, c2ds.PY},
                {c2ds.KW, c2ds.KH},
                {c2ds.SX, c2ds.SY},
                {c2ds.DX, c2ds.DY},
                {c2ds.OW, c2ds.OH, c2ds.OC_4, c2ds.OC},
                {c2ds.W, c2ds.H, c2ds.C_4, c2ds.C}};
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

  WorkGroupSize workGroupSize{1, 1, c2ds.OC_4};
  ComputeUnit computeUnit{at::native::vulkan::GLSL_SPV(vulkan_conv_tex_IKnc4hw),
                          descriptorSetLayout,
                          workGroupSize};
  computeUnit.createCommandBuffer(descriptorSet);
  auto commandBuffer = computeUnit.commandBuffer();
  output.image()->addImageMemoryBarrierUndefinedToGeneral(commandBuffer);
  input.image()->addImageMemoryBarrierGeneralToShaderRead(commandBuffer);
  kernelImage.addImageMemoryBarrierGeneralToShaderRead(commandBuffer);
  computeUnit.dispatchCommandBuffer(
      UP_DIV(c2ds.OW, 4 * workGroupSize.x),
      UP_DIV(c2ds.OH, workGroupSize.y),
      UP_DIV(c2ds.OC_4, workGroupSize.z));
  computeUnit.runCommandBuffer();

  vkDestroyDescriptorPool(device, descriptorPool, nullptr);
  vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
}

void conv2d(
    VulkanTensor& output,
    const VulkanTensor& input,
    const VImage& kernelImage,
    const c10::optional<float*> bias,
    const Conv2DParams& c2ds) {
  TORCH_INTERNAL_ASSERT(
      c2ds.G == 1, "Prepacked kernel VImage for non-group conv2d only");
  conv2d(
      output,
      input,
      kernelImage,
      bufferFromOptionalHostData(
          bias, conv2d_biasBufferSize(ALIGN_UP4(c2ds.OC))),
      c2ds);
}

void conv2d(
    VulkanTensor& output,
    const VulkanTensor& input,
    const VulkanTensor& weight_prepacked,
    int64_t KH,
    int64_t KW,
    c10::optional<float*> bias,
    int64_t SY,
    int64_t SX,
    int64_t PY,
    int64_t PX,
    int64_t DY,
    int64_t DX,
    int64_t G) {
  conv2d(
      output,
      input,
      *(weight_prepacked.image()),
      bias,
      Conv2DParams{
          input.sizes(), output.sizes()[1], KH, KW, SY, SX, PY, PX, DY, DX, G});
}

void conv2d(
    VulkanTensor& output,
    const VulkanTensor& input,
    const VulkanTensor& weight_prepacked,
    int64_t KH,
    int64_t KW,
    const VulkanTensor& bias,
    int64_t SY,
    int64_t SX,
    int64_t PY,
    int64_t PX,
    int64_t DY,
    int64_t DX,
    int64_t G) {
  conv2d(
      output,
      input,
      *(weight_prepacked.image()),
      *(bias.buffer()),
      Conv2DParams{
          input.sizes(), output.sizes()[1], KH, KW, SY, SX, PY, PX, DY, DX, G});
}

void conv2d(
    VulkanTensor& output,
    const VulkanTensor& input,
    const float* weight,
    int64_t KH,
    int64_t KW,
    const c10::optional<float*> bias,
    int64_t SY,
    int64_t SX,
    int64_t PY,
    int64_t PX,
    int64_t DY,
    int64_t DX,
    int64_t G) {
  Conv2DParams c2ds{
      input.sizes(), output.sizes()[1], KH, KW, SY, SX, PY, PX, DY, DX, G};
  if (G > 1) {
    TORCH_INTERNAL_ASSERT(
        G == c2ds.C, "Vulkan conv2d supports only no-group and depthwise");
    conv2d_depthwise(
        output, input, weight, KH, KW, bias, SY, SX, PY, PX, DY, DX, G);
    return;
  }
  conv2d(
      output,
      input,
      conv2d_prepack_weights_image(weight, c2ds.OC, c2ds.C, KH, KW),
      bias,
      c2ds);
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
  auto C_4 = UP_DIV(C, 4);

  auto device = context().device();
  auto physicalDevice = context().physicalDevice();
  struct ConstBlock {
    int32_t W;
    int32_t H;
    int32_t C_4;
    int32_t C;
    float min;
    float max;
  };
  ConstBlock cb{W, H, C_4, C, min, max};
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
  ComputeUnit computeUnit{at::native::vulkan::GLSL_SPV(vulkan_clamp),
                          descriptorSetLayout,
                          workGroupSize};
  computeUnit.createCommandBuffer(descriptorSet);
  auto commandBuffer = computeUnit.commandBuffer();
  output.image()->addImageMemoryBarrierUndefinedToGeneral(commandBuffer);
  input.image()->addImageMemoryBarrierGeneralToShaderRead(commandBuffer);
  computeUnit.dispatchCommandBuffer(W, H, C, workGroupSize);
  computeUnit.runCommandBuffer();
  vkDestroyDescriptorPool(device, descriptorPool, nullptr);
  vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
}

void addmm(
    VulkanTensor& output,
    const VulkanTensor& t,
    const VulkanTensor& m1,
    const VulkanTensor& m2,
    float beta,
    float alpha) {
  auto m1Sizes = m1.sizes();
  auto m2Sizes = m2.sizes();
  TORCH_INTERNAL_ASSERT(m1Sizes.size() == 2);
  TORCH_INTERNAL_ASSERT(m2Sizes.size() == 2);
  uint32_t m1H = m1Sizes[0];
  uint32_t m1W = m1Sizes[1];
  uint32_t m1C = 1;
  uint32_t m1C_4 = UP_DIV(m1C, 4);

  uint32_t m2H = m2Sizes[0];
  uint32_t m2W = m2Sizes[1];
  uint32_t m2C = 1;
  uint32_t m2C_4 = UP_DIV(m2C, 4);

  uint32_t OH = m1Sizes[0];
  uint32_t OW = m2Sizes[1];

  TORCH_INTERNAL_ASSERT(m1W == m2H);
  TORCH_INTERNAL_ASSERT(m1C == m2C);

  uint32_t C = m1C;
  uint32_t C_4 = UP_DIV(C, 4);
  uint32_t K = m1W;

  auto tSizes = t.sizes();
  uint32_t TH = tSizes[0];
  uint32_t TW = tSizes[1];
  uint32_t TC = 1;

  auto device = context().device();
  auto physicalDevice = context().physicalDevice();

  struct ConstBlock {
    int32_t OW;
    int32_t OH;
    int32_t C_4;
    int32_t C;
    float beta;
    float alpha;
    int32_t K;
  };
  ConstBlock cb{OW, OH, C_4, C, beta, alpha, K};
  VBuffer constBuffer = makeUniformConstBuffer((void*)&cb, sizeof(cb));

  VkDescriptorSetLayout descriptorSetLayout{};
  VkDescriptorPool descriptorPool{};
  VkDescriptorSet descriptorSet{};
  std::vector<VkDescriptorType> descriptorTypes{
      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
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
  m1.image()->bindShaderRead(descriptorSet, 1);
  m2.image()->bindShaderRead(descriptorSet, 2);
  t.image()->bindShaderRead(descriptorSet, 3);
  constBuffer.bind(descriptorSet, 4);

  WorkGroupSize workGroupSize{8, 8, 1};
  ComputeUnit computeUnit{at::native::vulkan::GLSL_SPV(vulkan_addmm),
                          descriptorSetLayout,
                          workGroupSize};
  computeUnit.createCommandBuffer(descriptorSet);
  auto commandBuffer = computeUnit.commandBuffer();
  output.image()->addImageMemoryBarrierUndefinedToGeneral(commandBuffer);
  m1.image()->addImageMemoryBarrierGeneralToShaderRead(commandBuffer);
  m2.image()->addImageMemoryBarrierGeneralToShaderRead(commandBuffer);
  t.image()->addImageMemoryBarrierGeneralToShaderRead(commandBuffer);
  computeUnit.dispatchCommandBuffer(OW, OH, C_4, workGroupSize);
  computeUnit.runCommandBuffer();
  vkDestroyDescriptorPool(device, descriptorPool, nullptr);
  vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
}

void mean(VulkanTensor& output, const VulkanTensor& input) {
  auto isizes = input.sizes();
  auto N = isizes[0];
  auto C = isizes[1];
  auto H = isizes[2];
  auto W = isizes[3];
  auto C_4 = UP_DIV(N * C, 4);

  auto device = context().device();
  auto physicalDevice = context().physicalDevice();
  struct ConstBlock {
    int32_t W;
    int32_t H;
    int32_t OW;
    int32_t OH;
  };
  ConstBlock cb{W, H, C, N};
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
  ComputeUnit computeUnit{at::native::vulkan::GLSL_SPV(vulkan_mean),
                          descriptorSetLayout,
                          workGroupSize};
  computeUnit.createCommandBuffer(descriptorSet);
  auto commandBuffer = computeUnit.commandBuffer();
  output.image()->addImageMemoryBarrierUndefinedToGeneral(commandBuffer);
  input.image()->addImageMemoryBarrierGeneralToShaderRead(commandBuffer);
  computeUnit.dispatchCommandBuffer(1, 1, C_4, workGroupSize);
  computeUnit.runCommandBuffer();
  vkDestroyDescriptorPool(device, descriptorPool, nullptr);
  vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
}
} // namespace vulkan
} // namespace details
} // namespace vulkan
} // namespace native
} // namespace at
