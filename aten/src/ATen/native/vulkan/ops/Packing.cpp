#include <ATen/native/vulkan/ops/Packing.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

using namespace api::utils;

vTensor pack_image2d_h2w2(vTensor v_src, api::Context* context, api::Command::Buffer& command_buffer) {
  TORCH_CHECK(v_src.has_image() , "Not implemented!");
  TORCH_CHECK(v_src.sizes().size() <= 2 , "pack_image2d_h2w2 not supported for tensors with dims > 2");

  uint32_t orig_w, orig_h;
  uint32_t packed_h, packed_w;
  if (v_src.sizes().size() == 2) {
    packed_h = div_up(v_src.sizes()[Layout::Parameter::height], INT64_C(2));
    packed_w = div_up(v_src.sizes()[Layout::Parameter::width], INT64_C(2));
    orig_w = v_src.sizes()[Layout::Parameter::width];
    orig_h = v_src.sizes()[Layout::Parameter::height];
  }
  else {
    packed_h = 1;
    packed_w = div_up(v_src.sizes()[Layout::Parameter::height], INT64_C(2));
    orig_w = v_src.sizes()[Layout::Parameter::height];
    orig_h = 1;
  }
  vTensor v_src_packed {
    context,
    {
      4,
      packed_h,
      packed_w,
    },
    v_src.options()
  };
  const struct {
    uvec3 extents;
    uint32_t fill;
    uvec4 orig_size;
  } block {
    v_src_packed.extents(),
    0u,
    {
      orig_w,
      orig_h,
    }
  };

  context->dispatch(
    command_buffer,
    {
      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    },
    VK_KERNEL(hw_to_image2d),
    v_src_packed.extents(),
    {8,8,1},
    v_src_packed.image(
      command_buffer,
      vTensor::Stage::Compute,
      vTensor::Access::Write),
    v_src.buffer(
      command_buffer,
      vTensor::Stage::Compute,
      vTensor::Access::Read),
    context->resource().pool.uniform(block).object);

  return v_src_packed;
}

vTensor unpack_image2d_h2w2(vTensor v_src, c10::SmallVector<int64_t, 4u> output_sizes, api::Context* context, api::Command::Buffer& command_buffer) {
  TORCH_CHECK(v_src.has_image() , "Not implemented!");
  TORCH_CHECK(v_src.sizes().size() > 2 , "pack_image2d_h2w2 not supported for tensors with dims > 2");

  vTensor v_src_unpacked {
    context,
    output_sizes,
    v_src.options()
  };

  uint32_t orig_w = output_sizes[output_sizes.size() - 1];
  uint32_t orig_h = output_sizes[output_sizes.size() - 2];
  const struct {
    uvec3 extents;
    uint32_t fill;
    uvec2 orig_sizes;
  } block {
    v_src.extents(),
    0u,
    {
      orig_w,
      orig_h,
    }
  };

  context->dispatch(
    command_buffer,
    {
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    },
    VK_KERNEL(image2d_to_hw),
    v_src.extents(),
    {8,8,1},
    v_src.image(
      command_buffer,
      vTensor::Stage::Compute,
      vTensor::Access::Read),
    v_src_unpacked.buffer(
      command_buffer,
      vTensor::Stage::Compute,
      vTensor::Access::Write),
    context->resource().pool.uniform(block).object);

  return v_src_unpacked;
}

vTensor unpack_image1x1(vTensor v_src, c10::SmallVector<int64_t, 4u> output_sizes, api::Context* context, api::Command::Buffer& command_buffer) {
  vTensor v_src_unpacked {
    context,
    output_sizes,
    v_src.options()
  };

  const struct {
    uvec3 extents;
  } block {
    v_src.extents(),
  };

  context->dispatch(
    command_buffer,
    {
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    },
    VK_KERNEL(image1x1_to_buffer),
    v_src.extents(),
    {1,1,64},
    v_src.image(
      command_buffer,
      vTensor::Stage::Compute,
      vTensor::Access::Read),
    v_src_unpacked.buffer(
      command_buffer,
      vTensor::Stage::Compute,
      vTensor::Access::Write),
    context->resource().pool.uniform(block).object);

  return v_src_unpacked;
}

}
}
}
}
