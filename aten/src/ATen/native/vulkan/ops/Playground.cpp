#include <ATen/native/vulkan/ops/Playground.h>
#include <ATen/native/vulkan/ops/Persistent.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

api::Resource::Image get_image(api::Resource::Pool* const pool, long x, long y, long z) {
  VkFormat image_format = VK_FORMAT_R32G32B32A32_SFLOAT;
  return pool->image({
      VK_IMAGE_TYPE_3D,
      image_format,
      {x,y,div_up(z, INT64_C(4))},
      // Usage
      {
        VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_STORAGE_BIT,
        {
          VMA_MEMORY_USAGE_GPU_ONLY,
          0u,
          0u,
        },
      },
      // View
      {
        VK_IMAGE_VIEW_TYPE_3D,
        image_format
      },
      // Sampler
      {
        VK_FILTER_NEAREST,
        VK_SAMPLER_MIPMAP_MODE_NEAREST,
        VK_SAMPLER_ADDRESS_MODE_REPEAT,
        VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
      },
    });
}

api::Resource::Buffer get_buffer(api::Resource::Pool* const pool, long x, long y, long z) {
  VkDeviceSize buffer_size = x * y * align_up(z, INT64_C(4)) * 4;
  const VkFlags usage =
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
      VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  return pool->buffer({
      buffer_size,
      {
        usage,
        {
          VMA_MEMORY_USAGE_GPU_TO_CPU,
          0u,
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        }
      }
    });
}

} // namespace

void PlaygroundOpContext::fill_image(const api::Resource::Buffer& buffer, api::Resource::Fence& fence) {
  api::Context* const context = api::context();
  api::Command::Pool& command_pool = context->command().pool;
  if (!initted) {
    std::cout << "Creating command buffer" << std::endl;

    cmd_buffer.begin();

    context->dispatch(
        cmd_buffer,
        context->get_playground_cache(),
        {3*3*4, 1, 1},
        descriptor_set,
        buffer.object);

    cmd_buffer.end();

    initted = true;
  }
  else {
    std::cout << "Using cached command buffer" << std::endl;
  }

  command_pool.submit(context->gpu().queue, cmd_buffer, fence);
}

PlaygroundOpContext::PlaygroundOpContext(const Tensor& test)
  : packed_{
      convert(test.is_vulkan() ? test : test.vulkan())
    },
    unpacked_{
      test
    },
    cmd_buffer(api::context()->command().pool.allocate_persistent()),
    in_buffer{},
    initted(false) {
  api::Resource::Pool& resource_pool = persistent()->pool;
  in_buffer = get_buffer(&resource_pool, 3, 3, 4);

  api::Descriptor::Pool& descriptor_pool = persistent()->descriptor_pool;
  const api::Shader::Layout::Object shader_layout =
  {
    api::context()->get_playground_cache().set_layout.get(),
    api::context()->get_playground_cache().layout_descriptor.signature,
  };
  descriptor_set = descriptor_pool.allocate_single(shader_layout);
}

PlaygroundOpContext PlaygroundOpContext::create(const Tensor& test) {
  // Pass in the originals
  return PlaygroundOpContext{test};
}

Tensor PlaygroundOpContext::run(const Tensor& input) {
  api::Context* const context = api::context();

  api::Resource::Pool& resource_pool = persistent()->pool;

  /*
  const vTensor& v_input = convert(input);
  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();
  api::Resource::Buffer buffer = v_input.buffer_straight(command_buffer, vTensor::Stage::Compute);
  */

  void* data = nullptr;
  VK_CHECK(vmaMapMemory(in_buffer.memory.allocator, in_buffer.memory.allocation, &data));
  float* float_data = reinterpret_cast<float*>(data);

  memcpy(
      float_data,
      input.cpu().data_ptr<float>(),
      in_buffer.object.range);

  Tensor test_cpu = at::rand({1, 4, 3, 3}, at::device(at::kCPU).dtype(at::kFloat));

  api::Resource::Fence fence = resource_pool.fence();
  fill_image(in_buffer, fence);
  fence.wait();

  memcpy(
      test_cpu.data_ptr<float>(),
      float_data,
      in_buffer.object.range);

  context->flush();

  return test_cpu;
}

PlaygroundOpContext::State PlaygroundOpContext::unpack() const {
  return PlaygroundOpContext::State{
      unpacked_.test,
  };
}

c10::intrusive_ptr<PlaygroundOpContext> playground_prepack(Tensor&& test) {
  return c10::make_intrusive<PlaygroundOpContext>(
      PlaygroundOpContext::create(std::move(test)));
}

Tensor playground_run(
    const Tensor& input,
    const c10::intrusive_ptr<PlaygroundOpContext>& context) {
  return context->run(input);
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
