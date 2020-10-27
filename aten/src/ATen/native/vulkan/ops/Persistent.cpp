#include <ATen/native/vulkan/ops/Persistent.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

void release_buffer(
    std::reference_wrapper<api::Resource::Pool> pool,
    const api::Resource::Buffer& buffer) {
  pool.get().release(buffer);
}

void release_image(
    std::reference_wrapper<api::Resource::Pool> pool,
    const api::Resource::Image& image) {
  pool.get().release(image);
}

} // namespace

Persistent::Pool::Pool(const api::GPU& gpu)
  : pool_(gpu) {
}

Persistent::Buffer Persistent::Pool::buffer(
    const api::Resource::Buffer::Descriptor& descriptor,
    const c10::ArrayRef<const uint8_t> data) {
  api::Resource::Buffer buffer = pool_.buffer(descriptor);

  {
    api::Resource::Memory::Handle<uint8_t*> memory = buffer.memory.template map<
        uint8_t,
        api::Resource::Memory::Access::Write>();

    memcpy();
  }

  return Persistent::Buffer{
    buffer,
    std::bind(release_buffer, std::ref(pool_), std::placeholders::_1),
  };
}

Persistent::Image Persistent::Pool::image(
    const api::Resource::Image::Descriptor& descriptor,
    const c10::ArrayRef<const uint8_t> data) {
  return Persistent::Image{
    api::Resource::Image{},
    std::bind(release_image, std::ref(pool_), std::placeholders::_1),
  };
}

Persistent::Pool* persistent() {
  typedef Persistent::Pool Pool;

  static const std::unique_ptr<Pool> pool([]() -> Pool* {
    try {
      return new Pool(api::context()->gpu());
    }
    catch (...) {
      return nullptr;
    }
  }());

  TORCH_CHECK(
      pool,
      "Vulkan: Failed to initialize the persistent resource pool!");

  return pool.get();
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
