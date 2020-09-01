#pragma once

#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/VulkanOpaqueTensorImpl.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

//
// This class represents a Vulkan tensor and provides an abstraction layer
// that allows both the CPU, and the GPU, to view a Vulkan (buffer, image)
// combo as one coherent, synchronized unit of storage on both UMA and NUMA
// architectures.  Expanding on the previous sentence, this class tries to
// address 2 orthogonal complexities in implementation that arise as a result
// of the aforementioned goal:
//
// 1) First, synchronization across processors. CPUs and GPUs are separate
//    processors, and even though they share the same address space in a system
//    with a UMA architecture, their addresses spaces only partially overlap
//    on NUMA, with different access latencies between CPU and GPU.  We want to
//    keep the tensor in, or as close to, GPU memory as possible even if that
//    requires maintaining two copies on NUMA.  Maintaining these two copies
//    requires synchronization.
//
// 2) Second, synchronization across resources (i.e. buffers and images). GPU
//    drivers pack images in proprietory formats for better locality of
//    access.  Even on a UMA architecture conversion between buffers and
//    textures is expensive and manual in Vulkan.  This requires a second order
//    of synchronization.
//
// It is extremely important to keep in mind that the functions this class
// provides are generally expensive.  Far optimal performance, the user of
// this class should
//
// 1) Avoid frequent CPU <=> GPU transfers which will be triggered if data is
//    write accessed on one and read / write accessed on the other processor.
//
// 2) Avoid frequent buffer <=> image conversions which will be trigerred if
//    data is read from / written to both as a buffer and as an image.
//
// For optimal performance, access the data as images, and keep the data on GPU,
// and above all understand the expensive data flow that this class abstracts
// away.
//

class vTensor final {
 public:
  vTensor();
  vTensor(IntArrayRef sizes, const TensorOptions& options);

  /*
    Access
  */

  struct Access final {
    typedef uint8_t Flags;

    enum Type : Flags {
      Read = 1u << 0u,
      Write = 1u << 1u,
    };
  };

  /*
    Host
  */

  template<
      typename Type,
      typename Pointer = std::add_pointer_t<std::add_const_t<Type>>>
  api::Resource::Memory::Data<Pointer> host() const;

  template<
      typename Type,
      typename Pointer = std::add_pointer_t<Type>>
  api::Resource::Memory::Data<Pointer> host(Access::Flags access);

  /*
    Device
  */

  VkBuffer buffer() const;
  VkBuffer buffer(Access::Flags access);

  VkImage image() const;
  VkImage image(Access::Flags access);

 private:
  c10::SmallVector<int64_t, 4u> sizes_;
  TensorOptions options_;
  api::Resource::Buffer staging_;
  api::Resource::Buffer buffer_;
  api::Resource::Image image_;
};

using vTensorImpl = VulkanOpaqueTensorImpl<vTensor>;
void verify(const TensorOptions& options);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
