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
// pair as one coherent, synchronized unit of storage on both UMA and NUMA
// systems.  Expanding on the previous sentence, this class tries to address
// two orthogonal implementation complexities that arise as a result of the
// aforementioned goal of memory coherence:
//
// 1) First, synchronization across processors; CPUs and GPUs are separate
//    processors, and even though they share the same address space in a system
//    with a unified memory architecture, their address spaces only partially
//    overlap on NUMA.  Consequently on NUMA, while it is still technically
//    possible to take advantage of this shared address space to maintain one
//    single copy of the data, different access latencies from CPU and GPU to
//    this shared location usually necessitates maintaining two copies each in
//    processor-local memory, otherwise memory access latency will hurt from
//    the processor to which this data is not close.  This shared memory is more
//    often than not located in system memory, making for slow GPU read and
//    write access over the PCI-e bus.  Maintaining two separate copies on the
//    other hand, requires synchronization to guarantee coherence.  This is
//    not an issue on UMA and this implementation accounts for that optimization.
//
// 2) Second, synchronization across resources (i.e. buffers and images); GPU
//    drivers pack images in proprietory formats for better locality of access
//    and to enable lossless compression.  These conversions are both expensive
//    (in general) and manual (in Vulkan.)  This requires a second order of
//    synchronization to guarantee coherence between the contents of the buffer
//    and image otherwise they will go out of sync.
//
// It is extremely important to keep in mind that the functionality this class
// provides is generally expensive.  For optimal performance, the user of this
// class should:
//
// 1) Avoid frequent CPU <=> GPU transfers which will be triggered if data is
//    write accessed on one processor and read / write accessed on the other.
//
// 2) Avoid frequent buffer <=> image conversions which will be trigerred if
//    data is write accessed as a buffer (image) and read accessed as an
//    image (buffer).
//
// For optimal performance, access the data as images, and keep the data on GPU,
// and above all understand the expensive data flow that this class abstracts
// away.
//
// vTensor tries to address a specific concern and intentionally does not expose
// GPU tensor memory directly.  Please keep that behavior intact as the whole
// data model fundamentally depends on limiting what the user can achieve through
// the interface to guarantee performance and coherence.
//
// A vTensor is associated with an api::Context as preparation for multi-GPU
// support.
//

class vTensor final {
 public:
  vTensor();
  vTensor(
      api::Context* context,
      IntArrayRef sizes,
      const TensorOptions& options);

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
    Future
  */

  class Future final {
   public:

   private:
    VkFence fence_;
  };

  /*
    Host access - these functions can be expensive.
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
    Device access - these functions can be expensive.
  */

  VkBuffer buffer() const;
  VkBuffer buffer(Access::Flags access);

  VkImage image() const;
  VkImage image(Access::Flags access);

 private:
  api::Resource::Image image_;
  api::Resource::Buffer buffer_;
  api::Resource::Buffer staging_;
  api::Context* context_;
  c10::SmallVector<int64_t, 4u> sizes_;
  TensorOptions options_;

  mutable struct {
    uint32_t image : 1u;
    uint32_t buffer : 1u;
    uint32_t staging : 1u;
  } dirty_;
};

using vTensorImpl = VulkanOpaqueTensorImpl<vTensor>;
void verify(const TensorOptions& options);

//
// Impl
//

template<typename Type, typename Pointer>
api::Resource::Memory::Data<Pointer> vTensor::host() const {
}

template<typename Type, typename Pointer>
api::Resource::Memory::Data<Pointer> vTensor::host(
    const Access::Flags access) {
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
