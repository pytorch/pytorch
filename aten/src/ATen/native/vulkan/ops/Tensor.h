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

class C10_EXPORT vTensor final {
 public:
  vTensor() = default;
  vTensor(
      api::Context* context,
      IntArrayRef sizes,
      const TensorOptions& options);

  /*
    Types
  */

  typedef api::Resource::Memory::Access Access;
  typedef api::Resource::Buffer Buffer;
  typedef api::Resource::Fence Fence;
  typedef api::Resource::Image Image;
  typedef api::Resource::Memory Memory;

  /*
    Future
  */

  template<typename Type, Access::Flags kAccess>
  class Future final {
    template<typename T, Access::Flags A>
    using is_convertible = std::enable_if_t<
        std::is_convertible<
            Access::Pointer<T, A>,
            Access::Pointer<Type, kAccess>>::value>;

   public:
    explicit Future(vTensor* tensor);
    Future(const Future&) = delete;
    Future& operator=(const Future&) = delete;
    Future(Future&&);
    Future& operator=(Future&&) &;
    Future& operator=(Future&&) && = delete;
    template<typename T, Access::Flags A, typename = is_convertible<T, A>>
    Future(Future<T, A>&&);
    template<typename T, Access::Flags A, typename = is_convertible<T, A>>
    Future& operator=(Future<T, A>&&) &;
    template<typename T, Access::Flags A>
    Future& operator=(Future<T, A>&&) && = delete;
    ~Future();

    typedef Memory::Data<
        Access::Pointer<
            Type,
            kAccess>> Payload;

    // This is a blocking operation as the name suggests.  A call to host() will
    // trigger an async copy if pending writes are detected.  Consequently, for
    // optimal performance, put as much time and distance between the place
    // where a vTensor::host() call occurs and the location where the returned
    // future is explicitly waited on as a result of a call to this function.

    Payload wait() const &;

   private:
    // Intentionally disabed to enforce a usage pattern wherein the Future's
    // lifetime exceeds that of the Payload as we use the Future's destructor
    // to eagerly (as opposed to lazily and upon first use) upload the
    // modifications back onto the GPU in an effort to hide the upload latency.

    Payload wait() const && = delete;

   private:
    template<typename, Access::Flags>
    friend class Future;

   private:
    vTensor* tensor_;
  };

  /*
    Host access - these functions will be expensive if they trigger a GPU -> CPU
    sync due to pending writes.  A call to host() will trigger an async copy in
    such scenarios, which is then explictly waited on as part of Future::wait().
    Consequently, for optimal performance, put as much time and distance between
    the place where this function is called, and the location where the future is
    waited on.
  */

  template<typename Type>
  Future<Type, Access::Read> host() const &;

  template<typename Type, Access::Flags kAccess>
  Future<Type, kAccess> host() &;

  /*
    Device access - these functions will be expensive if they trigger a buffer
    <-> image or CPU -> GPU sync due to pending writes.  These functions are
    non-blocking on the host as the copy operation is carried out by the GPU
    asynchronously.  Regardless, they result in extra work that could have been
    avoided or at least minimized if all data access had occured through one
    single processor (GPU in this case) and on one type of resource (image for
    best performance.)  Consequently, for optimal performance, avoid mixed reads
    and writes across processor boundaries, and do your best to minimize layout
    transitions as a result of working with images only (as opposed to mixed
    buffer - image usage.)
    This implementation intentionally restricts user access to the buffer and
    image objects only, as opposed to their underlying memory, for the sake of
    predictability of usage and efficiency.
  */

  Buffer::Object buffer() const &;
  Buffer::Object buffer(Access::Flags access) &;

  Image::Object image() const &;
  Image::Object image(Access::Flags access) &;

 private:
  // Some overloads below are intentionally disabled to enforce a usage pattern
  // that ensures the Tensor's lifetime exceeds that of the scope in which the
  // underlying data is accessed.  Allowing deleted overloads below to be
  // invoked on a temporary would open the door to the possibility of accessing
  // the underlying memory out of the expected scope.

  /*
    Future
  */

  Memory& wait();

  /*
    Host
  */

  const vTensor* host() const;
  vTensor* host(Access::Flags access);

  template<typename Type>
  Future<Type, Access::Read> host() const && = delete;

  template<typename Type, Access::Flags kAccess>
  Future<Type, kAccess> host() && = delete;

  /*
    Device
  */

  Buffer::Object buffer() const && = delete;
  Buffer::Object buffer(Access::Flags access) && = delete;

  Image::Object image() const && = delete;
  Image::Object image(Access::Flags access) && = delete;

 private:
  class View final {
   public:
    View();
    View(
        api::Context* context,
        IntArrayRef sizes,
        const TensorOptions& options);

    struct Component final {
      typedef uint8_t Flags;

      enum Type : Flags {
        Unknown = 0u << 0u,
        Buffer = 1u << 0u,
        Image = 1u << 1u,
        Staging = 1u << 2u,
      };
    };

    struct Active final {
      Component::Type component;
      Access::Flags access;

      friend std::ostream& operator<<(
          std::ostream& stream,
          const Active view) {
        stream << "Component: [";
        switch (view.component) {
          case Component::Unknown:
            stream << "Unknown";
            break;

          case Component::Buffer:
            stream << "Buffer";
            break;

          case Component::Image:
            stream << "Image";
            break;

          case Component::Staging:
            stream << "Staging";
            break;

          default:
            stream << "Unknown";
        }

        stream << "], Access: [";

        if (Access::Read == (view.access & Access::Read)) {
          stream << "Read";
        }
        else if (Access::Write == (view.access & Access::Write)) {
          stream << "Write";
        }
        else if (view.access) {
          stream << "Read | Write";
        }
        else {
          stream << "Unknown";
        }

        return stream << "]";
      }
    };

    Buffer& buffer() const;
    Image& image() const;
    Buffer& staging() const;
    Fence& fence() const;

    void transition(Active view) const;

   private:
    void verify() const;

   private:
    api::Context* context_;
    mutable Image image_;
    mutable Buffer buffer_;
    mutable Buffer staging_;
    mutable Fence fence_;
    Component::Flags required_;
    mutable Active active_;

    c10::SmallVector<int64_t, 6u> sizes_;
    TensorOptions options_;
  } view_;
};

using vTensorImpl = VulkanOpaqueTensorImpl<vTensor>;
void verify(const TensorOptions& options);

//
// Impl
//

template<typename Type, vTensor::Access::Flags kAccess>
inline vTensor::Future<Type, kAccess>::Future(
    vTensor* const tensor)
  : tensor_(tensor) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      tensor_,
      "Invalid Vulkan tensor!");
}

template<typename Type, vTensor::Access::Flags kAccess>
inline vTensor::Future<Type, kAccess>::Future(
    Future&& future)
  : tensor_(std::move(future.tensor_)) {
  future.tensor_ = nullptr;
}

template<typename Type, vTensor::Access::Flags kAccess>
inline vTensor::Future<Type, kAccess>&
vTensor::Future<Type, kAccess>::operator=(
    Future&& future) & {
  tensor_ = std::move(future.tensor_);
  future.tensor_ = nullptr;
  return *this;
}

template<typename Type, vTensor::Access::Flags kAccess>
template<typename Type_, vTensor::Access::Flags kAccess_, typename>
inline vTensor::Future<Type, kAccess>::Future(
    Future<Type_, kAccess_>&& future)
  : tensor_(std::move(future.tensor_)) {
  future.tensor_ = nullptr;
}

template<typename Type, vTensor::Access::Flags kAccess>
template<typename Type_, vTensor::Access::Flags kAccess_, typename>
inline vTensor::Future<Type, kAccess>&
vTensor::Future<Type, kAccess>::operator=(
    Future<Type_, kAccess_>&& future) & {
  tensor_ = std::move(future.tensor_);
  future.tensor_ = nullptr;
  return *this;
}

template<typename Type, vTensor::Access::Flags kAccess>
inline vTensor::Future<Type, kAccess>::~Future() {
  // Sync eagerly in an effort to hide latency.
  if (Access::Write & kAccess) {
    tensor_->view_.transition({
      View::Component::Image,
      vTensor::Access::Read,
    });
  }
}

template<typename Type, vTensor::Access::Flags kAccess>
inline typename vTensor::Future<Type, kAccess>::Payload
vTensor::Future<Type, kAccess>::wait() const & {
  TORCH_CHECK(
      tensor_,
      "vTensor::Future is in an invalid state!  "
      "Potential reason: This future is moved from.");

  return tensor_->wait().template map<Type, kAccess>();
}

template<typename Type>
inline vTensor::Future<Type, vTensor::Access::Read> vTensor::host() const & {
  return Future<Type, vTensor::Access::Read>(host());
}

template<typename Type, vTensor::Access::Flags kAccess>
inline vTensor::Future<Type, kAccess> vTensor::host() & {
  return Future<Type, kAccess>(host(kAccess));
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
