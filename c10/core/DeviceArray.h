#include <c10/core/Allocator.h>
#include <c10/util/Exception.h>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace c10 {

template <typename T>
class DeviceArray {
 public:
  DeviceArray(c10::Allocator& allocator, size_t size)
      : data_ptr_(allocator.allocate(size * sizeof(T))) {
    static_assert(std::is_trivial_v<T>, "T must be a trivial type");
    TORCH_INTERNAL_ASSERT(
        0 == (reinterpret_cast<intptr_t>(data_ptr_.get()) % alignof(T)),
        "c10::DeviceArray: Allocated memory is not aligned for this data type");
  }

  T* get() {
    return static_cast<T*>(data_ptr_.get());
  }

 private:
  c10::DataPtr data_ptr_;
};

} // namespace c10
