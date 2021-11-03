#include <c10/core/Allocator.h>

namespace c10 {

template <typename T>
class C10_API DeviceArray {
 public:
  DeviceArray(c10::Allocator& allocator, size_t size)
      : data_ptr_(allocator.allocate(size * sizeof(T))) {
    static_assert(std::is_pod<T>::value, "T must be 'plain old data'");
  }

  T* get() {
    return static_cast<T*>(data_ptr_.get());
  }

 private:
  c10::DataPtr data_ptr_;
};

} // namespace c10
