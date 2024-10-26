#pragma once

namespace c10 {

template <typename T>
class OptionalRef {
 public:
  OptionalRef() : data_(nullptr) {}
  OptionalRef(const T* data) : data_(data) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(data_);
  }
  OptionalRef(const T& data) : data_(&data) {}

  bool has_value() const {
    return data_ != nullptr;
  }

  const T& get() const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(data_);
    return *data_;
  }

  operator bool() const {
    return has_value();
  }

 private:
  const T* data_;
};

} // namespace c10
