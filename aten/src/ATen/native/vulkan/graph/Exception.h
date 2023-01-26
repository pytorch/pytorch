#pragma once

#ifdef USE_VULKAN_API

#include <exception>

namespace at {
namespace native {
namespace vulkan {

/*
 * Simple error class modeled after c10::Error
 */
class Error : public std::exception {
 public:
  // Constructor
  Error(std::string msg) : msg_(std::move(msg)) {}

 private:
  std::string msg_;

 public:
  const char* what() const noexcept override {
    return msg_.c_str();
  }

  const std::string& msg() const {
    return msg_;
  }
};

} // namespace vulkan
} // namespace native
} // namespace at

#define VKGRAPH_THROW(...) \
  throw ::at::native::vulkan::Error(__VA_ARGS__);

#endif /* USE_VULKAN_API */
