#pragma once

#ifdef USE_VULKAN_API

#include <exception>

namespace at {
namespace native {
namespace vulkan {

/*
 * Same as c10::SourceLocation, represents a location in source code
 */
struct SourceLocation {
  const char* func;
  const char* file;
  uint32_t line;
};

std::ostream& operator<<(std::ostream& out, const SourceLocation& loc);

/*
 * Simple error class modeled after c10::Error
 */
class Error : public std::exception {
 public:
  // Constructors
  Error(SourceLocation location, std::string msg);
  Error(std::string msg, std::string backtrace);

 private:
  // The actual error message
  std::string msg_;
  // The source location of the exception
  std::string backtrace_;

 public:
  const char* what() const noexcept override {
    return msg_.c_str();
  }

  const std::string& msg() const {
    return msg_;
  }

 private:
  void refresh_what();
  std::string compute_what(bool include_backtrace) const;
};

} // namespace vulkan
} // namespace native
} // namespace at

#define VKGRAPH_THROW(...) \
  throw ::at::native::vulkan::Error(__VA_ARGS__);

#endif /* USE_VULKAN_API */
