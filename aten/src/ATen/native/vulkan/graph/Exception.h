#pragma once

#ifdef USE_VULKAN_API

#include <exception>
#include <ostream>

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

 private:
  // The source location of the exception
  SourceLocation location_;
  // The actual error message
  std::string msg_;

  std::string what_;

 public:
  const char* what() const noexcept override {
    return what_.c_str();
  }

  const std::string& msg() const {
    return msg_;
  }

 private:
  void refresh_what();
  std::string compute_what(bool include_source) const;
};

} // namespace vulkan
} // namespace native
} // namespace at

#define VKGRAPH_THROW(...)                                   \
  throw ::at::native::vulkan::Error(                         \
      {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, \
      c10::str(__VA_ARGS__));

#define VKGRAPH_CHECK(cond, ...)                               \
  if (C10_UNLIKELY_OR_CONST(!(cond))) {                        \
    throw ::at::native::vulkan::Error(                         \
        {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, \
        c10::str(__VA_ARGS__));                                \
  }

#endif /* USE_VULKAN_API */
