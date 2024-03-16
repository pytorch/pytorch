#pragma once
// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName
#ifdef USE_VULKAN_API

#include <exception>
#include <ostream>
#include <string>
#include <vector>

#include <ATen/native/vulkan/api/StringUtil.h>
#include <ATen/native/vulkan/api/vk_api.h>

#define VK_CHECK(function)                                       \
  do {                                                           \
    const VkResult result = (function);                          \
    if (VK_SUCCESS != result) {                                  \
      throw ::at::native::vulkan::api::Error(                    \
          {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, \
          ::at::native::vulkan::api::concat_str(                 \
              #function, " returned ", result));                 \
    }                                                            \
  } while (false)

#define VK_CHECK_COND(cond, ...)                                 \
  do {                                                           \
    if (!(cond)) {                                               \
      throw ::at::native::vulkan::api::Error(                    \
          {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, \
          #cond,                                                 \
          ::at::native::vulkan::api::concat_str(__VA_ARGS__));   \
    }                                                            \
  } while (false)

#define VK_THROW(...)                                          \
  do {                                                         \
    throw ::at::native::vulkan::api::Error(                    \
        {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, \
        ::at::native::vulkan::api::concat_str(__VA_ARGS__));   \
  } while (false)

namespace at {
namespace native {
namespace vulkan {
namespace api {

std::ostream& operator<<(std::ostream& out, const VkResult loc);

struct SourceLocation {
  const char* function;
  const char* file;
  uint32_t line;
};

std::ostream& operator<<(std::ostream& out, const SourceLocation& loc);

class Error : public std::exception {
 public:
  Error(SourceLocation source_location, std::string msg);
  Error(SourceLocation source_location, const char* cond, std::string msg);

 private:
  std::string msg_;
  SourceLocation source_location_;
  std::string what_;

 public:
  const std::string& msg() const {
    return msg_;
  }

  const char* what() const noexcept override {
    return what_.c_str();
  }
};

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
