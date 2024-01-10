#include <ATen/native/vulkan/api/Exception.h>

#include <sstream>

namespace at {
namespace native {
namespace vulkan {
namespace api {

#define VK_RESULT_CASE(code) \
  case code:                 \
    out << #code;            \
    break;

std::ostream& operator<<(std::ostream& out, const VkResult result) {
  switch (result) {
    VK_RESULT_CASE(VK_SUCCESS)
    VK_RESULT_CASE(VK_NOT_READY)
    VK_RESULT_CASE(VK_TIMEOUT)
    VK_RESULT_CASE(VK_EVENT_SET)
    VK_RESULT_CASE(VK_EVENT_RESET)
    VK_RESULT_CASE(VK_INCOMPLETE)
    VK_RESULT_CASE(VK_ERROR_OUT_OF_HOST_MEMORY)
    VK_RESULT_CASE(VK_ERROR_OUT_OF_DEVICE_MEMORY)
    VK_RESULT_CASE(VK_ERROR_INITIALIZATION_FAILED)
    VK_RESULT_CASE(VK_ERROR_DEVICE_LOST)
    VK_RESULT_CASE(VK_ERROR_MEMORY_MAP_FAILED)
    VK_RESULT_CASE(VK_ERROR_LAYER_NOT_PRESENT)
    VK_RESULT_CASE(VK_ERROR_EXTENSION_NOT_PRESENT)
    VK_RESULT_CASE(VK_ERROR_FEATURE_NOT_PRESENT)
    VK_RESULT_CASE(VK_ERROR_INCOMPATIBLE_DRIVER)
    VK_RESULT_CASE(VK_ERROR_TOO_MANY_OBJECTS)
    VK_RESULT_CASE(VK_ERROR_FORMAT_NOT_SUPPORTED)
    VK_RESULT_CASE(VK_ERROR_FRAGMENTED_POOL)
    VK_RESULT_CASE(VK_ERROR_UNKNOWN)
    VK_RESULT_CASE(VK_ERROR_OUT_OF_POOL_MEMORY)
    VK_RESULT_CASE(VK_ERROR_INVALID_EXTERNAL_HANDLE)
    VK_RESULT_CASE(VK_ERROR_FRAGMENTATION)
    VK_RESULT_CASE(VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS)
    VK_RESULT_CASE(VK_PIPELINE_COMPILE_REQUIRED_EXT)
    VK_RESULT_CASE(VK_ERROR_SURFACE_LOST_KHR)
    VK_RESULT_CASE(VK_ERROR_NATIVE_WINDOW_IN_USE_KHR)
    VK_RESULT_CASE(VK_SUBOPTIMAL_KHR)
    VK_RESULT_CASE(VK_ERROR_OUT_OF_DATE_KHR)
    VK_RESULT_CASE(VK_ERROR_INCOMPATIBLE_DISPLAY_KHR)
    VK_RESULT_CASE(VK_ERROR_VALIDATION_FAILED_EXT)
    VK_RESULT_CASE(VK_ERROR_INVALID_SHADER_NV)
    VK_RESULT_CASE(VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT)
    VK_RESULT_CASE(VK_ERROR_NOT_PERMITTED_EXT)
    VK_RESULT_CASE(VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT)
    VK_RESULT_CASE(VK_THREAD_IDLE_KHR)
    VK_RESULT_CASE(VK_THREAD_DONE_KHR)
    VK_RESULT_CASE(VK_OPERATION_DEFERRED_KHR)
    VK_RESULT_CASE(VK_OPERATION_NOT_DEFERRED_KHR)
    VK_RESULT_CASE(VK_RESULT_MAX_ENUM)
    default:
      out << "VK_UNKNOWN";
      break;
  }
  return out;
}

#undef VK_RESULT_CASE

//
// SourceLocation
//

std::ostream& operator<<(std::ostream& out, const SourceLocation& loc) {
  out << loc.function << " at " << loc.file << ":" << loc.line;
  return out;
}

//
// Exception
//

Error::Error(SourceLocation source_location, std::string msg)
    : msg_(std::move(msg)), source_location_{source_location} {
  std::ostringstream oss;
  oss << "Exception raised from " << source_location_ << ": ";
  oss << msg_;
  what_ = oss.str();
}

Error::Error(SourceLocation source_location, const char* cond, std::string msg)
    : msg_(std::move(msg)), source_location_{source_location} {
  std::ostringstream oss;
  oss << "Exception raised from " << source_location_ << ": ";
  oss << "(" << cond << ") is false! ";
  oss << msg_;
  what_ = oss.str();
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
