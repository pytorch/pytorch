#include <ATen/native/vulkan/graph/Types.h>

namespace at {
namespace native {
namespace vulkan {

std::ostream& operator<<(std::ostream& out, const TypeTag& tag) {
  switch (tag) {
    case TypeTag::NONE:
      out << "NONE";
      break;
    case TypeTag::TENSOR:
      out << "TENSOR";
      break;
    case TypeTag::STAGING:
      out << "STAGING";
      break;
    default:
      out << "UNKNOWN";
      break;
  }
  return out;
}

} // namespace vulkan
} // namespace native
} // namespace at
