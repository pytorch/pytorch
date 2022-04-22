#include <ATen/core/List.h>

namespace c10 {
namespace detail {
bool operator==(const ListImpl& lhs, const ListImpl& rhs) {
  return *lhs.elementType == *rhs.elementType &&
      lhs.list.size() == rhs.list.size() &&
      // see: [container equality]
      std::equal(
          lhs.list.cbegin(),
          lhs.list.cend(),
          rhs.list.cbegin(),
          _fastEqualsForContainer);
}
} // namespace detail
} // namespace c10
