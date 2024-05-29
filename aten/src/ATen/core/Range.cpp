#include <ATen/core/Range.h>

#include <ostream>

namespace at {

std::ostream& operator<<(std::ostream& out, const Range& range) {
  out << "Range[" << range.begin << ", " << range.end << "]";
  return out;
}

}  // namespace at
