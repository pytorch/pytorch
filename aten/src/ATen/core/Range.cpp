#include <ATen/core/Range.h>

#include <iostream>

namespace at {

std::ostream& operator<<(std::ostream& out, const Range& range) {
  out << "Range[" << range.begin << ", " << range.end << "]";
  return out;
}

}  // namespace at
