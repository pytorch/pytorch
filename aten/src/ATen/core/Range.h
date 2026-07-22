#pragma once

#include <cstdint>
#include <iosfwd>

namespace at {

struct Range {
  Range(int64_t begin, int64_t end)
    : begin(begin)
    , end(end) {}

  int64_t size() const { return end - begin; }

  Range operator/(int64_t divisor) {
    return Range(begin / divisor, end / divisor);
  }

  int64_t begin;
  int64_t end;
};

std::ostream& operator<<(std::ostream& out, const Range& range);

}  // namespace at
