#pragma once

#include "lazy_tensors/span.h"
#include "lazy_tensors/types.h"

namespace lazy_tensors {

class Tile {};

class Layout {
 public:
  lazy_tensors::Span<const int64> minor_to_major() const {
    return minor_to_major_;
  }

  Layout& add_minor_to_major(int64 value) {
    minor_to_major_.push_back(value);
    return *this;
  }

 private:
  std::vector<int64> minor_to_major_;
};

}  // namespace lazy_tensors
