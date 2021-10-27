#pragma once

#include <stdint.h>

#include <vector>

namespace lazy_tensors {

class Tile {};

class Layout {
 public:
  int64_t minor_to_major(int index) const { return minor_to_major_.at(index); }

  const std::vector<int64_t>& minor_to_major() const { return minor_to_major_; }

  std::vector<int64_t>* mutable_minor_to_major() { return &minor_to_major_; }

  Layout& add_minor_to_major(int64_t value) {
    minor_to_major_.push_back(value);
    return *this;
  }

 private:
  std::vector<int64_t> minor_to_major_;
};

}  // namespace lazy_tensors
