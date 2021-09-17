#pragma once

#include <cstddef>

namespace torch {
namespace jit {
namespace mobile {

struct Code;

class Frame {
 public:
  Frame(const Code& code) : code_(code) {}
  const Code& getCode() const {
    return code_;
  }

  void step() {
    pc_++;
  }

  void jump(size_t n) {
    pc_ += n;
  }

  size_t getPC() const {
    return pc_;
  }

 private:
  const Code& code_;
  size_t pc_{0};
};

} // namespace mobile
} // namespace jit
} // namespace torch
