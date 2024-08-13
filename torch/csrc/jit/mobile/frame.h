#pragma once

#include <cstddef>

#include <torch/csrc/jit/mobile/code.h>
#include <optional>

namespace torch::jit::mobile {

class Frame {
 public:
  explicit Frame(const Code& code) : code_(code) {}
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

  const Instruction& getInstruction() const {
    return code_.instructions_.at(pc_);
  }

  std::optional<int64_t> getDebugHandle() const {
    return getDebugHandle(pc_);
  }

  std::optional<int64_t> getDebugHandle(size_t pc) const {
    if (pc >= code_.debug_handles_.size()) {
      return {};
    }
    return code_.debug_handles_[pc];
  }

 private:
  const Code& code_;
  size_t pc_{0};
};

} // namespace torch::jit::mobile
