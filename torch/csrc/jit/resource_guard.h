#pragma once
#include <functional>

namespace torch::jit {

class ResourceGuard {
  std::function<void()> _destructor;
  bool _released{false};

 public:
  ResourceGuard(std::function<void()> destructor)
      : _destructor(std::move(destructor)) {}

  // NOLINTNEXTLINE(bugprone-exception-escape)
  ~ResourceGuard() {
    if (!_released)
      _destructor();
  }

  void release() {
    _released = true;
  }
};

} // namespace torch::jit
