#pragma once
#include <functional>

namespace torch {
namespace jit {

class ResourceGuard {
  std::function<void()> _destructor;
  bool _released;

 public:
  ResourceGuard(std::function<void()> destructor)
      : _destructor(std::move(destructor)), _released(false) {}

  // NOLINTNEXTLINE(bugprone-exception-escape)
  ~ResourceGuard() {
    if (!_released)
      _destructor();
  }

  void release() {
    _released = true;
  }
};

} // namespace jit
} // namespace torch
