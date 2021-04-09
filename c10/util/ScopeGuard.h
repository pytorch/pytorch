#pragma once

#include <functional>

namespace c10 {

class ScopeGuard {
 public:
  explicit ScopeGuard(std::function<void()> cb) : callback(std::move(cb)) {}

  // Disallow copy, move, assignment.
  ScopeGuard(const ScopeGuard &) = delete;
  ScopeGuard &operator=(ScopeGuard &&) = delete;
  ScopeGuard &operator=(const ScopeGuard &) = delete;

  ~ScopeGuard() {
    callback();
  }

 private:
  std::function<void()> callback;
};

}
