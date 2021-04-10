#pragma once

#include <functional>

namespace c10 {

/**
 * Basic ScopeGuard class to allow executing a custom lambda on scope exit.
 */
class ScopeGuard {
public:
  explicit ScopeGuard(std::function<void()> cb) : callback(std::move(cb)) {}

  // Disallow copy, move, assignment, and move assignment.
  ScopeGuard(const ScopeGuard &) = delete;
  ScopeGuard(ScopeGuard &&) = delete;
  ScopeGuard &operator=(ScopeGuard &&) = delete;
  ScopeGuard &operator=(const ScopeGuard &) = delete;

  void disallow() {
    this->allowed = false;
  }

  ~ScopeGuard() {
    if (allowed) {
      callback();
    }
  }

private:
  bool allowed = true;
  std::function<void()> callback;
};

}
