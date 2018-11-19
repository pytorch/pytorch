#pragma once

#include <stdexcept>

namespace torch {
namespace jit {

struct  JITException
    : public std::runtime_error {
  JITException() = default;
  explicit JITException(const std::string& msg)
      : std::runtime_error(msg) {}
};

} // namespace jit
} // namespace torch
