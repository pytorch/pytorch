#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <utility>

#define TORCH_ARG(T, name)                                       \
 public:                                                         \
  TORCH_API inline auto name(const T& new_##name)->decltype(*this) { /* NOLINT */ \
    this->name##_ = new_##name;                                  \
    return *this;                                                \
  }                                                              \
  TORCH_API inline auto name(T&& new_##name)->decltype(*this) { /* NOLINT */      \
    this->name##_ = std::move(new_##name);                       \
    return *this;                                                \
  }                                                              \
  TORCH_API inline const T& name() const noexcept { /* NOLINT */                  \
    return this->name##_;                                        \
  }                                                              \
 private:                                                        \
  T name##_ /* NOLINT */
