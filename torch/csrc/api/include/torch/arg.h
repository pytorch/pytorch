#pragma once

#include <utility>

#define TORCH_ARG(T, name)                                       \
  auto name(const T& new_##name)->decltype(*this) { /* NOLINT */ \
    this->name##_ = new_##name;                                  \
    return *this;                                                \
  }                                                              \
  auto name(T&& new_##name)->decltype(*this) { /* NOLINT */      \
    this->name##_ = std::move(new_##name);                       \
    return *this;                                                \
  }                                                              \
  const T& name() const noexcept { /* NOLINT */                  \
    return this->name##_;                                        \
  }                                                              \
  T name##_ /* NOLINT */
