#pragma once

#include <utility>

#define TORCH_ARG(T, name)                                       \
 public:                                                         \
  inline auto name(const T& new_##name)->decltype(*this) { /* NOLINT */ \
    this->name##_ = new_##name;                                  \
    return *this;                                                \
  }                                                              \
  inline auto name(T&& new_##name)->decltype(*this) { /* NOLINT */      \
    this->name##_ = std::move(new_##name);                       \
    return *this;                                                \
  }                                                              \
  inline const T& name() const noexcept { /* NOLINT */                  \
    return this->name##_;                                        \
  }                                                              \
  inline T& name() noexcept { /* NOLINT */                       \
    return this->name##_;                                        \
  }                                                              \
 private:                                                        \
  T name##_ /* NOLINT */
