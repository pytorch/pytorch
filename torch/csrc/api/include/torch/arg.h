#pragma once

#include <utility>

#define TORCH_ARG(T, name)                                              \
 public:                                                                \
  inline auto name(const T& new_##name)->decltype(*this) { /* NOLINT */ \
    this->name##_ = new_##name;                                         \
    return *this;                                                       \
  }                                                                     \
  inline auto name(T&& new_##name)->decltype(*this) { /* NOLINT */      \
    this->name##_ = std::move(new_##name);                              \
    return *this;                                                       \
  }                                                                     \
  inline const T& name() const noexcept { /* NOLINT */                  \
    return this->name##_;                                               \
  }                                                                     \
  inline T& name() noexcept { /* NOLINT */                              \
    return this->name##_;                                               \
  }                                                                     \
                                                                        \
 private:                                                               \
  T name##_ /* NOLINT */

#define TORCH_ARG_WITH_CHECK(T, name, check_func)                       \
 public:                                                                \
  inline auto name(const T& new_##name)->decltype(*this) { /* NOLINT */ \
    check_func(new_##name);                                             \
    this->name##_ = new_##name;                                         \
    return *this;                                                       \
  }                                                                     \
  inline auto name(T&& new_##name)->decltype(*this) { /* NOLINT */      \
    check_func(new_##name);                                             \
    this->name##_ = std::move(new_##name);                              \
    return *this;                                                       \
  }                                                                     \
  inline const T& name() const noexcept { /* NOLINT */                  \
    return this->name##_;                                               \
  }                                                                     \
  inline T& name() noexcept { /* NOLINT */                              \
    return this->name##_;                                               \
  }                                                                     \
                                                                        \
 private:                                                               \
  T name##_ /* NOLINT */

// default arg check, the arg should be positive
#define TORCH_ARG_WITH_DEFAULT_CHECK(T, name)                            \
  TORCH_ARG_WITH_CHECK(T, name, [&](const T& arg) {                      \
    if (arg <= T(0)) {                                                   \
      TORCH_CHECK(                                                       \
          false,                                                         \
          #name " should be positive, and the dtype should be " #T "."); \
    }                                                                    \
  })
