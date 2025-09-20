#pragma once

#include <utility>

// CLEAN COMPILE-TIME: Auto field ID generation
#define TORCH_FIELD_ID() (__COUNTER__)

// Original TORCH_ARG (no tracking)
#define TORCH_ARG(T, name)                                                \
 public:                                                                  \
  inline auto name(const T& new_##name) -> decltype(*this) { /* NOLINT */ \
    this->name##_ = new_##name;                                           \
    return *this;                                                         \
  }                                                                       \
  inline auto name(T&& new_##name) -> decltype(*this) { /* NOLINT */      \
    this->name##_ = std::move(new_##name);                                \
    return *this;                                                         \
  }                                                                       \
  inline const T& name() const noexcept { /* NOLINT */                    \
    return this->name##_;                                                 \
  }                                                                       \
  inline T& name() noexcept { /* NOLINT */                                \
    return this->name##_;                                                 \
  }                                                                       \
                                                                          \
 private:                                                                 \
  T name##_ /* NOLINT */

// CLEAN COMPILE-TIME: TORCH_ARG with field tracking
#define TORCH_ARG_WITH_TRACKING(T, name, field_id)                        \
 public:                                                                  \
  static constexpr size_t name##_FIELD_ID = field_id;                     \
  inline auto name(const T& new_##name) -> decltype(*this) { /* NOLINT */ \
    this->name##_ = new_##name;                                           \
    this->mark_field_as_set(field_id);                                    \
    return *this;                                                         \
  }                                                                       \
  inline auto name(T&& new_##name) -> decltype(*this) { /* NOLINT */      \
    this->name##_ = std::move(new_##name);                                \
    this->mark_field_as_set(field_id);                                    \
    return *this;                                                         \
  }                                                                       \
  inline const T& name() const noexcept { /* NOLINT */                    \
    return this->name##_;                                                 \
  }                                                                       \
  inline T& name() noexcept { /* NOLINT */                                \
    return this->name##_;                                                 \
  }                                                                       \
                                                                          \
 private:                                                                 \
  T name##_ /* NOLINT */
