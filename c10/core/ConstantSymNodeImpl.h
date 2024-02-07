#pragma once

#include <c10/core/SymNodeImpl.h>
#include <c10/macros/Export.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <cstdint>
#include <string>
#include <variant>

namespace c10 {

// Unlike other SymNodeImpl, this cannot be "dispatched" conventionally,
// as it typically needs to defer to another SymNodeImpl
//
// Can either represent a bool, int (don't support float yet) this is useful
// for representing otherwise unrepresentable large negative integer constant.
template <typename T>
class C10_API ConstantSymNodeImpl : public SymNodeImpl {
  static_assert(
      ::std::is_same_v<T, int64_t> || ::std::is_same_v<T, bool>,
      "ConstantSymNodeImpl can only accept int64_t or bool types");

 public:
  ConstantSymNodeImpl(T val) : value_(val) {}

  bool is_int() override {
    return is_int_();
  }
  bool is_bool() override {
    return is_bool_();
  }
  bool is_float() override {
    return false;
  }
  int64_t guard_int(const char* file, int64_t line) override {
    TORCH_CHECK(is_int(), "not an int");
    return int_();
  }
  bool guard_bool(const char* file, int64_t line) override {
    TORCH_CHECK(is_bool(), "not a bool");
    return bool_();
  }
  double guard_float(const char* file, int64_t line) override {
    TORCH_CHECK(false, "not a float");
  }
  int64_t int_() override {
    TORCH_CHECK(is_int(), "not an int");
    return ::std::get<int64_t>(value_);
  }
  bool bool_() override {
    TORCH_CHECK(is_bool(), "not a bool");
    return ::std::get<bool>(value_);
  }
  bool has_hint() override {
    return true;
  }
  c10::SymNode eq(const c10::SymNode& other) override;
  c10::SymNode ne(const c10::SymNode& other) override;
  c10::SymNode ge(const c10::SymNode& other) override;
  c10::SymNode le(const c10::SymNode& other) override;
  c10::SymNode lt(const c10::SymNode& other) override;
  c10::SymNode gt(const c10::SymNode& other) override;
  c10::SymNode mul(const c10::SymNode& other) override;
  ::std::string str() override {
    if constexpr (is_int_()) {
      return ::std::to_string(::std::get<int64_t>(value_));
    } else {
      return ::std::get<bool>(value_) ? "true" : "false";
    }
  }
  c10::optional<int64_t> constant_int() override {
    if constexpr (is_int_()) {
      return ::std::get<int64_t>(value_);
    } else {
      return c10::nullopt;
    }
  }
  c10::optional<bool> constant_bool() override {
    if constexpr (is_bool_()) {
      return ::std::get<bool>(value_);
    } else {
      return c10::nullopt;
    }
  }
  bool is_constant() override {
    return true;
  }
  bool is_symbolic() override {
    return false;
  }

 private:
  ::std::variant<int64_t, bool> value_;

  static constexpr bool is_int_() {
    return ::std::is_same_v<T, int64_t>;
  }
  static constexpr bool is_bool_() {
    return ::std::is_same_v<T, bool>;
  }
};

} // namespace c10
