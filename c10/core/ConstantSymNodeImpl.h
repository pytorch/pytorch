#include <c10/core/SymNodeImpl.h>
#include <c10/util/variant.h>

namespace c10 {

// Unlike other SymNodeImpl, this cannot be "dispatched" conventionally,
// as it typically needs to defer to another SymNodeImpl
//
// Can either represent a bool, int (don't support float yet) this is useful
// for representing otherwise unrepresentable large negative integer constant.
template <typename T>
class C10_API ConstantSymNodeImpl : public SymNodeImpl {
  static_assert(
      std::is_same<T, int64_t>::value || std::is_same<T, bool>::value,
      "ConstantSymNodeImpl can only accept int64_t or bool types");

 public:
  ConstantSymNodeImpl(T val) : value_(val) {}

  bool is_int() override {
    return std::is_same<T, int64_t>::value;
  }
  bool is_bool() override {
    return std::is_same<T, bool>::value;
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
    return c10::get<int64_t>(value_);
  }
  bool bool_() override {
    TORCH_CHECK(is_bool(), "not a bool");
    return c10::get<bool>(value_);
  }
  bool has_hint() override {
    return true;
  }
  c10::SymNode eq(const c10::SymNode& other) override;
  c10::SymNode ne(const c10::SymNode& other) override;
  std::string str() override {
    if (is_int()) {
      return std::to_string(c10::get<int64_t>(value_));
    } else {
      return c10::get<bool>(value_) ? "true" : "false";
    }
  }
  c10::optional<int64_t> constant_int() override {
    if (is_int()) {
      return c10::get<int64_t>(value_);
    } else {
      return c10::nullopt;
    }
  }
  c10::optional<bool> constant_bool() override {
    if (is_bool()) {
      return c10::get<bool>(value_);
    } else {
      return c10::nullopt;
    }
  }

 private:
  c10::variant<int64_t, bool> value_;
};

} // namespace c10
