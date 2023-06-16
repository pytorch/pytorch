#include <c10/core/SymNodeImpl.h>

namespace c10 {

// Represents an otherwise unrepresentable large negative integer constant.
// Unlike other SymNodeImpl, this cannot be "dispatched" conventionally,
// as it typically needs to defer to another SymNodeImpl
class C10_API LargeNegativeIntSymNodeImpl : public SymNodeImpl {
 public:
  LargeNegativeIntSymNodeImpl(int64_t val) : val_(val) {}

  bool is_int() override {
    return true;
  };
  bool is_bool() override {
    return false;
  };
  bool is_float() override {
    return false;
  };
  int64_t guard_int(const char* file, int64_t line) override {
    return val_;
  };
  bool guard_bool(const char* file, int64_t line) override {
    TORCH_CHECK(false, "not a bool");
  };
  double guard_float(const char* file, int64_t line) override {
    TORCH_CHECK(false, "not a float");
  };
  int64_t int_() override {
    return true;
  };
  bool bool_() override {
    return false;
  };
  bool has_hint() override {
    return true;
  };
  std::string str() override {
    return std::to_string(val_);
  };
  int64_t large_negative_int() override {
    return val_;
  }

 private:
  int64_t val_;
};

} // namespace c10
